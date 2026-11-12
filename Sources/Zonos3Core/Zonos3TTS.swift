// Zonos3TTS.swift — the pipeline: prompt rows, generation, incremental decode.
// Prompt layout and decode rules are ports of zonos3/tts/plan.py + prompt.py (speaker row, condition rows, text rows,
// codec-matched silence prefill) and zonos3/tts/stream.py (`PrefixDecoder`, `undelay_window`) @ 4c66cea (Zyphra, MIT).
// First cut: the 192-d embedding voice (or none). Not ported: raw-reference prefix, speaker CFG, LACI, steering,
// accents, editing, text normalization (text is tokenized as written).
import Foundation
import MLX

/// Raw values of the three acoustic condition rows; nil = the table's "unknown" row.
public struct Conditions: Sendable {
    public var loudnessLUFS: Double?
    public var fmaxHz: Double?
    public var speakingRate: Double?  // UTF-8 bytes per second
    public init(loudnessLUFS: Double? = nil, fmaxHz: Double? = nil, speakingRate: Double? = nil) {
        self.loudnessLUFS = loudnessLUFS; self.fmaxHz = fmaxHz; self.speakingRate = speakingRate
    }
}

public struct SynthesisOptions: Sendable {
    public var sampling = Sampling()
    public var conditions = Conditions()
    public var seed: UInt64 = 0
    public var maxFrames = 2000
    public var silencePrefillFrames = 3
    public var headroomDB: Double? = nil
    /// Early partial-frame decode of the first chunk: a frame is decoded once its first `streamCodebooks` codebooks
    /// arrived (n_codebooks = complete frames only).
    public var streamCodebooks = 1
    public var chunkFrames = 15
    public var firstChunkFrames = 1
    public init() {}
}

public struct SynthesisResult: Sendable {
    public let pcm: [Float]
    public let sampleRate: Int
    public let generatedRows: Int
    public let eosFrame: Int?
    public let hitFrameLimit: Bool
    public let firstAudioSeconds: Double
    public let totalSeconds: Double
}

/// One request's incremental decode (zonos3/tts/stream.py `StreamDecoder` in prefix mode + `PrefixDecoder`): the stream
/// decodes [silence-prefill frames | generated frames] from frame 0 and drops the prefill samples.
final class PrefixStream {
    let stream: CodecStream
    let nCodebooks, pad, eoa, hop, prefill, chunk, firstChunk, codebooks: Int
    var buffer: [[Int32]]
    var emitted: Int
    var produced = 0
    var skip: Int
    var eos: Int? = nil

    init(stream: CodecStream, nCodebooks: Int, audioPadID: Int, hop: Int, prefillRows: [[Int32]], chunk: Int, firstChunk: Int, codebooks: Int) {
        self.stream = stream; self.nCodebooks = nCodebooks; pad = audioPadID; eoa = audioPadID - 1; self.hop = hop
        buffer = prefillRows; prefill = prefillRows.count; emitted = prefill
        self.chunk = chunk; self.firstChunk = firstChunk; self.codebooks = codebooks
        skip = prefill * hop
    }

    private var target: Int {
        let t = buffer.count - (nCodebooks - 1)
        return eos.map { min(t, $0) } ?? t
    }
    private var need: Int { min(chunk, firstChunk + emitted - prefill) }

    /// Frames decodable now: the complete ones, or with early decode (first chunk only) every frame whose first
    /// `codebooks` codebooks arrived, before any frame whose semantic code is not a code.
    private func ready(_ complete: Int) -> Int {
        if codebooks >= nCodebooks || emitted > prefill { return complete }
        var r = buffer.count - (codebooks - 1)
        if let eos { r = min(r, eos) }
        while r > max(complete, emitted) && Int(buffer[r - 1][0]) >= pad - 1 { r -= 1 }
        return r
    }

    /// Undelayed frames [start, end) of the delayed rows (row r holds codebook j of frame r - j).
    private func undelay(_ start: Int, _ end: Int) -> [[Int32]] {
        let rawEnd = min(end + nCodebooks - 1, buffer.count)
        return (start..<end).map { f in
            (0..<nCodebooks).map { j in f + j < rawEnd ? buffer[f + j][j] : Int32(pad) }
        }
    }

    private func decode(to end: Int) throws -> [Float] {
        let start = stream.frames == 0 ? 0 : emitted  // the first call decodes the prefill rows too
        var codes = undelay(start, end)
        if codebooks < nCodebooks {
            for i in codes.indices { for j in 1..<nCodebooks where Int(codes[i][j]) == pad { codes[i][j] = -1 } }
        }
        emitted = end
        return out(try stream.decode(codes))
    }

    private func out(_ raw: [Float]) -> [Float] {
        produced += raw.count
        let drop = min(raw.count, skip)
        skip -= drop
        return drop == 0 ? raw : Array(raw[drop...])
    }

    /// One generated row (`eosFrame`: the aligned EOS frame in generated rows) -> the PCM it completes (maybe none).
    func push(_ row: [Int32], eosFrame: Int?) throws -> [Float] {
        let codes = Array(row[0..<nCodebooks])
        buffer.append(codes)
        var candidates: [Int] = []
        if let eosFrame { candidates.append(max(0, eosFrame) + prefill) }
        if let top = (0..<nCodebooks).filter({ Int(codes[$0]) == eoa }).max() { candidates.append(max(0, buffer.count - 1 - top)) }
        if let observed = candidates.min() { eos = eos.map { min($0, observed) } ?? observed }
        let decodable = ready(target)
        if decodable - emitted < need { return [] }
        return try decode(to: decodable)
    }

    /// End of stream: frames up to the target, then the stream's flush (no samples past the target's end).
    func finish() throws -> [Float] {
        let t = target
        var parts: [Float] = t > emitted ? try decode(to: t) : []
        let tail = stream.flush()
        let cap = max(0, max(t, emitted) * hop - produced)
        parts += out(Array(tail.prefix(cap)))
        return parts
    }
}

public final class Zonos3TTS {
    public let lm: Zonos3LM
    public let decoder: CodecDecoder
    public let tokenizer: SuperBPETokenizer
    public private(set) var speakerEncoder: SpeakerEncoder?
    let checkpoint: CheckpointConfig
    public var sampleRate: Int { decoder.sampleRate }

    /// `directory`: a converted checkpoint (Tools/convert_gguf.py). `fullDecoder`: the `full_mr` head instead of
    /// `keep2_plain`. The speaker encoder loads on first use.
    public init(directory: URL, fullDecoder: Bool = false) throws {
        checkpoint = try CheckpointConfig(directory: directory)
        lm = try Zonos3LM(config: checkpoint)
        decoder = try CodecDecoder(config: checkpoint, file: fullDecoder ? "codec-full/model.safetensors" : "codec/model.safetensors")
        let sha = (try checkpoint.meta("lm/model.safetensors"))["tokenizer_sha256"] as? String
        tokenizer = try SuperBPETokenizer(url: directory.appendingPathComponent("tokenizer.json"), sha256: sha)
        guard lm.config.textVocab == 8192 + 4 else { throw Zonos3Error.invalid("SuperBPE ID convention mismatch") }
    }

    /// The 192-d speaker embedding of mono 44.1 kHz audio (at most its first 60 s are read).
    public func embedSpeaker(pcm44100: [Float]) throws -> [Float] {
        if speakerEncoder == nil { speakerEncoder = try SpeakerEncoder(config: checkpoint) }
        return try speakerEncoder!.embed(pcm44100: pcm44100)
    }

    /// Bucket id of a raw condition value: bucket i = [edge_{i-1}, edge_i) (bisect_right); nil = the unknown row.
    func bucket(_ value: Double?, _ name: String, _ table: Int) -> Int {
        guard let value, let edges = lm.config.conditionEdges[name] else { return lm.config.conditionBucketCounts[table] }
        return edges.filter { $0 <= value }.count
    }

    /// Prompt rows: [speaker row] [condition rows] [text rows] [silence prefill rows].
    public func prompt(text: String, speakerEmbedding: [Float]?, conditions: Conditions = Conditions(), silencePrefillFrames: Int = 3) throws -> (PromptRows, [[Int32]]) {
        let c = lm.config, nCB = c.nCodebooks
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw Zonos3Error.invalid("text is empty") }
        let slot = [Int32](repeating: Int32(c.audioPadID), count: nCB) + [Int32(c.textVocab)]
        var rows: [[Int32]] = []
        var speakerRows: [Int: MLXArray] = [:]
        if let e = speakerEmbedding {
            guard e.count == c.speakerEmbeddingDim else { throw Zonos3Error.invalid("speaker embedding must have \(c.speakerEmbeddingDim) values") }
            speakerRows[0] = MLXArray(e)
            rows.append(slot)
        }
        var conditionRows: [Int: Int] = [:]
        let values = [conditions.loudnessLUFS, conditions.fmaxHz, conditions.speakingRate]
        for (i, name) in c.conditionNames.enumerated() {
            conditionRows[rows.count] = bucket(i < values.count ? values[i] : nil, name, i)
            rows.append(slot)
        }
        for id in tokenizer.textIDs(text) { rows.append([Int32](repeating: Int32(c.audioPadID), count: nCB) + [Int32(id)]) }
        // codec-matched silence prefill: `frames` copies of the silence frame, sheared, without the flushed tail
        var prefill: [[Int32]] = []
        for r in 0..<max(0, silencePrefillFrames) {
            var row = slot
            for j in 0...min(r, nCB - 1) { row[j] = Int32(c.silenceFrame[j]) }
            prefill.append(row); rows.append(row)
        }
        let ids = MLXArray(rows.flatMap { $0 }, [rows.count, nCB + 1])
        return (PromptRows(ids: ids, speakerRows: speakerRows, conditionRows: conditionRows, prefillFrames: prefill.count), prefill)
    }

    /// Synthesize `text`. `onChunk` receives each PCM chunk as it becomes final (44.1 kHz mono, output gain and soft
    /// limiter applied); returning false stops the generation.
    /// `onRow` runs once per generated row (15 rows = 1 s of audio) with the row count; what it throws stops the
    /// generation and propagates unchanged (the host's progress and cancellation seam).
    public func synthesize(text: String, speakerEmbedding: [Float]?, options: SynthesisOptions = SynthesisOptions(),
                           onRow: ((Int) throws -> Void)? = nil, onChunk: (([Float]) -> Bool)? = nil) throws -> SynthesisResult {
        let c = lm.config
        let t0 = Date()
        let (prompt, prefill) = try self.prompt(text: text, speakerEmbedding: speakerEmbedding, conditions: options.conditions, silencePrefillFrames: options.silencePrefillFrames)
        let gen = try RowGenerator(lm: lm, prompt: prompt, sampling: options.sampling, maxTokens: options.maxFrames, seed: options.seed)
        let codebooks = decoder.earlyDecode ? min(max(options.streamCodebooks, 1), c.nCodebooks) : c.nCodebooks
        let ps = PrefixStream(stream: try decoder.stream(), nCodebooks: c.nCodebooks, audioPadID: c.audioPadID, hop: decoder.hop,
                              prefillRows: prefill.map { Array($0[0..<c.nCodebooks]) }, chunk: options.chunkFrames, firstChunk: options.firstChunkFrames, codebooks: codebooks)
        let gain = decoder.outputGain(headroomDB: options.headroomDB)
        var pcm: [Float] = []
        var first: Double? = nil
        var eos: Int? = nil
        var finished = false, stopped = false
        func emit(_ raw: [Float]) {
            guard !raw.isEmpty else { return }
            let out = outputPCM(raw, gain: gain)
            if first == nil { first = Date().timeIntervalSince(t0) }
            pcm += out
            if let onChunk, !onChunk(out) { stopped = true }
        }
        while !finished && !stopped {
            try Task.checkCancellation()
            let r = try gen.next()
            try onRow?(gen.generated)
            eos = r.eosFrame; finished = r.finished
            emit(try ps.push(r.row, eosFrame: r.eosFrame))
        }
        if !stopped { emit(try ps.finish()) }
        return SynthesisResult(pcm: pcm, sampleRate: decoder.sampleRate, generatedRows: gen.generated, eosFrame: eos,
                               hitFrameLimit: eos == nil && finished, firstAudioSeconds: first ?? 0, totalSeconds: Date().timeIntervalSince(t0))
    }
}
