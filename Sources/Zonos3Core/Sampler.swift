// Sampler.swift — the reference sampler on the MLX device and the generation loop.
// Ports of zonos3_mlx/sampling.py (`sample_row_mx`, `repetition_ids`), zonos3/tts/sampler.py (`clamp_sampling`) and the
// non-LACI, single-branch path of zonos3_mlx/scheduler.py (`_start`, `_sample`, `_tick`) @ 4c66cea (Zyphra, MIT).
//
// Upstream draws its exponential noise from a torch.Generator; this port draws it from MLXRandom, so a seed is
// reproducible here but does not reproduce upstream's rows. The parity gate injects upstream's noise instead.
import Foundation
import MLX
import MLXRandom

public struct Sampling: Sendable {
    public var temperature: Double = 0.75
    public var topK: Int = 80
    public var topP: Double = 0.69
    public var minP: Double = 0.0
    public var repetitionWindow: Int = 96
    public var repetitionPenalty: Double = 2.4
    public var repetitionCodebooks: Int = 6
    public var minFrames: Int = 0
    public init() {}

    /// `clamp_sampling`: topk < 1 = the whole codebook, repetition_codebooks < 0 = every codebook.
    func clamped(nCodebooks: Int, codebookSize: Int) -> Sampling {
        var s = self
        s.temperature = max(temperature, 1e-6)
        s.topK = topK >= 1 ? topK : codebookSize
        s.topP = min(max(topP, 0.0), 1.0)
        s.minP = max(minP, 0.0)
        s.repetitionWindow = max(repetitionWindow, 0)
        s.repetitionPenalty = max(repetitionPenalty, 1.0)
        s.repetitionCodebooks = repetitionCodebooks < 0 ? nCodebooks : min(max(repetitionCodebooks, 0), nCodebooks)
        return s
    }
    func activeRepetitionWindow(available: Int) -> Int {
        (repetitionWindow > 0 && repetitionPenalty > 1.0 && repetitionCodebooks > 0) ? min(repetitionWindow, available) : 0
    }
}

struct RowSampler {
    let nCodebooks, vocab, codebookSize, audioPadID, stopID, textVocab: Int
    let invalid: MLXArray  // [n_cb, vocab]: padding, special ids other than the stop id
    let vocabRange: MLXArray
    let cbRange: MLXArray

    init(config c: LMConfig, stopID: Int) {
        nCodebooks = c.nCodebooks; vocab = c.audioVocab; codebookSize = c.codebookSize
        audioPadID = c.audioPadID; self.stopID = stopID; textVocab = c.textVocab
        var mask = [Bool](repeating: false, count: nCodebooks * vocab)
        for cb in 0..<nCodebooks {
            for id in 0..<vocab where id == audioPadID || (id >= codebookSize && id != stopID) { mask[cb * vocab + id] = true }
        }
        invalid = MLXArray(mask, [nCodebooks, vocab])
        vocabRange = MLXArray(0..<Int32(vocab)).expandedDimensions(axis: 0)
        cbRange = MLXArray(0..<Int32(nCodebooks))
    }

    /// One sampled row [n_codebooks + 1] int32 (lazy). `logits` [n_cb, V]; `noise` [n_cb, width] exponential draws;
    /// `repetition`: [n_cb][ids] recent ordinary codes per penalised codebook; `stopBlocked`: codebooks whose stop id
    /// is masked (min_frames).
    func sample(_ logits: MLXArray, noise: MLXArray, _ s: Sampling, repetition: [[Int]]?, stopBlocked: [Bool]?, totalGenerated: Int, hasAudioPrefix: Bool) -> MLXArray {
        var bad = invalid
        if let stopBlocked {
            var blocked = [Bool](repeating: false, count: nCodebooks * vocab)
            for cb in 0..<nCodebooks where stopBlocked[cb] { blocked[cb * vocab + stopID] = true }
            bad = bad .|| MLXArray(blocked, [nCodebooks, vocab])
        }
        let negInf = MLXArray(-Float.infinity)
        var x = which(bad, negInf, logits.asType(.float32))
        if let repetition {
            var rep = [Bool](repeating: false, count: nCodebooks * vocab)
            for (cb, ids) in repetition.enumerated() { for id in ids { rep[cb * vocab + id] = true } }
            let penalty = Float(s.repetitionPenalty)
            x = which(MLXArray(rep, [nCodebooks, vocab]), which(x .> 0, x / penalty, x * penalty), x)
        }
        x = x / Float(s.temperature)
        if s.topK > 0 && s.topK < vocab {
            let kth = sorted(x, axis: -1)[0..., (vocab - s.topK)..<(vocab - s.topK + 1)]
            x = which(x .< kth, negInf, x)
        }
        var probs = softmax(x, axis: -1)
        if s.topP > 0.0 && s.topP < 1.0 {
            let order = argSort(-probs, axis: -1)
            var ranked = takeAlong(probs, order, axis: -1)
            ranked = which((cumsum(ranked, axis: -1) - ranked) .> Float(s.topP), MLXArray(Float(0)), ranked)
            probs = putAlong(MLXArray.zeros(like: probs), order, values: ranked, axis: -1)
            probs = probs / maximum(probs.sum(axis: -1, keepDims: true), MLXArray(Float(1e-8)))
        }
        if s.minP > 0.0 {
            probs = which(probs .< Float(s.minP) * probs.max(axis: -1, keepDims: true), MLXArray(Float(0)), probs)
            probs = probs / maximum(probs.sum(axis: -1, keepDims: true), MLXArray(Float(1e-8)))
        }
        // a row the filters emptied falls back to greedy over the filtered logits
        let greedy = vocabRange .== argMax(x, axis: -1, keepDims: true)
        probs = which(probs.sum(axis: -1, keepDims: true) .<= 0, greedy.asType(.float32), probs)
        let width = max(stopID, audioPadID) + 1
        var token = argMax(probs[0..., 0..<width] / noise, axis: -1).asType(.int32)
        if !hasAudioPrefix && totalGenerated < nCodebooks - 1 {
            token = which(cbRange .> Int32(totalGenerated), MLXArray(Int32(audioPadID)), token)
        }
        return concatenated([token, MLXArray([Int32(textVocab)])])
    }
}

/// One generated row and where the stream stands after it.
public struct GeneratedRow: Sendable {
    public let row: [Int32]  // [n_codebooks + 1], delayed
    public let eosFrame: Int?  // aligned end-of-audio frame in generated rows, once seen
    public let finished: Bool
}

/// The single-branch generation loop: prefill the prompt, then one row per `next()`.
public final class RowGenerator {
    let lm: Zonos3LM
    let sampler: RowSampler
    let sampling: Sampling
    let state = LMState()
    var logits: MLXArray  // last row's logits [n_cb, V]
    var history: [[Int32]] = []  // generated rows, audio columns
    let hasAudioPrefix: Bool
    var remain: Int
    var eosFrame = -1
    var countdown = -1
    var key: MLXArray
    /// Parity gates: noise per step [steps, n_cb, width] in place of the generator's own draws.
    public var injectedNoise: MLXArray? = nil
    /// Parity gates: teacher forcing — row k of [steps, n_cb + 1] replaces the sampled row as the next input.
    public var forcedRows: MLXArray? = nil
    public private(set) var generated = 0

    public init(lm: Zonos3LM, prompt: PromptRows, sampling: Sampling = Sampling(), maxTokens: Int = 2000, seed: UInt64 = 0) throws {
        self.lm = lm
        let c = lm.config
        sampler = RowSampler(config: c, stopID: c.eoaID)
        self.sampling = sampling.clamped(nCodebooks: c.nCodebooks, codebookSize: c.codebookSize)
        let n = prompt.ids.shape[0]
        guard n < c.maxSeqlen else { throw Zonos3Error.invalid("TTS input len \(n) exceeds \(c.maxSeqlen)") }
        remain = min(maxTokens, c.maxSeqlen - n)
        let last = prompt.ids[n - 1, 0..<c.nCodebooks].asArray(Int32.self)
        hasAudioPrefix = last.contains { Int($0) != c.audioPadID }
        key = MLXRandom.key(seed)
        let out = try lm.forward(prompt.ids, state: state, speakerRows: prompt.speakerRows, conditionRows: prompt.conditionRows)
        logits = out[n - 1]
        eval(logits, state.arrays())
    }

    public var isFinished: Bool { remain <= 0 || countdown == 0 }

    public func next() throws -> GeneratedRow {
        let c = lm.config, nCB = c.nCodebooks
        let total = generated
        var repetition: [[Int]]? = nil
        let active = sampling.activeRepetitionWindow(available: min(total, history.count))
        if active > 0 {
            var rep = [[Int]](repeating: [], count: nCB)
            for row in history.suffix(active) {
                for cb in 0..<sampling.repetitionCodebooks where row[cb] >= 0 && Int(row[cb]) < c.codebookSize { rep[cb].append(Int(row[cb])) }
            }
            repetition = rep
        }
        var stopBlocked: [Bool]? = nil
        if sampling.minFrames > 0 { stopBlocked = (0..<nCB).map { total < sampling.minFrames + $0 } }
        let width = max(c.eoaID, c.audioPadID) + 1
        let noise: MLXArray
        if let injectedNoise {
            noise = injectedNoise[total]
        } else {
            let (k1, k2) = MLXRandom.split(key: key)
            key = k2
            // an exponential draw: -log(U), U in (0, 1]
            noise = -log(1 - MLXRandom.uniform(low: Float(0), high: Float(1), [nCB, width], key: k1))
        }
        let sampled = sampler.sample(logits, noise: noise, sampling, repetition: repetition, stopBlocked: stopBlocked, totalGenerated: total, hasAudioPrefix: hasAudioPrefix)
        // queue the next forward on the lazy sample, then read the token back (upstream's `ahead`)
        let input = forcedRows.map { $0[total].asType(.int32) } ?? sampled
        let out = try lm.forward(input.expandedDimensions(axis: 0), state: state)
        let nextLogits = out[0]
        asyncEval([sampled, nextLogits] + state.arrays())
        let token = sampled.asArray(Int32.self)
        history.append(Array((forcedRows == nil ? token : input.asArray(Int32.self))[0..<nCB]))
        generated += 1
        remain -= 1
        var finished = remain <= 0
        if eosFrame < 0 {
            let cols = (0..<nCB).filter { Int(token[$0]) == c.eoaID }
            if let top = cols.max() { eosFrame = max(0, total - top); countdown = nCB + 1 }
        }
        if countdown > 0 {
            countdown -= 1
            if countdown == 0 { finished = true }
        }
        logits = nextLogits
        return GeneratedRow(row: token, eosFrame: eosFrame >= 0 ? eosFrame : nil, finished: finished)
    }
}

/// The LM input of one request: prompt rows [rows, n_codebooks + 1], the speaker-embedding row and the condition rows.
public struct PromptRows {
    public let ids: MLXArray
    public let speakerRows: [Int: MLXArray]
    public let conditionRows: [Int: Int]
    public let prefillFrames: Int
    public init(ids: MLXArray, speakerRows: [Int: MLXArray], conditionRows: [Int: Int], prefillFrames: Int) {
        self.ids = ids; self.speakerRows = speakerRows; self.conditionRows = conditionRows; self.prefillFrames = prefillFrames
    }
}
