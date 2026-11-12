// zonos3-gates — parity and validation CLI. Gates read goldens from Tools/oracle_dump.py (upstream zonos3_mlx @ 4c66cea).
//   zonos3-gates --s0 <checkpoint dir>
//   zonos3-gates --s1-lm <checkpoint dir> <goldens dir>
import Foundation
import MLX
import Zonos3Core

func err(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
func npy(_ dir: URL, _ rel: String) throws -> MLXArray { try NPY.load(dir.appendingPathComponent(rel + ".npy")) }
func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) }
func relErr(_ a: MLXArray, _ b: MLXArray) -> Float {
    let d = (a.asType(.float32) - b.asType(.float32)); let n = sqrt((d * d).sum()).item(Float.self)
    let m = sqrt((b.asType(.float32) * b.asType(.float32)).sum()).item(Float.self)
    return n / max(m, 1e-12)
}

let args = CommandLine.arguments
guard args.count >= 3 else { err("usage: zonos3-gates --s0|--s1-lm|--s1-codec|--s1-spk|--s1-tok|--s2-gen|--s2-prompt <checkpoint dir> <goldens dir>\n       zonos3-gates --synth <checkpoint dir> <out.wav> <text> [--ref ref.wav] [--seed N] [--full] [--runs N]"); exit(2) }
let mode = args[1]
if mode == "validate" {
    guard args.count >= 5 else { err("validate <checkpoint dir> <reference clip> <out dir>"); exit(2) }
    do {
        let ok = try await validate(checkpoint: URL(fileURLWithPath: args[2]), reference: URL(fileURLWithPath: args[3]), out: URL(fileURLWithPath: args[4]))
        exit(ok ? 0 : 1)
    } catch { err("error: \(error)"); exit(1) }
}
let ckpt = URL(fileURLWithPath: args[2])
var failed = false
func check(_ name: String, _ ok: Bool, _ detail: String) {
    print("\(ok ? "PASS" : "FAIL")  \(name)  \(detail)"); if !ok { failed = true }
}

do {
    let config = try CheckpointConfig(directory: ckpt)
    switch mode {
    case "--s0":
        let lm = try Zonos3LM(config: config)
        let want = lm.expectedKeys(), have = lm.loadedKeys
        check("lm key contract", want == have, "expected \(want.count), on disk \(have.count), missing \(want.subtracting(have).sorted().prefix(5)), unused \(have.subtracting(want).sorted().prefix(5))")
    case "--s1-lm":
        let gold = URL(fileURLWithPath: args[3])
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: gold.appendingPathComponent("meta.json"))) as! [String: Any]
        let plan = meta["plan"] as! [String: Any]
        let lm = try Zonos3LM(config: config)
        let ids = try npy(gold, "lm/prompt_ids")
        let spk = try npy(gold, "lm/speaker_embedding_row")
        let condStart = plan["condition_token_start"] as! Int
        let condValues = plan["condition_values"] as! [Int]
        var cond: [Int: Int] = [:]
        for (i, v) in condValues.enumerated() { cond[condStart + i] = v }
        lm.capture = [:]
        let state = LMState()
        err("loaded; forward…")
        let logits = try lm.forward(ids, state: state, speakerRows: [plan["speaker_row"] as! Int: spk], conditionRows: cond)
        err("graph built; eval…")
        if ProcessInfo.processInfo.environment["Z3_BISECT"] != nil {
            for name in (meta["lm_captures"] as! [String]) { if let a = lm.capture![name] { err("  eval \(name) \(a.shape)"); eval(a) } }
        }
        eval(logits)
        err("evaluated")
        let names = (meta["lm_captures"] as! [String])
        var worst: (String, Float) = ("", 0)
        var missing = 0
        for name in names {
            guard let mine = lm.capture![name] else { missing += 1; continue }
            if ProcessInfo.processInfo.environment["Z3_BISECT"] != nil { err("  cmp \(name) \(mine.shape)") }
            let g = try npy(gold, "lm/prefill/\(name)")
            let r = relErr(mine.reshaped(g.shape), g)
            if r > worst.1 { worst = (name, r) }
            if ["emb_norm", "attn_out_0", "ffn_out_0", "ffn_out_1", "attn_out_3", "layer_x_5", "layer_x_10", "out_norm", "logits"].contains(name) {
                print(String(format: "   %@ rel %.3e  max_abs %.3e", name.padding(toLength: 16, withPad: " ", startingAt: 0), r, maxAbs(mine.reshaped(g.shape), g)))
            }
        }
        lm.capture = nil
        check("prefill captures (\(names.count))", missing == 0 && worst.1 < 2e-3, String(format: "worst %@ rel %.3e, missing %d", worst.0, worst.1, missing))
        let last = try npy(gold, "lm/prefill_logits_last")
        let mineLast = logits[logits.shape[0] - 1]
        let agree = (argMax(mineLast, axis: -1) .== argMax(last, axis: -1)).asType(.int32).sum().item(Int32.self)
        check("prefill last-row logits", relErr(mineLast, last) < 2e-3 && agree == 10, String(format: "rel %.3e, argmax agree %d/10", relErr(mineLast, last), agree))
        // teacher-forced decode: rows[k] in, logits predicting rows[k + 1]
        let rows = try npy(gold, "gen/rows")
        let tf = try npy(gold, "gen/tf_logits")
        var worstTF: Float = relErr(mineLast, tf[0]); var agreeTF = 0
        for k in 0..<11 {
            let l = try lm.forward(rows[k].expandedDimensions(axis: 0), state: state)[0]
            eval(l, state.arrays())
            worstTF = max(worstTF, relErr(l, tf[k + 1]))
            agreeTF += Int((argMax(l, axis: -1) .== argMax(tf[k + 1], axis: -1)).asType(.int32).sum().item(Int32.self))
        }
        check("teacher-forced decode (11 steps)", worstTF < 2e-3 && agreeTF == 110, String(format: "worst rel %.3e, argmax agree %d/110", worstTF, agreeTF))
    case "--s2-gen":
        // the seeded generation of the oracle, replayed with its own noise: the sampled rows must be identical
        let gold = URL(fileURLWithPath: args[3])
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: gold.appendingPathComponent("meta.json"))) as! [String: Any]
        let plan = meta["plan"] as! [String: Any]
        let lm = try Zonos3LM(config: config)
        let condStart = plan["condition_token_start"] as! Int
        var cond: [Int: Int] = [:]
        for (i, v) in (plan["condition_values"] as! [Int]).enumerated() { cond[condStart + i] = v }
        let prompt = PromptRows(ids: try npy(gold, "lm/prompt_ids"), speakerRows: [plan["speaker_row"] as! Int: try npy(gold, "lm/speaker_embedding_row")],
                                conditionRows: cond, prefillFrames: plan["prefill_frames"] as! Int)
        let rows = try npy(gold, "gen/rows")
        let want = rows.asArray(Int32.self), width = rows.shape[1], n = rows.shape[0]
        let gen = try RowGenerator(lm: lm, prompt: prompt, maxTokens: plan["max_tokens"] as! Int)
        gen.injectedNoise = try npy(gold, "gen/noise")
        // (a) teacher-forced: every step sees the oracle's history, so one flipped near-tie cannot cascade
        gen.forcedRows = rows
        var flips = 0, flippedRows: [Int] = [], count = 0
        let t0 = Date()
        while count < n {
            let r = try gen.next(); count += 1
            let w = Array(want[(count - 1) * width ..< count * width])
            let d = zip(w, r.row).filter { $0 != $1 }.count
            if d > 0 { flips += d; flippedRows.append(count - 1) }
        }
        let dt = Date().timeIntervalSince(t0)
        check("teacher-forced sampling, oracle noise", flips <= max(1, n * 10 / 200),
              "\(n) rows x 10 codes: \(flips) flipped codes in rows \(flippedRows), \(String(format: "%.1f", Double(count) / dt)) rows/s")
        // (b) free-running: identical until the first flip
        let free = try RowGenerator(lm: lm, prompt: prompt, maxTokens: plan["max_tokens"] as! Int)
        free.injectedNoise = try npy(gold, "gen/noise")
        var firstBad = -1, k = 0, eos: Int? = nil
        while k < n {
            let r = try free.next(); k += 1
            if firstBad < 0 && Array(want[(k - 1) * width ..< k * width]) != r.row { firstBad = k - 1 }
            eos = r.eosFrame
            if r.finished { break }
        }
        let wantEOS = (meta["gen"] as! [String: Any])["eos_frame"] as? Int
        let expectFirst = flippedRows.first ?? -1
        check("free-running generation", firstBad == expectFirst,
              "first divergent row \(firstBad) (teacher-forced first flip \(expectFirst)); rows \(k), eos \(String(describing: eos)) vs oracle \(String(describing: wantEOS))")
    case "--s1-codec":
        let gold = URL(fileURLWithPath: args[3])
        let dec = try CodecDecoder(config: config)
        let want = dec.expectedKeys(), have = dec.loadedKeys
        check("codec key contract", want == have, "expected \(want.count), on disk \(have.count), missing \(want.subtracting(have).sorted().prefix(5)), unused \(have.subtracting(want).sorted().prefix(5))")
        let codesA = try npy(gold, "codec/codes")
        let flat = codesA.asArray(Int32.self), F = codesA.shape[0]
        let codes = (0..<F).map { Array(flat[$0 * 10 ..< $0 * 10 + 10]) }
        func run(_ sizes: [Int]) throws -> ([Float], Double) {
            let st = try dec.stream(); var out: [Float] = []; var i = 0
            let t0 = Date()
            for n in sizes { if i >= F { break }; out += try st.decode(Array(codes[i..<min(i + n, F)])); i += n }
            out += st.flush()
            return (out, Date().timeIntervalSince(t0))
        }
        func cmp(_ a: [Float], _ g: MLXArray) -> (Float, Float, Int) {
            let b = g.asArray(Float.self); let n = min(a.count, b.count)
            var m: Float = 0, e: Double = 0, s: Double = 0
            for i in 0..<n { let d = abs(a[i] - b[i]); m = max(m, d); e += Double(d * d); s += Double(b[i] * b[i]) }
            return (m, Float(10 * log10(s / max(e, 1e-30))), a.count - b.count)
        }
        let ramp = [3, 1, 2, 4, 8] + Array(repeating: 15, count: 50)
        _ = try run(ramp)  // warm-up (kernel compiles)
        let (a, dtA) = try run(ramp)
        let (m1, snr1, d1) = cmp(a, try npy(gold, "codec/pcm_stream"))
        check("stream decode vs oracle stream", d1 == 0 && snr1 > 50, String(format: "max_abs %.2e, SNR %.1f dB, length diff %d, %.1fx real time", m1, snr1, d1, Double(a.count) / 44100 / dtA))
        let (b, _) = try run([F])
        let (m2, snr2, d2) = cmp(b, try npy(gold, "codec/pcm_whole"))
        check("one-chunk decode vs oracle whole decode", d2 == 0 && snr2 > 40, String(format: "max_abs %.2e, SNR %.1f dB, length diff %d", m2, snr2, d2))
        var m3: Float = 0; for i in 0..<min(a.count, b.count) { m3 = max(m3, abs(a[i] - b[i])) }
        check("chunking invariance (ramp vs one chunk)", a.count == b.count && m3 < 1e-2, String(format: "max_abs %.2e", m3))
    case "--s1-spk":
        let gold = URL(fileURLWithPath: args[3])
        let enc = try SpeakerEncoder(config: config)
        let pcm = try npy(gold, "spk/pcm_44100").asArray(Float.self)
        var dump: [String: MLXArray] = [:]
        let t0 = Date()
        let emb = try enc.embed(pcm44100: pcm, dump: &dump)
        let dt = Date().timeIntervalSince(t0)
        for name in ["fbank", "fbank_cmn", "feat_1", "feat_2", "feat_3", "feat_4", "pooled", "embedding"] {
            let g = try npy(gold, "spk/\(name)")
            guard let mine = dump[name], mine.size == g.size else { check("spk \(name)", false, "shape \(String(describing: dump[name]?.shape)) vs \(g.shape)"); continue }
            let r = relErr(mine.reshaped(g.shape), g)
            check("spk \(name)", r < (name.hasPrefix("fbank") ? 1e-4 : 2e-3), String(format: "rel %.3e, max_abs %.3e", r, maxAbs(mine.reshaped(g.shape), g)))
        }
        let g = try npy(gold, "spk/embedding").reshaped(-1)
        let e = MLXArray(emb)
        let cosv = ((e * g).sum() / (sqrt((e * e).sum()) * sqrt((g * g).sum()))).item(Float.self)
        check("speaker embedding cosine", cosv > 0.9999, String(format: "cosine %.6f, %.2f s for %.1f s of audio", cosv, dt, Double(pcm.count) / 44100))
    case "--s1-tok":
        let gold = URL(fileURLWithPath: args[3])
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: gold.appendingPathComponent("meta.json"))) as! [String: Any]
        let tok = try SuperBPETokenizer(url: ckpt.appendingPathComponent("tokenizer.json"))
        var bad = 0
        for case let item as [String: Any] in meta["tokenizer"] as! [Any] {
            let text = item["text"] as! String, want = (item["ids"] as! [NSNumber]).map(\.intValue)
            let got = tok.textIDs(text)
            if got != want { bad += 1; print("   MISMATCH \(text.debugDescription)\n     want \(want)\n     got  \(got)") }
        }
        check("tokenizer ids (\((meta["tokenizer"] as! [Any]).count) texts)", bad == 0, "\(bad) mismatched")
    case "--s2-prompt":
        // the Swift prompt builder against the oracle's rows (speaker embedding from the Swift encoder)
        let gold = URL(fileURLWithPath: args[3])
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: gold.appendingPathComponent("meta.json"))) as! [String: Any]
        let plan = meta["plan"] as! [String: Any]
        let tts = try Zonos3TTS(directory: ckpt)
        let emb = try tts.embedSpeaker(pcm44100: try npy(gold, "spk/pcm_44100").asArray(Float.self))
        let (prompt, _) = try tts.prompt(text: meta["text"] as! String, speakerEmbedding: emb)
        let wantIDs = try npy(gold, "lm/prompt_ids")
        let same = prompt.ids.shape == wantIDs.shape && (prompt.ids .== wantIDs).all().item(Bool.self)
        let condStart = plan["condition_token_start"] as! Int
        var cond: [Int: Int] = [:]
        for (i, v) in (plan["condition_values"] as! [Int]).enumerated() { cond[condStart + i] = v }
        check("prompt rows", same && prompt.conditionRows == cond && Array(prompt.speakerRows.keys) == [plan["speaker_row"] as! Int],
              "shape \(prompt.ids.shape) vs \(wantIDs.shape), condition rows \(prompt.conditionRows.sorted { $0.key < $1.key }), speaker row \(Array(prompt.speakerRows.keys))")
    case "--synth":
        guard args.count >= 5 else { err("--synth <checkpoint dir> <out.wav> <text> [--ref ref.wav] [--seed N] [--full] [--runs N]"); exit(2) }
        var ref: String? = nil, seed: UInt64 = 0, full = false, runs = 1
        var i = 5
        while i < args.count {
            switch args[i] {
            case "--ref": ref = args[i + 1]; i += 2
            case "--seed": seed = UInt64(args[i + 1])!; i += 2
            case "--runs": runs = Int(args[i + 1])!; i += 2
            case "--full": full = true; i += 1
            default: err("unknown option \(args[i])"); exit(2)
            }
        }
        let t0 = Date()
        let tts = try Zonos3TTS(directory: ckpt, fullDecoder: full)
        err(String(format: "loaded in %.2f s", Date().timeIntervalSince(t0)))
        var emb: [Float]? = nil
        if let ref {
            let (samples, sr) = try WAV.read(URL(fileURLWithPath: ref))
            guard sr == 44100 else { err("reference must be 44.1 kHz mono WAV in this CLI (got \(sr) Hz)"); exit(2) }
            let t1 = Date()
            emb = try tts.embedSpeaker(pcm44100: samples)
            err(String(format: "speaker embedding in %.2f s", Date().timeIntervalSince(t1)))
        }
        for run in 0..<runs {
            var opt = SynthesisOptions(); opt.seed = seed + UInt64(run)
            let r = try tts.synthesize(text: args[4], speakerEmbedding: emb, options: opt)
            let secs = Double(r.pcm.count) / Double(r.sampleRate)
            print(String(format: "run %d seed %llu: %.2f s audio, %d rows, eos %@, first audio %.3f s, total %.3f s (%.1fx real time)%@",
                         run, opt.seed, secs, r.generatedRows, r.eosFrame.map(String.init) ?? "none", r.firstAudioSeconds, r.totalSeconds, secs / r.totalSeconds, r.hitFrameLimit ? " HIT FRAME LIMIT" : ""))
            let out = runs == 1 ? args[3] : args[3].replacingOccurrences(of: ".wav", with: "_\(run).wav")
            try WAV.write(r.pcm, sampleRate: r.sampleRate, to: URL(fileURLWithPath: out))
        }
    default:
        err("unknown mode \(mode)"); exit(2)
    }
} catch {
    err("error: \(error)"); exit(1)
}
exit(failed ? 1 : 0)
