// SpeakerEncoder.swift — ZONOS3 Mini speaker encoder (ownmect_s2: Kaldi fbank + MECT-B2 + ASTP, 192-d).
// A port of zonos3_mlx/models/speaker_encoder.py and audio_io.py `polyphase_resample` (Zyphra, MIT) @ 4c66cea.
// The frontend (16 kHz resampling, Kaldi fbank, CMN) runs on the host in Double; the network runs in MLX (NHWC
// activations [1, F, T, C]: mel / feature axis = height, time = width).
import Accelerate
import Foundation
import MLX
import MLXFast

public final class SpeakerEncoder {
    static let arch = "mect_spk"
    static let hidden = 64
    static let maxSamples16k = 60 * 16000
    // MECT_B2 per stage: (cnn depth, out channels, stride F, stride T, expansion) per block.
    static let mectB2: [[(Int, Int, Int, Int, Int)]] = [
        Array(repeating: (3, 32, 1, 1, 4), count: 2),
        [(3, 64, 2, 1, 2)] + Array(repeating: (3, 64, 1, 2, 2), count: 3),
        [(3, 128, 2, 1, 2)] + Array(repeating: (3, 128, 1, 2, 2), count: 4),
        [(3, 256, 2, 1, 1)] + Array(repeating: (2, 256, 1, 2, 1), count: 2),
    ]

    public let embeddingLength: Int
    let nMels, frameLength, frameShift, nFFT, nHeads, posKernel: Int
    let preemph, scale, logFloor: Double
    let lnEps: Float, varFloor: Float
    let window: [Double], melBanks: [Double], resampleKernel: [Double]  // [n_mels * 257], [160 * 475]
    var t: [String: MLXArray] = [:]
    var grouped: [String: MLXArray] = [:]
    public private(set) var loadedKeys: Set<String> = []

    public init(config: CheckpointConfig, file: String = "speaker_encoder/model.safetensors") throws {
        let m = Meta(try config.meta(file), arch: Self.arch)
        guard try m.string("backbone") == "MECT_B2" else { throw Zonos3Error.invalid("expected a MECT-B2 speaker encoder") }
        embeddingLength = try m.int("embedding_length")
        nMels = try m.int("n_mels"); frameLength = try m.int("frame_length"); frameShift = try m.int("frame_shift")
        nFFT = try m.int("n_fft"); nHeads = try m.int("n_heads"); posKernel = try m.int("pos_kernel")
        preemph = try m.double("preemphasis"); scale = try m.double("waveform_scale"); logFloor = try m.double("log_floor")
        lnEps = Float(try m.double("ln_eps")); varFloor = Float(try m.double("pool_var_floor"))
        let raw = try Device.withDefaultDevice(.cpu) { try loadArrays(url: config.root.appendingPathComponent(file)) }
        func host(_ name: String) throws -> [Double] {
            guard let a = raw[name] else { throw Zonos3Error.missing(name) }
            return a.asType(.float32).asArray(Float.self).map(Double.init)
        }
        window = try host("frontend.window"); melBanks = try host("frontend.mel_banks"); resampleKernel = try host("frontend.resample_44100")
        loadedKeys = Set(raw.keys)
        for (name, a) in raw where !name.hasPrefix("frontend.") { t[name] = a.asType(.float32) }
        // Grouped 3x3 convs: the per-slot w{j} [OC, 1, 3, 3] tensors -> MLX [OC, 3, 3, a] with groups = IC / a.
        for name in raw.keys.sorted() where name.hasSuffix(".w0") {
            let prefix = String(name.dropLast(3))
            var slots: [MLXArray] = []
            while let s = raw["\(prefix).w\(slots.count)"] { slots.append(s.asType(.float32)) }
            grouped[prefix] = concatenated(slots, axis: 1).transposed(0, 2, 3, 1)
        }
        eval(Array(t.values) + Array(grouped.values))
    }

    // MARK: frontend

    /// torchaudio's polyphase sinc resampler: out[n*new + i] = sum_k K[i][k] * x[n*orig + k - width].
    static func polyphaseResample(_ x: [Float], kernel: [Double], orig: Int, new: Int, width: Int) -> [Double] {
        let L = x.count
        let total = (new * L + orig - 1) / orig
        if total == 0 { return [] }
        let blocks = (total + new - 1) / new
        let taps = 2 * width + orig
        var padded = [Double](repeating: 0, count: width + blocks * orig + taps)
        for i in 0..<L { padded[width + i] = Double(x[i]) }
        var out = [Double](repeating: 0, count: blocks * new)
        padded.withUnsafeBufferPointer { p in
            kernel.withUnsafeBufferPointer { k in
                out.withUnsafeMutableBufferPointer { o in
                    for b in 0..<blocks {
                        cblas_dgemv(CblasRowMajor, CblasNoTrans, Int32(new), Int32(taps), 1.0, k.baseAddress!, Int32(taps),
                                    p.baseAddress! + b * orig, 1, 0.0, o.baseAddress! + b * new, 1)
                    }
                }
            }
        }
        return Array(out[0..<total])
    }

    /// Kaldi fbank of the first <= 60 s of 16 kHz audio -> [T][80] (log mel energies, before CMN).
    func fbank(_ wav16: [Double]) throws -> [[Double]] {
        let wav = Array(wav16.prefix(Self.maxSamples16k))
        let fl = frameLength, fs = frameShift
        guard wav.count >= fl else { throw Zonos3Error.invalid("speaker audio is shorter than 25 ms") }
        let T = 1 + (wav.count - fl) / fs
        let bins = nFFT / 2 + 1
        let log2n = vDSP_Length(log2(Double(nFFT)).rounded())
        guard let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else { throw Zonos3Error.invalid("FFT setup failed") }
        defer { vDSP_destroy_fftsetupD(setup) }
        var out: [[Double]] = []
        out.reserveCapacity(T)
        var re = [Double](repeating: 0, count: nFFT), im = [Double](repeating: 0, count: nFFT)
        var frame = [Double](repeating: 0, count: fl)
        for ti in 0..<T {
            var mean = 0.0
            for j in 0..<fl { frame[j] = wav[ti * fs + j] * scale; mean += frame[j] }
            mean /= Double(fl)
            for j in 0..<fl { frame[j] -= mean }
            for j in 0..<nFFT { re[j] = 0; im[j] = 0 }
            for j in 0..<fl { re[j] = (frame[j] - preemph * frame[max(j - 1, 0)]) * window[j] }
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    vDSP_fft_zipD(setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
            var row = [Double](repeating: 0, count: nMels)
            for mi in 0..<nMels {
                var acc = 0.0
                let base = mi * bins
                for k in 0..<bins where melBanks[base + k] != 0 { acc += (re[k] * re[k] + im[k] * im[k]) * melBanks[base + k] }
                row[mi] = Foundation.log(max(acc, logFloor))
            }
            out.append(row)
        }
        return out
    }

    // MARK: network

    private func bn(_ x: MLXArray, _ p: String) -> MLXArray { x * t[p + ".scale"]! + t[p + ".shift"]! }
    private func pw(_ x: MLXArray, _ p: String, bias: Bool = true) -> MLXArray {
        let w = t[p + ".weight"]!
        let y = matmul(x, w.reshaped(w.shape[0], w.shape[1]).T)
        return bias ? y + t[p + ".bias"]! : y
    }
    private func lin(_ x: MLXArray, _ p: String, bias: Bool = true) -> MLXArray {
        let y = matmul(x, t[p + ".weight"]!.T)
        return bias ? y + t[p + ".bias"]! : y
    }
    private func ln(_ x: MLXArray, _ p: String) -> MLXArray { MLXFast.layerNorm(x, weight: t[p + ".weight"]!, bias: t[p + ".bias"]!, eps: lnEps) }
    private func groupedConv(_ x: MLXArray, _ p: String, _ sF: Int, _ sT: Int) -> MLXArray {
        let w = grouped[p]!
        return conv2d(x, w, stride: .init((sF, sT)), padding: .init(1), groups: x.shape[3] / w.shape[3])
    }
    private func newGELU(_ x: MLXArray) -> MLXArray { 0.5 * x * (1.0 + tanh(Float((2.0 / Double.pi).squareRoot()) * (x + 0.044715 * x * x * x))) }
    private func geluErf(_ x: MLXArray) -> MLXArray { 0.5 * x * (1.0 + erf(x * Float(1.0 / 2.0.squareRoot()))) }

    private func resBlock(_ x: MLXArray, _ q: String, _ sF: Int, _ sT: Int) -> MLXArray {
        var y = groupedConv(x, q + "conv1", sF, sT)
        y = maximum(bn(pw(y, q + "pw1"), q + "bn1"), MLXArray(Float(0)))
        y = groupedConv(y, q + "conv2", 1, 1)
        y = maximum(bn(pw(y, q + "pw2"), q + "bn2"), MLXArray(Float(0)))
        var sc = x
        if t[q + "sc.weight"] != nil {
            sc = bn(pw(x[0..., .stride(by: sF), .stride(by: sT), 0...], q + "sc", bias: false), q + "sc.bn")
        }
        return y + sc
    }

    private func attention(_ h: MLXArray, _ p: String) -> MLXArray {
        let T = h.shape[0], nh = nHeads, hd = Self.hidden / nHeads
        func heads(_ z: MLXArray) -> MLXArray { z.reshaped(T, nh, hd).transposed(1, 0, 2).expandedDimensions(axis: 0) }
        let q = heads(lin(h, p + "attn.q") * Float(1.0 / Double(hd).squareRoot()))
        let k = heads(lin(h, p + "attn.k")), v = heads(lin(h, p + "attn.v"))
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1.0, mask: nil).squeezed(axis: 0).transposed(1, 0, 2).reshaped(T, Self.hidden)
        return lin(o, p + "attn.o")
    }

    private func moe(_ h: MLXArray, _ p: String) -> MLXArray {
        let r = softmax(matmul(h, t[p + "moe.gate.weight"]!.T), axis: -1)  // [T, 4]
        let e = newGELU(matmul(h, t[p + "moe.experts.weight"]!.T)).reshaped(h.shape[0], -1, Self.hidden)  // [T, 4, 64]
        let mix = (e * r.expandedDimensions(axis: -1)).sum(axis: 1)
        return lin(newGELU(mix), p + "ffn_out")
    }

    /// Zero-padded depthwise Conv1d (K = 113) over time, BN, exact GELU, pointwise.
    private func posConv(_ h: MLXArray, _ p: String) -> MLXArray {
        let pad = posKernel / 2
        let w = t[p + "pos.dw.weight"]!  // [64, K]
        let z = MLXArray.zeros([pad, Self.hidden], dtype: .float32)
        let hp = concatenated([z, h, z], axis: 0)
        var acc = conv1d(hp.expandedDimensions(axis: 0), w.expandedDimensions(axis: -1), groups: Self.hidden).squeezed(axis: 0)  // [T, 64]
        acc = acc + t[p + "pos.dw.bias"]!
        acc = acc * t[p + "pos.bn.scale"]! + t[p + "pos.bn.shift"]!
        return lin(geluErf(acc), p + "pos.pw")
    }

    /// [1, F, T, C] -> [T, F*C] with k = f*C + c.
    private func flatten(_ x: MLXArray) -> MLXArray {
        let F = x.shape[1], T = x.shape[2], C = x.shape[3]
        return x.squeezed(axis: 0).transposed(1, 0, 2).reshaped(T, F * C)
    }
    private func unflatten(_ x: MLXArray, _ F: Int, _ C: Int) -> MLXArray {
        x.reshaped(x.shape[0], F, C).transposed(1, 0, 2).expandedDimensions(axis: 0)
    }

    private func block(_ x: MLXArray, _ xf: MLXArray, _ cfg: (Int, Int, Int, Int, Int), _ p: String) -> (MLXArray, MLXArray) {
        let (cnn, _, sF, sT, _) = cfg
        let T = x.shape[2]
        var y = x
        for r in 0..<cnn { y = resBlock(y, "\(p)cnn.\(r).", r == 0 ? sF : 1, r == 0 ? sT : 1) }
        if t[p + "reduce.weight"] != nil { y = bn(pw(y, p + "reduce"), p + "reduce.bn") }
        let F2 = y.shape[1], C2 = y.shape[3]
        let x2 = flatten(y)
        var h = ln(lin(x2, p + "rproj"), p + "rproj.ln")
        h = h + posConv(h, p)
        h = ln(h + attention(h, p), p + "ln1")
        h = ln(h + moe(h, p), p + "ln2")
        var e = lin(h, p + "eproj") + x2
        if sT > 1 { e = repeated(e, count: sT, axis: 0)[0..<T] }
        let out = e + xf
        return (unflatten(out, F2, C2), out)
    }

    /// Mono 44.1 kHz PCM -> the 192-d speaker embedding (not L2-normalized). `dump`: intermediates for parity gates.
    public func embed(pcm44100: [Float], dump: UnsafeMutablePointer<[String: MLXArray]>? = nil) throws -> [Float] {
        let wav16 = Self.polyphaseResample(pcm44100, kernel: resampleKernel, orig: 441, new: 160, width: 17).map { Double(Float($0)) }
        let fb = try fbank(wav16).map { $0.map { Double(Float($0)) } }  // float32 rounding, as upstream stores it
        let T = fb.count
        var mean = [Double](repeating: 0, count: nMels)
        for row in fb { for j in 0..<nMels { mean[j] += row[j] } }
        for j in 0..<nMels { mean[j] /= Double(T) }
        // x [1, 80, T, 1]: mel axis = height, time = width
        var flat = [Float](repeating: 0, count: nMels * T)
        for ti in 0..<T { for j in 0..<nMels { flat[j * T + ti] = Float(fb[ti][j] - mean[j]) } }
        if let dump {
            dump.pointee["fbank"] = MLXArray(fb.flatMap { $0.map { Float($0) } }, [T, nMels])
            var cmn = [Float](repeating: 0, count: T * nMels)
            for ti in 0..<T { for j in 0..<nMels { cmn[ti * nMels + j] = flat[j * T + ti] } }
            dump.pointee["fbank_cmn"] = MLXArray(cmn, [T, nMels])
        }
        var x = MLXArray(flat, [1, nMels, T, 1])
        let w = t["fe.head.conv.weight"]!.transposed(0, 2, 3, 1)  // [32, 3, 3, 1]
        x = bn(conv2d(x, w, padding: .init(1)) + t["fe.head.conv.bias"]!, "fe.head.bn")
        var xf = flatten(x)
        var feats: [MLXArray] = []
        for (s, stage) in Self.mectB2.enumerated() {
            for (k, cfg) in stage.enumerated() {
                (x, xf) = block(x, xf, cfg, "fe.b\(s + 1).\(k).")
                eval(x, xf)
            }
            feats.append(xf)
        }
        let dense = t["fe.dense_w"]!  // [4, 2560]
        var fused = feats[0] * dense[0]
        for s in 1..<4 { fused = fused + feats[s] * dense[s] }
        let att = lin(tanh(lin(fused, "pool.l1")), "pool.l2")  // [T, 2560]
        let alpha = softmax(att, axis: 0)
        let mu = (alpha * fused).sum(axis: 0)
        let variance = (alpha * fused * fused).sum(axis: 0) - mu * mu
        let std = sqrt(maximum(variance, MLXArray(varFloor)))
        let pooled = concatenated([mu, std])
        let emb = matmul(pooled * t["bn.scale"]! + t["bn.shift"]!, t["linear.weight"]!.T)
        eval(emb, pooled)
        let out = emb.asArray(Float.self)
        if let dump {
            for s in 0..<4 { dump.pointee["feat_\(s + 1)"] = feats[s] }
            dump.pointee["pooled"] = pooled.expandedDimensions(axis: 0)
            dump.pointee["embedding"] = emb.expandedDimensions(axis: 0)
        }
        guard out.allSatisfy({ $0.isFinite }) else { throw Zonos3Error.invalid("speaker encoder produced a non-finite embedding") }
        return out
    }
}
