// CodecDecoder.swift — ZONOS3 Mini codec decoder (`mossdual_istft`, ZAC-1): codes -> 44.1 kHz PCM, incrementally.
// A port of zonos3_mlx/models/codec_decoder.py (`CodecDecoder`, `CodecStream`; Zyphra, MIT) @ 4c66cea — the incremental
// form only: semantic table + causal ConvNeXt head, residual tables, five windowed-causal transformer stages with patch
// upsampling (15 -> 240 Hz), the multi-resolution iSTFT head (ConvNeXt trunk, two branches) on the device; overlap-add,
// FIR mixer and resampler on the host (CodecHost.swift). Decoding every frame through one stream and flushing equals the
// whole decode.
import Foundation
import MLX
import MLXFast

struct StageHP { let inDim, outDim, dModel, nHead, nLayer, nFF, context, patch: Int }

public final class CodecDecoder {
    static let arch = "mossdual_istft"
    let f: WeightFile
    public let id: String  // keep2_plain | full_mr
    public let sampleRate, hop, nCodebooks, codebookSize: Int
    let semDepth, headDepth, headHop: Int
    let nFFT: [Int]
    let maxLogMag: Float, lnEps: Float, convnextEps: Float
    public let headroomDB: Double
    let rs: (orig: Int, new: Int, width: Int)
    let stages: [StageHP]
    let hostWindows: [[Float]], hostMixer: [[Float]], hostKernel: [Float]
    let resBiases: MLXArray?  // [n_codebooks - 1, latent]
    /// Windowed inverse-rFFT bases per branch: frames = re @ cosBasis + im @ sinBasis ([bins, n], the window folded in).
    let cosBasis: [MLXArray], sinBasis: [MLXArray]

    public init(config: CheckpointConfig, file: String = "codec/model.safetensors") throws {
        let host: Set<String> = ["head.window.0", "head.window.1", "head.mixer.weight", "resample.kernel"]
        let f = try WeightFile(url: config.root.appendingPathComponent(file), quantized: config.quantized(file)) { name, shape in
            // matrices: 2-D `.weight` tensors that are not depthwise taps, norms, code tables or the mixer
            !(shape.count == 2 && name.hasSuffix(".weight") && !name.contains(".dw.") && !name.contains("norm")
                && !name.hasPrefix("sem.") && !name.hasPrefix("res.") && name != "head.mixer.weight") || host.contains(name)
        }
        self.f = f
        let m = Meta(try config.meta(file), arch: Self.arch)
        id = try m.string("head.variant")
        sampleRate = try m.int("sample_rate"); hop = try m.int("hop"); nCodebooks = try m.int("n_codebooks"); codebookSize = try m.int("codebook_size")
        semDepth = try m.int("sem_head.depth"); headDepth = try m.int("head.depth"); headHop = try m.int("head.hop")
        nFFT = try m.ints("head.n_fft")
        maxLogMag = Float(try m.double("head.max_log_magnitude")); lnEps = Float(try m.double("ln_eps")); convnextEps = Float(try m.double("convnext_eps"))
        headroomDB = try m.double("headroom_db")
        rs = (try m.int("resample.orig"), try m.int("resample.new"), try m.int("resample.width"))
        let cols = try ["in_dim", "out_dim", "d_model", "n_head", "n_layer", "n_ff", "context", "patch"].map { try m.ints("stage.\($0)") }
        stages = (0..<cols[0].count).map { StageHP(inDim: cols[0][$0], outDim: cols[1][$0], dModel: cols[2][$0], nHead: cols[3][$0], nLayer: cols[4][$0], nFF: cols[5][$0], context: cols[6][$0], patch: cols[7][$0]) }
        guard sampleRate == 44100, hop == 2940, try m.int("internal_hop") == 2944, nCodebooks == 10 else { throw Zonos3Error.invalid("inconsistent decoder dimensions") }
        let wins = try (0..<2).map { try f.table("head.window.\($0)").asArray(Float.self) }
        hostWindows = wins
        let mix = try f.table("head.mixer.weight")
        hostMixer = (0..<2).map { mix[$0].asArray(Float.self) }
        hostKernel = try f.table("resample.kernel").asArray(Float.self)
        let nCB = try m.int("n_codebooks")
        let res = (0..<(nCB - 1)).map { "res.\($0).bias" }
        resBiases = res.allSatisfy { f.t[$0] != nil } ? stacked(res.map { f.t[$0]! }, axis: 0) : nil
        var cb: [MLXArray] = [], sb: [MLXArray] = []
        for (i, n) in (try m.ints("head.n_fft")).enumerated() {
            let bins = n / 2 + 1
            var c = [Float](repeating: 0, count: bins * n), s = [Float](repeating: 0, count: bins * n)
            let win = wins[i]
            for k in 0..<bins {
                let edge = k == 0 || k == bins - 1
                let scale = (edge ? 1.0 : 2.0) / Double(n)
                for t in 0..<n {
                    let ang = 2.0 * Double.pi * Double((k * t) % n) / Double(n)
                    c[k * n + t] = Float(scale * cos(ang) * Double(win[t]))
                    s[k * n + t] = edge ? 0 : Float(-scale * sin(ang) * Double(win[t]))
                }
            }
            cb.append(MLXArray(c, [bins, n])); sb.append(MLXArray(s, [bins, n]))
        }
        cosBasis = cb; sinBasis = sb
        eval(cb + sb)
    }

    public var loadedKeys: Set<String> { f.names }
    /// Whether streams decode frames with missing residual codes (code -1).
    public var earlyDecode: Bool { resBiases != nil }
    public func outputGain(headroomDB db: Double? = nil) -> Float { Float(pow(10.0, -(db ?? headroomDB) / 20.0)) }
    public func stream() throws -> CodecStream { try CodecStream(self) }

    /// Every tensor name the architecture reads (the S0 key contract).
    public func expectedKeys() -> Set<String> {
        var k: Set<String> = ["sem.table", "sem.bias", "res.bias", "sem_head.out.weight", "sem_head.out.bias", "head.norm.weight", "head.norm.bias",
                              "head.out.0.weight", "head.out.0.bias", "head.out.1.weight", "head.out.1.bias", "head.mixer.weight",
                              "head.window.0", "head.window.1", "resample.kernel"]
        for c in 0..<(nCodebooks - 1) { k.formUnion(["res.\(c).table", "res.\(c).bias"]) }
        func block(_ p: String) { k.formUnion(["dw.weight", "dw.bias", "norm.weight", "norm.bias", "pw1.weight", "pw1.bias", "pw2.weight", "pw2.bias"].map { "\(p).\($0)" }) }
        for b in 0..<semDepth { block("sem_head.\(b)") }
        for b in 0..<headDepth { block("head.\(b)") }
        for (s, hp) in stages.enumerated() {
            k.insert("stage.\(s).in.weight")
            if hp.outDim > 0 { k.insert("stage.\(s).out.weight") }
            for l in 0..<hp.nLayer {
                k.formUnion(["norm1.weight", "norm1.bias", "qkv.weight", "wo.weight", "attn_scale", "norm2.weight", "norm2.bias", "ffn1.weight", "ffn2.weight", "ffn_scale"].map { "stage.\(s).l\(l).\($0)" })
            }
        }
        return k
    }

    func lin(_ name: String, _ x: MLXArray, bias: String? = nil) -> MLXArray {
        let y = zmatmul(x, f.w[name]!)
        return bias.map { y + f.t[$0]! } ?? y
    }
    func t(_ name: String) -> MLXArray { f.t[name]! }

    /// x + the residual codebooks of `ids` [n, n_codebooks]; `missing` [n][n_codebooks - 1] marks residual codes left
    /// out: their table rows and biases are not added.
    func residual(_ x0: MLXArray, _ ids: MLXArray, missing: [[Bool]]?) throws -> MLXArray {
        var x = x0
        guard let missing, missing.contains(where: { $0.contains(true) }) else {
            for c in 1..<nCodebooks { x = x + t("res.\(c - 1).table")[ids[0..., c]] }
            return x + t("res.bias")
        }
        guard let resBiases else { throw Zonos3Error.invalid("decoder has no per-codebook residual biases: residual codes cannot be missing") }
        let n = missing.count
        let flat = missing.flatMap { $0 }
        let present = MLXArray(flat.map { !$0 }, [n, nCodebooks - 1])
        for c in 1..<nCodebooks {
            x = x + which(present[0..., (c - 1)..<c], t("res.\(c - 1).table")[ids[0..., c]], MLXArray(Float(0)))
        }
        x = x + t("res.bias")
        let dropped = matmul(MLXArray(flat.map { $0 ? Float(1) : Float(0) }, [n, nCodebooks - 1]), resBiases)
        let any = MLXArray(missing.map { $0.contains(true) }, [n, 1])
        return which(any, x - dropped, x)
    }
}

@inline(__always) private func gelu(_ x: MLXArray) -> MLXArray { 0.5 * x * (1.0 + erf(x * Float(1.0 / 2.0.squareRoot()))) }

/// x [T, C], w [C, K] (tap 0 oldest): h[t] = sum_k x[t + k - (K - 1)] * w[:, k]; the K-1 rows before x are `hist`.
private func causalDepthwise(_ x: MLXArray, _ w: MLXArray, hist: MLXArray?) -> MLXArray {
    let T = x.shape[0], K = w.shape[1]
    let h0 = hist ?? MLXArray.zeros([K - 1, x.shape[1]], dtype: x.dtype)
    let xf = concatenated([h0.asType(x.dtype), x], axis: 0)
    var h = xf[0..<T] * w[0..., 0]
    for k in 1..<K { h = h + xf[k..<(k + T)] * w[0..., k] }
    return h
}

/// out[t*h + j, ch] = in[t, ch*h + j].
private func patchUp(_ x: MLXArray, _ h: Int) -> MLXArray {
    let T = x.shape[0], C = x.shape[1]
    return x.reshaped(T, C / h, h).transposed(0, 2, 1).reshaped(T * h, C / h)
}

/// Incremental decode: `decode(codes)` returns the 44100 Hz samples that are final, `flush()` the rest. Conv histories
/// and per-layer KV caches carry the trunk state. A residual code of -1 is missing (early partial-frame decode).
public final class CodecStream {
    let dec: CodecDecoder
    let maxChunk = 32
    public private(set) var frames = 0
    var semHist: [MLXArray?]
    var headHist: [MLXArray?]
    var kv: [[(MLXArray, MLXArray)?]]
    let head: IstftHeadStream
    let resampler: ResamplerStream

    init(_ dec: CodecDecoder) throws {
        self.dec = dec
        semHist = Array(repeating: nil, count: dec.semDepth)
        headHist = Array(repeating: nil, count: dec.headDepth)
        kv = dec.stages.map { Array(repeating: nil, count: $0.nLayer) }
        head = try IstftHeadStream(nFFTs: dec.nFFT, hop: dec.headHop, windows: dec.hostWindows, mixer: dec.hostMixer)
        resampler = try ResamplerStream(kernel: dec.hostKernel, orig: dec.rs.orig, new: dec.rs.new, width: dec.rs.width)
    }

    /// codes [n][n_codebooks]: codec ids 0..size-1 (residual codebooks: -1 = missing).
    public func decode(_ codes: [[Int32]]) throws -> [Float] {
        let size = Int32(dec.codebookSize)
        for row in codes {
            guard row.count == dec.nCodebooks, row[0] >= 0, row.allSatisfy({ $0 >= -1 && $0 < size }) else {
                throw Zonos3Error.invalid("codes must be codec ids 0..\(size - 1) (residual codebooks: -1 = missing)")
            }
        }
        var out: [Float] = []
        var i = 0
        while i < codes.count {
            out.append(contentsOf: try chunk(Array(codes[i..<min(i + maxChunk, codes.count)])))
            i += maxChunk
        }
        return out
    }

    public func flush() -> [Float] {
        let out = resampler.push(head.flush()) + resampler.flush()
        MLX.Memory.clearCache()
        return out
    }

    private func block(_ x: MLXArray, _ p: String, _ hists: inout [MLXArray?], _ b: Int) -> MLXArray {
        // causal ConvNeXt block over the previous inputs' history; keeps the last K-1 inputs for the next chunk
        let d = dec
        let w = d.t(p + ".dw.weight")
        let k = w.shape[1]
        var h = causalDepthwise(x, w, hist: hists[b]) + d.t(p + ".dw.bias")
        h = MLXFast.layerNorm(h, weight: d.t(p + ".norm.weight"), bias: d.t(p + ".norm.bias"), eps: d.convnextEps)
        h = gelu(d.lin(p + ".pw1.weight", h, bias: p + ".pw1.bias"))
        h = d.lin(p + ".pw2.weight", h, bias: p + ".pw2.bias")
        let hist = hists[b] ?? MLXArray.zeros([k - 1, x.shape[1]], dtype: x.dtype)
        let all = concatenated([hist, x], axis: 0)
        hists[b] = all[(all.shape[0] - (k - 1))...]
        return x + h
    }

    private func mossLayerCached(_ x0: MLXArray, _ p: String, _ hp: StageHP, pos0: Int, cache: (MLXArray, MLXArray)?) -> (MLXArray, (MLXArray, MLXArray)?) {
        // pre-LN windowed-causal attention (adjacent-pair RoPE over absolute stage positions) + GELU FFN, layer scales
        let d = dec
        let T = x0.shape[0], dm = x0.shape[1], nh = hp.nHead, hd = dm / nh
        var x = x0
        var h = MLXFast.layerNorm(x, weight: d.t(p + ".norm1.weight"), bias: d.t(p + ".norm1.bias"), eps: d.lnEps)
        let qkv = d.lin(p + ".qkv.weight", h).reshaped(T, 3, nh, hd).transposed(1, 2, 0, 3)  // [3, nh, T, hd]
        let q = MLXFast.RoPE(qkv[0], dimensions: hd, traditional: true, base: 10000.0, scale: 1.0, offset: pos0)
        var k = MLXFast.RoPE(qkv[1], dimensions: hd, traditional: true, base: 10000.0, scale: 1.0, offset: pos0)
        var v = qkv[2]
        if let cache { k = concatenated([cache.0, k], axis: 1); v = concatenated([cache.1, v], axis: 1) }
        let lc = k.shape[1] - T
        let diff = MLXArray(Int32(pos0)..<Int32(pos0 + T)).expandedDimensions(axis: 1) - MLXArray(Int32(pos0 - lc)..<Int32(pos0 + T)).expandedDimensions(axis: 0)
        let mask = (diff .>= 0) .&& (diff .< Int32(hp.context))
        let o = MLXFast.scaledDotProductAttention(queries: q.expandedDimensions(axis: 0), keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0),
                                                  scale: Float(pow(Double(hd), -0.5)), mask: mask).squeezed(axis: 0)
        x = x + d.lin(p + ".wo.weight", o.transposed(1, 0, 2).reshaped(T, dm)) * d.t(p + ".attn_scale")
        h = MLXFast.layerNorm(x, weight: d.t(p + ".norm2.weight"), bias: d.t(p + ".norm2.bias"), eps: d.lnEps)
        h = d.lin(p + ".ffn2.weight", gelu(d.lin(p + ".ffn1.weight", h)))
        let keep = hp.context - 1
        var newCache: (MLXArray, MLXArray)? = nil
        if keep > 0 {
            let L = k.shape[1], from = max(0, L - keep)
            newCache = (k[0..., from...], v[0..., from...])
        }
        return (x + h * d.t(p + ".ffn_scale"), newCache)
    }

    private func chunk(_ codes: [[Int32]]) throws -> [Float] {
        let d = dec, n = codes.count
        let ids = MLXArray(codes.flatMap { $0.map { max($0, 0) } }, [n, d.nCodebooks])
        var x = d.t("sem.table")[ids[0..., 0]] + d.t("sem.bias")
        for b in 0..<d.semDepth { x = block(x, "sem_head.\(b)", &semHist, b) }
        x = d.lin("sem_head.out.weight", x, bias: "sem_head.out.bias")
        x = try d.residual(x, ids, missing: codes.map { $0[1...].map { $0 < 0 } })
        eval(x)
        var cpf = 1  // stage positions per code frame
        for (s, hp) in d.stages.enumerated() {
            let p = "stage.\(s)"
            x = d.lin(p + ".in.weight", x)
            for layer in 0..<hp.nLayer {
                let (y, cache) = mossLayerCached(x, "\(p).l\(layer)", hp, pos0: frames * cpf, cache: kv[s][layer])
                x = y; kv[s][layer] = cache
                eval([x] + (cache.map { [$0.0, $0.1] } ?? []))
            }
            if hp.outDim > 0 { x = d.lin(p + ".out.weight", x) }
            if hp.patch > 1 { x = patchUp(x, hp.patch); cpf *= hp.patch }
        }
        for b in 0..<d.headDepth { x = block(x, "head.\(b)", &headHist, b); eval(x) }
        x = MLXFast.layerNorm(x, weight: d.t("head.norm.weight"), bias: d.t("head.norm.bias"), eps: d.convnextEps)
        var windowed: [[Float]] = []
        var cols = 0
        var pending: [MLXArray] = []
        for i in 0..<2 {
            let y = d.lin("head.out.\(i).weight", x, bias: "head.out.\(i).bias")  // [cols, n + 2] = logmag | phase
            let bins = d.nFFT[i] / 2 + 1
            let mag = exp(minimum(y[0..., 0..<bins], MLXArray(d.maxLogMag)))
            let ph = y[0..., bins..<(2 * bins)]
            pending.append(matmul(mag * cos(ph), d.cosBasis[i]) + matmul(mag * sin(ph), d.sinBasis[i]))
            cols = y.shape[0]
        }
        eval(pending + semHist.compactMap { $0 } + headHist.compactMap { $0 })
        for a in pending { windowed.append(a.asArray(Float.self)) }
        frames += n
        return resampler.push(head.push(windowed, cols: cols))
    }
}
