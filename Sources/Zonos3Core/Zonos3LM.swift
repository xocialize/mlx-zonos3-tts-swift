// Zonos3LM.swift — ZONOS3 Mini LM: the GDN / global-attention backbone with Full AttnRes and the latent MoE.
// A 1:1 port of zonos3_mlx/models/lm.py (Zyphra, MIT) @ 4c66cea, without the probe / accent-adapter paths.
// Activations are float32, dense weights float16 (upcast per matmul), quantized mixers / experts stay MLX affine.
// One sequence per forward: `ids` is [rows, n_codebooks + 1].
import Foundation
import MLX
import MLXFast

public struct LMConfig {
    public let nLayers, maxSeqlen, dim, headDim, nHeads, nKVHeads, ffnHidden, nExperts, moeTopK: Int
    public let nCodebooks, codebookSize, audioVocab, eoaID, audioPadID, textVocab, textExtraTokens: Int
    public let gdnHeads, gdnHeadDim, gdnKernel, speakerEmbeddingDim, conditionCount: Int
    public let conditionBucketCounts: [Int]
    public let conditionNames: [String]
    public let conditionEdges: [String: [Double]]
    public let ropeTheta: Float, normEps: Float, qkEps: Float, lossSoftcap: Float
    public let gdnPeriod: Int
    public let silenceFrame: [Int]

    public var nSources: Int { 2 * nLayers + 1 }
    public func usesGDN(_ layer: Int) -> Bool { (layer + 1) % gdnPeriod != 0 }
    public var gdnInner: Int { gdnHeads * gdnHeadDim }

    public init(meta m: Meta, textEmbdRows: Int) throws {
        guard try m.string("residual.type") == "full_attnres", try m.string("moe.balancing") == "quantile" else {
            throw Zonos3Error.invalid("unsupported residual / MoE balancing")
        }
        let pattern = try m.string("attention.pattern")
        guard pattern.hasPrefix("gdn"), pattern.hasSuffix("_global1"), let g = Int(pattern.dropFirst(3).dropLast("_global1".count)) else {
            throw Zonos3Error.invalid("unsupported attention pattern \(pattern)")
        }
        nLayers = try m.int("block_count"); maxSeqlen = try m.int("context_length"); dim = try m.int("embedding_length")
        headDim = try m.int("head_dim"); nHeads = try m.int("attention.head_count"); nKVHeads = try m.int("attention.head_count_kv")
        ffnHidden = try m.int("feed_forward_length"); nExperts = try m.int("expert_count")
        moeTopK = try m.ints("expert_used_count_per_layer")[1]
        nCodebooks = try m.int("n_codebooks"); codebookSize = try m.int("codebook_size"); audioVocab = try m.int("audio_vocab")
        eoaID = try m.int("eoa_id"); audioPadID = try m.int("audio_pad_id"); textVocab = try m.int("text_vocab")
        textExtraTokens = textEmbdRows - textVocab - 1
        gdnHeads = try m.int("gdn.head_count"); gdnHeadDim = try m.int("gdn.head_dim"); gdnKernel = try m.int("gdn.conv_kernel")
        speakerEmbeddingDim = try m.int("speaker.embedding_length"); conditionCount = try m.int("condition.count")
        conditionBucketCounts = try m.ints("condition.bucket_counts")
        conditionNames = try m.strings("condition.names")
        var edges: [String: [Double]] = [:]
        for n in conditionNames { edges[n] = try m.doubles("condition.edges.\(n)") }
        conditionEdges = edges
        ropeTheta = Float(try m.double("rope.freq_base")); normEps = Float(try m.double("attention.layer_norm_rms_epsilon"))
        qkEps = Float(try m.double("attention.qk_norm_epsilon")); lossSoftcap = Float(try m.double("logit_softcap"))
        gdnPeriod = g + 1
        silenceFrame = try m.ints("silence_frame")
    }
}

/// Decoding state of one sequence: the next absolute position, the global layers' keys / values ([n_kv_heads, rows,
/// head_dim]), each GDN layer's recurrent state ([heads, value, key]) and causal-conv history ([channels, kernel - 1]).
public final class LMState {
    public var position = 0
    var kv: [Int: (MLXArray, MLXArray)] = [:]
    var gdnState: [Int: MLXArray] = [:]
    var convState: [Int: MLXArray] = [:]
    public init() {}
    public func arrays() -> [MLXArray] {
        kv.values.flatMap { [$0.0, $0.1] } + Array(gdnState.values) + Array(convState.values)
    }
}

@inline(__always) private func rms(_ x: MLXArray, _ weight: MLXArray?, _ eps: Float) -> MLXArray {
    // mx.fast.rms_norm(x, None, eps): x * rsqrt(mean(x^2) + eps)
    if let weight { return MLXFast.rmsNorm(x, weight: weight, eps: eps) }
    return x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps)
}
@inline(__always) private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }
@inline(__always) private func softplus(_ x: MLXArray) -> MLXArray { logAddExp(x, MLXArray.zeros(like: x)) }

/// One recurrent step: q, k [H, Dk]; v [H, Dv]; g, beta [H]; state [H, Dv, Dk] -> (y [H, Dv], state).
private func gatedDeltaStep(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ g: MLXArray, _ beta: MLXArray, _ state0: MLXArray) -> (MLXArray, MLXArray) {
    var state = state0 * g.expandedDimensions(axes: [-1, -2])
    let kRow = k.expandedDimensions(axis: -2)  // [H, 1, Dk]
    let kvMem = (state * kRow).sum(axis: -1)  // [H, Dv]
    let delta = (v - kvMem) * beta.expandedDimensions(axis: -1)
    state = state + kRow * delta.expandedDimensions(axis: -1)
    let y = (state * q.expandedDimensions(axis: -2)).sum(axis: -1)
    return (y, state)
}

public final class Zonos3LM {
    public let config: LMConfig
    let f: WeightFile
    /// Optional capture of intermediate tensors (parity gates): name -> array, as upstream's `capture`.
    public var capture: [String: MLXArray]? = nil

    public init(config: CheckpointConfig, file: String = "lm/model.safetensors") throws {
        let quant = config.quantized(file)
        f = try WeightFile(url: config.root.appendingPathComponent(file), quantized: quant) { name, shape in
            // float32 tables: embeddings, AttnRes queries, 1-D tensors and the GDN conv taps; the rest are matmul weights
            shape.count == 1 || name.hasSuffix(".gdn_conv.weight") || name == "attnres_q.weight"
                || name.hasPrefix("audio_embd.") || name.hasPrefix("text_embd.") || name.hasPrefix("cond_embd.")
        }
        let meta = Meta(try config.meta(file), arch: "zonos3")
        self.config = try LMConfig(meta: meta, textEmbdRows: try f.table("text_embd.weight").shape[0])
    }

    /// Every tensor name the architecture reads, generated from the config (the S0 key contract).
    public func expectedKeys() -> Set<String> {
        let c = config
        var k: Set<String> = ["text_embd.weight", "output_norm.weight", "output.weight", "attnres_q.weight", "speaker_proj.weight", "speaker_proj.bias"]
        for cb in 0..<c.nCodebooks { k.insert("audio_embd.\(cb).weight") }
        for t in 0..<c.conditionCount { k.insert("cond_embd.\(t).weight") }
        for l in 0..<c.nLayers {
            let p = "blk.\(l)."
            k.formUnion([p + "attn_norm.weight", p + "ffn_norm.weight", p + "attn_output.weight"])
            if c.usesGDN(l) {
                k.formUnion([p + "gdn_in.weight", p + "gdn_norm.weight", p + "gdn_A_log", p + "gdn_dt_bias", p + "gdn_conv.weight"])
            } else {
                k.formUnion([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "attn_temp", p + "attn_gate.weight"])
            }
            if l == 0 {
                k.formUnion([p + "ffn_up.weight", p + "ffn_gate.weight", p + "ffn_down.weight"])
            } else {
                k.formUnion([p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight", p + "ffn_down_exps.weight", p + "ffn_latent_down.weight",
                             p + "ffn_latent_up.weight", p + "router.weight", p + "router_norm.weight", p + "router_bias"])
            }
        }
        return k
    }
    public var loadedKeys: Set<String> { f.names }

    private func save(_ name: String, _ value: MLXArray) { if capture != nil { capture![name] = value.asType(.float32) } }
    private func lin(_ base: String, _ x: MLXArray) -> MLXArray { zmatmul(x, f.w[base + ".weight"]!) }
    private func t(_ name: String) -> MLXArray { f.t[name]! }

    // MARK: sublayers

    private func globalAttention(_ x: MLXArray, _ layer: Int, _ state: LMState) -> MLXArray {
        let c = config, p = "blk.\(layer)."
        let n = x.shape[0], nh = c.nHeads, nkv = c.nKVHeads, hd = c.headDim
        let lo = state.position
        let gate = sigmoid(lin(p + "attn_gate", x))  // [n, nh]
        var q = lin(p + "attn_q", x).reshaped(n, nh, hd)
        var k = lin(p + "attn_k", x).reshaped(n, nkv, hd)
        var v = lin(p + "attn_v", x).reshaped(n, nkv, hd)
        q = rms(q, nil, c.qkEps) * t(p + "attn_temp").expandedDimensions(axis: -1)  // [nh, 1] against [n, nh, hd]
        k = rms(k, nil, c.qkEps)
        q = MLXFast.RoPE(q.transposed(1, 0, 2), dimensions: hd, traditional: true, base: c.ropeTheta, scale: 1.0, offset: lo)
        k = MLXFast.RoPE(k.transposed(1, 0, 2), dimensions: hd, traditional: true, base: c.ropeTheta, scale: 1.0, offset: lo)
        v = v.transposed(1, 0, 2)
        let keys: MLXArray, values: MLXArray
        if let (kc, vc) = state.kv[layer] {
            keys = concatenated([kc, k], axis: 1); values = concatenated([vc, v], axis: 1)
        } else {
            keys = k; values = v
        }
        state.kv[layer] = (keys, values)
        let need = lo + n
        var mask: MLXArray? = nil
        if n > 1 {
            // key j visible to query row i when j <= lo + i
            mask = MLXArray(0..<Int32(need)).expandedDimensions(axis: 0) .<= (MLXArray(0..<Int32(n)) + Int32(lo)).expandedDimensions(axis: 1)
        }
        // grouped-query attention: head h reads KV head h / (nh / nkv)
        let rep = nh / nkv
        let kk = repeated(keys, count: rep, axis: 0).expandedDimensions(axis: 0)
        let vv = repeated(values, count: rep, axis: 0).expandedDimensions(axis: 0)
        var out = MLXFast.scaledDotProductAttention(queries: q.expandedDimensions(axis: 0), keys: kk, values: vv, scale: Float(pow(Double(hd), -0.5)), mask: mask)
        out = out.squeezed(axis: 0).transposed(1, 0, 2) * gate.expandedDimensions(axis: -1)  // [n, nh, hd]
        return lin(p + "attn_output", out.reshaped(n, nh * hd))
    }

    private func gdn(_ x: MLXArray, _ layer: Int, _ state: LMState) -> MLXArray {
        let c = config, p = "blk.\(layer)."
        let H = c.gdnHeads, S = c.gdnHeadDim, K = c.gdnKernel
        let inner = H * S
        let n = x.shape[0]
        let projection = lin(p + "gdn_in", x)
        let qkv = projection[0..., 0..<(3 * inner)]
        let z = projection[0..., (3 * inner)..<(4 * inner)]
        let a = projection[0..., (4 * inner)..<(4 * inner + H)]
        let b = projection[0..., (4 * inner + H)..<(4 * inner + 2 * H)]
        let history = state.convState[layer] ?? MLXArray.zeros([3 * inner, K - 1], dtype: .float32)
        let sequence = concatenated([history.T, qkv], axis: 0)  // [K - 1 + n, channels], oldest first
        let weight = t(p + "gdn_conv.weight")  // [channels, K], tap 0 oldest
        var conv = sequence[0..<n] * weight[0..., 0]
        for j in 1..<K { conv = conv + sequence[j..<(j + n)] * weight[0..., j] }
        conv = silu(conv)
        state.convState[layer] = sequence[(sequence.shape[0] - (K - 1))...].T
        var q = conv[0..., 0..<inner].reshaped(n, H, S)
        var k = conv[0..., inner..<(2 * inner)].reshaped(n, H, S)
        let v = conv[0..., (2 * inner)...].reshaped(n, H, S)
        q = q * rsqrt((q * q).sum(axis: -1, keepDims: true) + 1e-6)
        k = k * rsqrt((k * k).sum(axis: -1, keepDims: true) + 1e-6)
        let decay = exp(-exp(t(p + "gdn_A_log")) * softplus(a + t(p + "gdn_dt_bias")))  // [n, H]
        let beta = sigmoid(b)
        var s = state.gdnState[layer] ?? MLXArray.zeros([H, S, S], dtype: .float32)
        let qs = q * Float(pow(Double(S), -0.5))
        let recurrent: MLXArray  // [n, H, S]
        if n == 1 {
            let (y, s1) = gatedDeltaStep(qs[0], k[0], v[0], decay[0], beta[0], s)
            recurrent = y.expandedDimensions(axis: 0); s = s1
        } else {
            // multi-row calls (prefill) run the Metal kernel, one row the ops step — upstream's split
            (recurrent, s) = GatedDeltaKernel.run(qs, k, v, decay, beta, s)
        }
        state.gdnState[layer] = s
        let normalized = rms(recurrent, t(p + "gdn_norm.weight"), 1e-5)
        let gated = (normalized * silu(z.reshaped(n, H, S))).reshaped(n, inner)
        if capture != nil {
            save("gdn_conv_\(layer)", conv); save("gdn_q_\(layer)", q); save("gdn_k_\(layer)", k); save("gdn_v_\(layer)", v)
            save("gdn_decay_\(layer)", decay); save("gdn_beta_\(layer)", beta); save("gdn_recurrent_\(layer)", recurrent)
            save("gdn_gated_\(layer)", gated); save("gdn_state_\(layer)", s); save("gdn_conv_state_\(layer)", state.convState[layer]!)
        }
        return lin(p + "attn_output", gated)
    }

    private func gatherMatmul(_ x: MLXArray, _ W: Weight, _ idx: MLXArray) -> MLXArray {
        // per routed pair x[t, r] @ W[idx[t, r]].T: x [n, k, 1, in], idx [n, k] -> [n, k, 1, out]
        if let s = W.scales {
            return gatherQuantizedMM(x, W.w, scales: s, biases: W.biases, rhsIndices: idx, transpose: true, groupSize: Weight.groupSize, bits: W.bits)
        }
        return gatherMM(x, W.w.asType(.float32).swappedAxes(-1, -2), rhsIndices: idx)
    }

    private func feedForward(_ x: MLXArray, _ layer: Int) -> MLXArray {
        let c = config, p = "blk.\(layer)."
        if f.w[p + "ffn_up.weight"] != nil {
            return lin(p + "ffn_down", lin(p + "ffn_up", x) * silu(lin(p + "ffn_gate", x)))
        }
        let logits = lin(p + "router", rms(x, t(p + "router_norm.weight"), c.normEps))
        save("router_logits_\(layer)", logits)
        let bias = t(p + "router_bias")
        let k = c.moeTopK
        let choices = argPartition(bias - logits, kth: k - 1, axis: -1)[0..., 0..<k]
        let weights = softmax(takeAlong(logits, choices, axis: -1), axis: -1)
        let latent = lin(p + "ffn_latent_down", x)
        let z = latent.expandedDimensions(axes: [1, 2])  // [n, 1, 1, latent]
        let gate = gatherMatmul(z, f.w[p + "ffn_gate_exps.weight"]!, choices)
        let up = gatherMatmul(z, f.w[p + "ffn_up_exps.weight"]!, choices)
        let hidden = silu(gate) * up
        let down = gatherMatmul(hidden, f.w[p + "ffn_down_exps.weight"]!, choices)
        let output = (down.squeezed(axis: 2) * weights.expandedDimensions(axis: -1)).sum(axis: 1)
        return lin(p + "ffn_latent_up", output)
    }

    // MARK: forward

    /// Logits [rows, n_codebooks, audio_vocab] (softcapped, lazy) of `ids` [rows, n_codebooks + 1] continuing `state`
    /// (updated in place). `speakerRows`: absolute row -> 192-d embedding replacing that row's embedding sum;
    /// `conditionRows`: absolute row -> bucket id, tables in ascending row order.
    public func forward(_ ids: MLXArray, state: LMState, speakerRows: [Int: MLXArray] = [:], conditionRows: [Int: Int] = [:]) throws -> MLXArray {
        let c = config
        let idx = ids.asType(.int32)
        let n = idx.shape[0]
        let lo = state.position
        guard lo + n <= c.maxSeqlen else { throw Zonos3Error.invalid("\(lo + n) rows exceed the model context \(c.maxSeqlen)") }
        var x = t("audio_embd.0.weight")[idx[0..., 0]]
        for cb in 1..<c.nCodebooks { x = x + t("audio_embd.\(cb).weight")[idx[0..., cb]] }
        x = x + t("text_embd.weight")[idx[0..., c.nCodebooks]]
        save("emb_sum", x)
        if !speakerRows.isEmpty || !conditionRows.isEmpty {
            var rows = (0..<n).map { x[$0] }
            for (row, vec) in speakerRows where row >= lo && row < lo + n {
                let projected = zmatmul(vec.asType(.float32).reshaped(1, -1), f.w["speaker_proj.weight"]!)[0]
                rows[row - lo] = projected + t("speaker_proj.bias")
            }
            for (table, row) in conditionRows.keys.sorted().enumerated() where row >= lo && row < lo + n {
                rows[row - lo] = t("cond_embd.\(table).weight")[conditionRows[row]!]
            }
            x = stacked(rows, axis: 0)
        }
        save("emb_cond", x)
        x = rms(x, nil, c.normEps)
        save("emb_norm", x)

        let queries = t("attnres_q.weight")  // [n_sources, dim]
        var bank: [MLXArray] = []
        var scores: [MLXArray] = []
        func route(_ source: MLXArray, _ query: Int) -> MLXArray {
            bank.append(source)
            scores.append(matmul(rms(source, nil, c.normEps), queries.T))  // [rows, n_sources]
            if bank.count == 1 { return source }
            let w = softmax(stacked(scores.map { $0[0..., query] }, axis: 0), axis: 0)  // [s, rows]
            return (stacked(bank, axis: 0) * w.expandedDimensions(axis: -1)).sum(axis: 0)
        }

        var mixed = route(x, 0)
        for layer in 0..<c.nLayers {
            let p = "blk.\(layer)."
            if layer > 0 { mixed = route(x, 2 * layer) }
            save("attnres_attn_\(layer)", mixed)
            let attnIn = rms(mixed, t(p + "attn_norm.weight"), c.normEps)
            save("attn_in_\(layer)", attnIn)
            let attention = c.usesGDN(layer) ? gdn(attnIn, layer, state) : globalAttention(attnIn, layer, state)
            save("attn_out_\(layer)", attention)
            mixed = route(attention, 2 * layer + 1)
            save("attnres_ffn_\(layer)", mixed)
            let ffnIn = rms(mixed, t(p + "ffn_norm.weight"), c.normEps)
            save("ffn_in_\(layer)", ffnIn)
            x = feedForward(ffnIn, layer)
            save("ffn_out_\(layer)", x)
            save("layer_x_\(layer)", x)
        }
        mixed = route(x, 2 * c.nLayers)
        save("attnres_final", mixed)
        let normalized = rms(mixed, t("output_norm.weight"), c.normEps)
        save("out_norm", normalized)
        var logits = lin("output", normalized).reshaped(n, c.nCodebooks, c.audioVocab)
        logits = c.lossSoftcap * tanh(logits / c.lossSoftcap)
        save("logits", logits)
        state.position = lo + n
        return logits
    }
}
