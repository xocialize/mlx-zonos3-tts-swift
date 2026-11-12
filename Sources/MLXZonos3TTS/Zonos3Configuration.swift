// Zonos3Configuration.swift — init-time configuration for `Zonos3Package` (C9). Per-request text / voice / metaData ride
// the canonical `TTSRequest`.
//
// ONE weight set: the converter's checkpoint (`Tools/convert_gguf.py` + `Tools/pack_voices.py`) — `lm/model.safetensors`
// (backbone: 4-bit experts, 8-bit mixers and logits head), `codec/` (the `keep2_plain` decoder), `speaker_encoder/`,
// `config.json`, `tokenizer.json`, `voices.json`. `modelDirectory` is the dev escape hatch.

import Foundation
import MLXToolKit

public struct Zonos3Configuration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The backbone's tier. Only `.int4` is offered: Zyphra's own post-training-quantized release (4-bit experts, 8-bit
    /// mixers and head), repacked exactly; the decoder and speaker encoder are fp16 / fp32 inside it.
    public var quant: Quant
    /// Revision of the weights repo; nil = `revisionPin`.
    public var revision: String?
    /// Explicit checkpoint directory (the converter's layout) — never touches the network.
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// MLX buffer-cache limit while the model is resident (the fleet's bounded-cache rule; the footprints were measured
    /// at this value).
    public var cacheLimitBytes: UInt64

    public init(quant: Quant = .int4, revision: String? = nil, modelDirectory: URL? = nil, modelsRootDirectory: URL? = nil,
                cacheLimitBytes: UInt64 = 1 << 30) {
        self.quant = quant; self.revision = revision; self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory; self.cacheLimitBytes = cacheLimitBytes
    }

    /// NOT PUBLISHED YET (2026-11-11): the name is a placeholder until the operator authorizes a publish and its name
    /// (`mlx-community/ZONOS3-Mini` is mlx-audio's conversion, a different layout). Until then the package loads from
    /// `modelDirectory` only, and there is no revision pin.
    public static let repo = "mlx-community/ZONOS3-Mini-swift-ptq"
    public static let revisionPin: String? = nil
    public static let isPublished = false
    /// The upstream repository and revision the converter read.
    public static let upstreamRepo = "Zyphra/ZONOS3-Mini"
    public static let upstreamRevision = "7146a686b62bcb0661528fa300984e815e10152b"

    public var effectiveRevision: String? { revision ?? Self.revisionPin }

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey { case quant, revision, cacheLimitBytes }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quant = try c.decodeIfPresent(Quant.self, forKey: .quant) ?? .int4
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        cacheLimitBytes = try c.decodeIfPresent(UInt64.self, forKey: .cacheLimitBytes) ?? (1 << 30)
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension Zonos3Configuration: WeightSourcing {
    /// The files `Zonos3TTS` opens with the small decoder. `codec-full/` (the 2.2 GB `full_mr` decoder) is excluded:
    /// every weight glob names its directory, because a glob without a `/` also matches on the basename alone and a bare
    /// `model.safetensors` would pull all four.
    public static let files = ["config.json", "tokenizer.json", "voices.json", "lm/*.safetensors", "codec/*.safetensors", "speaker_encoder/*.safetensors"]
    /// Representative file for the missing-probe (the last component `load()` opens).
    static let probe = "voices.json"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "checkpoint", repo: Self.repo, revision: effectiveRevision, matching: Self.files)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let modelDirectory {
            return FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent(Self.probe).path) ? [] : weightSources
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// The checkpoint directory `load()` opens: the explicit one, else the repo's snapshot or flat directory under the
    /// store. Nil when it does not resolve.
    public func resolvedDirectory(storeRoot: URL?) -> URL? {
        if let modelDirectory { return modelDirectory }
        let store = ModelStore(root: storeRoot)
        func has(_ dir: URL) -> Bool { FileManager.default.fileExists(atPath: dir.appendingPathComponent(Self.probe).path) }
        if let snapshot = store.snapshotDirectory(for: Self.repo, revision: effectiveRevision), has(snapshot) { return snapshot }
        if let flat = store.directory(for: Self.repo), has(flat) { return flat }
        return nil
    }
}

// MARK: - Cold-start prewarm

extension Zonos3Configuration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolvedDirectory(storeRoot: modelsRootDirectory) else { return [] }
        return [dir.appendingPathComponent("lm/model.safetensors"), dir.appendingPathComponent("codec/model.safetensors")]
    }
}
