import Foundation
import MLX
import MLXToolKit
import Zonos3Core

/// Zyphra's ZONOS3-Mini on the canonical `tts` surface: real-time multilingual speech at 44.1 kHz mono, in one of 27
/// shipped voices or a voice cloned from a reference clip (E30; operator approval AB-D-0225).
///
/// Engine-owned lifecycle (C13): the engine constructs from a `Zonos3Configuration`, materializes the declared
/// checkpoint into its store (or the host names one), pages it in with `load()`, drives `run(_:)` / `runStream(_:emit:)`,
/// and reclaims with `unload()`.
///
/// Voice (`TTSRequest.voice`):
/// - `.named(id)`: one of `Zonos3Package.voiceIDs` once loaded (the 27 voices Zyphra ships, e.g. "savannah", "jett",
///   "gaspard"), case-insensitive; "none" speaks with no voice conditioning.
/// - `.referenceAudio(clip)`: clones the clip's speaker from a 192-d embedding of at most its first 60 s. No transcript
///   is needed; `referenceTranscript` is ignored.
/// - `.auto`: the default voice, "savannah".
///
/// Cloning is weaker than the fleet's Character engines: cosine 0.79–0.80 to the reference where VoxCPM2 scores 0.875 on
/// the same voice (E30 V0). Text is spoken as written — there is no text normalization, so numbers and abbreviations
/// should be written out.
///
/// `metaData` keys (package-specific, C5):
/// - `seed` (int): reproducible sampling within Swift-MLX. Random when absent.
/// - `temperature` (double, default 0.75), `topK` (int, 80), `topP` (double, 0.69): the reference sampler's knobs.
/// - `speakingRate` (double, UTF-8 bytes per second, ~6.5–29), `loudnessLUFS` (double), `fmaxHz` (double): the model's
///   three bucketed acoustic conditions; absent = the model's "unknown" bucket.
/// - `maxFrames` (int, default 2000 = 133 s): generation cap, 15 frames per second of audio.
/// - `voiceCacheKey` (string) and `evictVoiceCacheKeys` (string or array of strings): below.
///
/// **Voice cache (MES Base §13 rules 4 and 6; the mlx-qwen3-tts-swift 0.7.0 API).** Without
/// `metaData["voiceCacheKey"]`, nothing derived from a reference outlives the call. With it, the clip's speaker
/// embedding is kept under that key (4 keys, least recently used dropped; a key reused with other bytes re-embeds).
/// `metaData["evictVoiceCacheKeys"]` drops keys before anything else runs, loaded or not; with an empty `text` the call
/// only evicts. `unload()` drops every key.
@InferenceActor
public final class Zonos3Package: ModelPackage, StreamEmitting {
    public typealias Configuration = Zonos3Configuration

    /// Footprints MEASURED as phys_footprint, pool-inclusive (AB-L-0113 / AB-L-0155), MLX cache 1 GiB, through MLXServeEngine
    /// (`zonos3-gates validate`, M5 Max, 2026-11-11; MEASUREMENTS.md): resident = phys after `prepare`, 1.60 GB in all
    /// three runs; activation = the lifetime peak over the validate requests minus the resident, 1.16 – 1.19 GB — the same with
    /// a 68 s request as with 6 s ones (the decode is incremental), and it includes one reference-clip embedding.
    /// Declared with the fleet's margins rounded up: 1.75 + 1.5 GB.
    nonisolated static let int4ResidentBytes: UInt64 = 1_750_000_000
    nonisolated static let int4PeakActivationBytes: UInt64 = 1_500_000_000

    public nonisolated static let sampleRate = 44_100
    public nonisolated static let defaultVoice = "savannah"
    public nonisolated static let voiceCacheCapacity = 4
    public nonisolated static let defaultMaxFrames = 2000
    /// The only tier offered.
    public nonisolated static let offeredQuants: [Quant] = [.int4]

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the weights are Zyphra's, Apache-2.0. C8: the port code is MIT — a translation of Zyphra's MIT runtime
            // (its own MLX implementation); the gated-delta Metal kernel is mlx-lm's, MIT.
            license: LicenseDeclaration(
                weightLicense: .apache2, portCodeLicense: .mit,
                notices: [
                    LicenseNotice(text: "Text is spoken as written: there is no text normalization, so write numbers, dates and "
                                  + "abbreviations out.", display: .informational),
                    LicenseNotice(text: "Voice cloning is approximate (speaker similarity below the fleet's Character engines). Clone a "
                                  + "voice only with its owner's consent.", display: .informational),
                ]),
            provenance: Provenance(sourceRepo: Zonos3Configuration.upstreamRepo, revision: Zonos3Configuration.upstreamRevision, tier: 2),
            requirements: RequirementsManifest(
                footprints: [QuantFootprint(quant: .int4, residentBytes: int4ResidentBytes, peakActivationBytes: int4PeakActivationBytes)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [
                SpecialtyWeight(.realtimeStreaming, strength: 1.0),
                SpecialtyWeight(.voiceClone, strength: 0.5),
            ],
            surfaces: [
                TTSContract.descriptor(
                    name: "zonos3-mini",
                    summary: "ZONOS3-Mini (Zyphra, Apache-2.0): real-time multilingual TTS, .wav 44.1 kHz mono, about 10x real time "
                        + "with first audio in tens of milliseconds. 27 named voices across English, French, German, Hindi, Italian, "
                        + "Korean, Mandarin, Portuguese, Spanish and Telugu (voice.named, e.g. savannah, jett, rupert, gaspard, "
                        + "meilin; 'none' = no voice), or a voice cloned from voice.referenceAudio with no transcript (approximate: "
                        + "below the Character engines). The model covers 43 languages. Text is spoken as written — write numbers "
                        + "out. Streams PCM chunks as frames are generated. metaData: seed, temperature, topK, topP, speakingRate "
                        + "(UTF-8 bytes per second), loudnessLUFS, fmaxHz, maxFrames, voiceCacheKey, evictVoiceCacheKeys.",
                    modes: [.neutral],
                    streaming: .audioChunk
                )
            ]
        )
    }

    private let configuration: Configuration
    private var tts: Zonos3TTS?
    private var voices: [String: [Float]] = [:]
    private var voiceCache = VoiceFeatureCache<CachedVoice>(capacity: Zonos3Package.voiceCacheCapacity)
    /// How many reference clips were embedded (tests: the cache's effect).
    private(set) var embeddingBuilds = 0

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// The named voices, once loaded (sorted).
    public var voiceIDs: [String] { voices.keys.sorted() }
    var cachedVoiceKeys: [String] { voiceCache.keys }

    // MARK: - Lifecycle

    public func load() async throws {
        guard tts == nil else { return }
        guard Self.offeredQuants.contains(configuration.quant) else {
            throw Zonos3PackageError.unsupportedTier("quant \(configuration.quant) — this package offers int4 only (Zyphra's quantized release)")
        }
        // Materialization is ENGINE-EXECUTED (contract 1.24): this guard is the offline backstop only.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw Zonos3PackageError.missingWeights(
                "source not materialized: \(missing.map(\.repo).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()
        guard let directory = configuration.resolvedDirectory(storeRoot: storeRoot) else {
            throw Zonos3PackageError.missingWeights("unresolved checkpoint directory (no store root)")
        }
        MLX.Memory.cacheLimit = Int(configuration.cacheLimitBytes)
        do {
            // Weight loads ride the CPU stream inside the Core (WeightFile) — the Metal-watchdog rule.
            let model = try Zonos3TTS(directory: directory)
            voices = try Self.readVoices(directory.appendingPathComponent("voices.json"), dim: model.lm.config.speakerEmbeddingDim)
            guard voices[Self.defaultVoice] != nil else { throw Zonos3Error.invalid("voices.json has no default voice \(Self.defaultVoice)") }
            tts = model
        } catch let error as Zonos3Error {
            throw Zonos3PackageError.loadFailed(error.description)
        }
        MLX.Memory.clearCache()
    }

    public func unload() async {
        tts = nil
        voices = [:]
        voiceCache.removeAll()
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    nonisolated static func readVoices(_ url: URL, dim: Int) throws -> [String: [Float]] {
        guard let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any], let list = doc["voices"] as? [[String: Any]] else {
            throw Zonos3Error.invalid("voices.json is not a voice list")
        }
        var out: [String: [Float]] = [:]
        for v in list {
            guard let id = v["id"] as? String, let e = v["embedding"] as? [NSNumber], e.count == dim else { throw Zonos3Error.invalid("voices.json: a voice without an id or a \(dim)-d embedding") }
            out[id.lowercased()] = e.map(\.floatValue)
        }
        return out
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try await synthesize(request, emit: nil)
    }

    /// Streaming twin of `run()`: the same plan and generation, with each PCM chunk emitted as it becomes final (the
    /// first after one generated frame). Returns the same aggregated response.
    public func runStream(_ request: any CapabilityRequest, emit: @escaping @Sendable (TTSStreamChunk) -> Void) async throws -> any CapabilityResponse {
        try await synthesize(request, emit: emit)
    }

    private func synthesize(_ request: any CapabilityRequest, emit: (@Sendable (TTSStreamChunk) -> Void)?) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(). Mid-run cadence: once per generated frame, on the same
        // seam as the RunProgress report; the CancellationError is rethrown UNCHANGED.
        try Task.checkCancellation()
        guard request.capability == .tts, let req = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        // Eviction first, and without weights: a key dropped on asset deletion must go even when the package isn't loaded.
        if let keys = req.metaData.stringList("evictVoiceCacheKeys") {
            for key in keys { voiceCache.remove(key) }
            if req.text.isEmpty { return Self.response([]) }
        }
        guard let tts else { throw PackageError.notLoaded }
        let plan = try Self.plan(req)
        let embedding = try voice(req, tts: tts)
        try Task.checkCancellation()

        var index = 0
        var pending: [Float]? = nil  // one chunk held back so the last one can carry isFinal
        let result: SynthesisResult
        do {
            result = try tts.synthesize(
                text: plan.text, speakerEmbedding: embedding, options: plan.options,
                onRow: { rows in
                    RunProgress.report(.generate, step: rows)
                    try Task.checkCancellation()
                },
                onChunk: { chunk in
                    if let emit {
                        if let held = pending { emit(TTSStreamChunk(samples: held, sampleRate: Self.sampleRate, index: index, isFinal: false)); index += 1 }
                        pending = chunk
                    }
                    return true
                })
        } catch let error as Zonos3Error {
            throw Zonos3PackageError.synthesisFailed(error.description)
        }
        try Task.checkCancellation()
        if let emit { emit(TTSStreamChunk(samples: pending ?? [], sampleRate: Self.sampleRate, index: index, isFinal: true)) }
        RunProgress.report(.postprocess)
        MLX.Memory.clearCache()
        return Self.response(result.pcm)
    }

    nonisolated static func response(_ samples: [Float]) -> TTSResponse {
        TTSResponse(audio: Audio(format: .wav, data: Zonos3AudioIO.encodeWAV16(samples: samples, sampleRate: sampleRate), sampleRate: sampleRate, channels: 1))
    }

    /// The request's text and options (everything but the voice), judged without weights.
    public struct Plan: Equatable {
        public let text: String
        public let seed: UInt64
        public let temperature: Double, topP: Double
        public let topK: Int, maxFrames: Int
        public let speakingRate: Double?, loudnessLUFS: Double?, fmaxHz: Double?

        var options: SynthesisOptions {
            var o = SynthesisOptions()
            o.seed = seed; o.maxFrames = maxFrames
            o.sampling.temperature = temperature; o.sampling.topK = topK; o.sampling.topP = topP
            o.conditions = Conditions(loudnessLUFS: loudnessLUFS, fmaxHz: fmaxHz, speakingRate: speakingRate)
            return o
        }
    }

    public nonisolated static func plan(_ req: TTSRequest) throws -> Plan {
        let text = req.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw PackageError.unsupportedRequestFeature("text — nothing to speak") }
        // the surface declares neither; the engine refuses them before admission, this is the direct caller's backstop
        if req.emotion != nil { throw PackageError.unsupportedRequestFeature("emotion — this surface declares no emotion control") }
        if req.targetDuration != nil { throw PackageError.unsupportedRequestFeature("targetDuration — this surface declares no duration control; use metaData.speakingRate") }
        if let extra = req.additionalSpeakers, !extra.isEmpty { throw PackageError.unsupportedRequestFeature("additionalSpeakers — a single-voice surface") }
        let m = req.metaData
        func positive(_ key: String) throws -> Double? {
            guard let v = m.doubleValue(key) else { return nil }
            guard v.isFinite, v > 0 else { throw PackageError.unsupportedRequestFeature("metaData.\(key) must be a positive number") }
            return v
        }
        let seed: UInt64
        if let v = m.intValue("seed") { seed = UInt64(bitPattern: Int64(v)) } else { seed = UInt64.random(in: 0 ... UInt64(UInt32.max)) }
        var loudness: Double? = nil
        if let v = m.doubleValue("loudnessLUFS") {
            guard v.isFinite else { throw PackageError.unsupportedRequestFeature("metaData.loudnessLUFS must be a number") }
            loudness = v
        }
        return Plan(text: text, seed: seed,
                    temperature: max(m.doubleValue("temperature") ?? 0.75, 1e-6), topP: min(max(m.doubleValue("topP") ?? 0.69, 0), 1),
                    topK: max(1, m.intValue("topK") ?? 80), maxFrames: min(max(1, m.intValue("maxFrames") ?? defaultMaxFrames), 9000),
                    speakingRate: try positive("speakingRate"), loudnessLUFS: loudness, fmaxHz: try positive("fmaxHz"))
    }

    // MARK: - Voice

    /// The speaker embedding the request asks for; nil = no voice conditioning.
    private func voice(_ req: TTSRequest, tts: Zonos3TTS) throws -> [Float]? {
        switch req.voice.selection {
        case .auto:
            return voices[Self.defaultVoice]
        case .named(let name):
            let id = name.trimmingCharacters(in: .whitespaces).lowercased()
            if id == "none" { return nil }
            guard let e = voices[id] else {
                throw PackageError.unsupportedRequestFeature("voice.named(\"\(name)\") — the voices are: \(voiceIDs.joined(separator: ", ")); 'none' for no voice")
            }
            return e
        case .referenceAudio(let clip):
            let fingerprint = Self.fingerprint(clip.data)
            let key = req.metaData.stringValue("voiceCacheKey").flatMap { $0.isEmpty ? nil : $0 }
            if let key, let cached = voiceCache.value(for: key), cached.fingerprint == fingerprint { return cached.embedding }
            RunProgress.report(.encode)
            let samples = try Zonos3AudioIO.decodeMono(clip, sampleRate: Self.sampleRate)
            let embedding: [Float]
            do { embedding = try tts.embedSpeaker(pcm44100: samples) } catch let error as Zonos3Error {
                throw PackageError.unsupportedRequestFeature("voice.referenceAudio — \(error.description)")
            }
            embeddingBuilds += 1
            if let key { voiceCache.insert(CachedVoice(fingerprint: fingerprint, embedding: embedding), for: key) }
            return embedding
        }
    }

    /// Fingerprint of the reference bytes, so a key reused with other bytes re-embeds. Never a cache key on its own.
    nonisolated static func fingerprint(_ data: Data) -> Int {
        var hasher = Hasher()
        hasher.combine(data)
        return hasher.finalize()
    }
}

extension Zonos3Package {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(Zonos3Package.self) }
}

/// Wrapper-level errors (weight resolution, tier, a Core failure). Runtime request errors use `PackageError`.
public enum Zonos3PackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    case unsupportedTier(String)
    case loadFailed(String)
    case synthesisFailed(String)
    public var description: String {
        switch self {
        case .missingWeights(let s): return "ZONOS3-Mini: \(s)"
        case .unsupportedTier(let s): return "ZONOS3-Mini: unsupported tier \(s)"
        case .loadFailed(let s): return "ZONOS3-Mini: load failed — \(s)"
        case .synthesisFailed(let s): return "ZONOS3-Mini: \(s)"
        }
    }
}

extension MetaData {
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }
    /// A double-valued key, accepting ints (JSON 1 vs 1.0).
    func doubleValue(_ key: String) -> Double? {
        switch self[key] {
        case .double(let value)?: return value
        case .int(let value)?: return Double(value)
        default: return nil
        }
    }
    func stringValue(_ key: String) -> String? {
        if case .string(let value)? = self[key] { return value }
        return nil
    }
    /// A string or an array of strings.
    func stringList(_ key: String) -> [String]? {
        switch self[key] {
        case .string(let value)?: return [value]
        case .array(let values)?:
            return values.compactMap { if case .string(let s) = $0 { return s } else { return nil } }
        default: return nil
        }
    }
}
