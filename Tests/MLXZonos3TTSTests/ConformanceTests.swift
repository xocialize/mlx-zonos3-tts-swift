// ConformanceTests.swift — the offline conformance gate for MLXZonos3TTS: manifest (C0 / C1 / C7 / C8 / C-memory / C11 /
// provenance), the MAT gate, the CAN gate (pre-cancelled run propagation, checkpoint cadence), the unloaded and
// unoffered-tier refusals, the request plane (text, sampling, conditions, seed; everything undeclared refused), the voice
// cache type and the audio helpers. Nothing here evals a kernel: the live run is `zonos3-gates validate` (GPU, through
// MLXServeEngine).

import Foundation
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXZonos3TTS
@testable import Zonos3Core

private func wav(seconds: Double = 1, sampleRate: Int = 44_100) -> Audio {
    let n = Int(seconds * Double(sampleRate))
    let samples = (0..<n).map { Float(sin(Double($0) * 2 * .pi * 440 / Double(sampleRate))) * 0.25 }
    return Audio(format: .wav, data: Zonos3AudioIO.encodeWAV16(samples: samples, sampleRate: sampleRate), sampleRate: sampleRate, channels: 1)
}

final class ManifestConformanceTests: XCTestCase {

    /// C7 / C8 — Apache-2.0 weights, MIT port code; admitted by the default policy with no advisory.
    func testLicenseIsPermissiveOnBothLayers() {
        let license = Zonos3Package.manifest.license
        XCTAssertEqual(license.weightLicense, .apache2)
        XCTAssertEqual(license.portCodeLicense, .mit)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(license).isAdmitted)
        XCTAssertNil(license.portCodeLicenseLackingNotice)
        XCTAssertTrue(license.missingRequiredDisplayNotices.isEmpty)
        let text = (license.notices ?? []).map(\.text).joined(separator: " ")
        for phrase in ["spoken as written", "consent", "approximate"] { XCTAssertTrue(text.contains(phrase), phrase) }
    }

    func testEngineGateAdmitsThePackage() async throws {
        for engine in [MLXServeEngine(), MLXServeEngine(licenseEnforcement: .blocking)] {
            _ = try await engine.register(Zonos3Package.registration, configuration: Zonos3Configuration(), id: PackageID("zonos3-mini"))
            let advisories = await engine.licenseAdvisories
            XCTAssertTrue(advisories.isEmpty, advisories.map(\.summary).joined(separator: "; "))
        }
    }

    /// C-memory — one tier, split footprint, at least what `zonos3-gates validate` measured (1.60 GB resident; 2.79 GB
    /// peak — the largest of three runs).
    func testFootprintIsTheMeasuredOne() {
        let footprints = Zonos3Package.manifest.requirements.footprints
        XCTAssertEqual(footprints.map(\.quant), [.int4])
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes, 1_600_000_000)
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes + footprints[0].peakActivationBytes, 2_790_000_000)
        XCTAssertEqual(Zonos3Package.offeredQuants, [.int4])
        XCTAssertEqual(Zonos3Configuration().quant, .int4)
    }

    /// C1 / C11 — `tts` only, streaming by audio chunk, no emotion / duration / multi-speaker declaration.
    func testCapabilityIsStreamingTTS() {
        XCTAssertEqual(Set(Zonos3Package.manifest.capabilities), [.tts])
        let surface = Zonos3Package.manifest.surfaces.first { $0.capability == .tts }
        XCTAssertEqual(surface?.name, "zonos3-mini")
        XCTAssertEqual(surface?.streaming, .audioChunk)
        XCTAssertTrue(surface?.ttsControls?.emotionModes.isEmpty ?? true)
        XCTAssertFalse(surface?.ttsControls?.supportsTargetDuration ?? false)
        XCTAssertEqual(surface?.ttsControls?.maxSpeakers ?? 1, 1)
        let names = Set(surface?.parameters.map(\.name) ?? [])
        XCTAssertTrue(names.isSuperset(of: ["text", "voice"]), "\(names)")
        XCTAssertFalse(names.contains("emotion"))
        XCTAssertFalse(names.contains("targetDuration"))
        for phrase in ["44.1 kHz", "spoken as written", "approximate", "savannah"] { XCTAssertTrue(surface?.summary.contains(phrase) ?? false, phrase) }
        let specialties = Dictionary(uniqueKeysWithValues: Zonos3Package.manifest.specialties.map { ($0.specialty, $0.strength) })
        XCTAssertEqual(specialties[.realtimeStreaming], 1.0)
        XCTAssertLessThan(specialties[.voiceClone] ?? 1, 1.0, "cloning is declared weaker than the Character engines")
    }

    func testProvenancePinsTheUpstreamRevision() {
        let p = Zonos3Package.manifest.provenance
        XCTAssertEqual(p.sourceRepo, "Zyphra/ZONOS3-Mini")
        XCTAssertEqual(p.revision, "7146a686b62bcb0661528fa300984e815e10152b")
    }
}

final class MaterializationConformanceTests: XCTestCase {
    private func emptyStoreRoot() -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("zonos3-empty-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    /// MAT-1..5: a fresh (dir-less) configuration reports the source missing; an explicit directory satisfies.
    func testMaterializationGate() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("zonos3-ckpt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data().write(to: dir.appendingPathComponent(Zonos3Configuration.probe))
        let report = MaterializationConformance.check(freshConfiguration: Zonos3Configuration(modelsRootDirectory: emptyStoreRoot()),
                                                      satisfiedConfiguration: Zonos3Configuration(modelDirectory: dir))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// The source is the published checkpoint at its pin, in a namespace we control.
    func testWeightSourceIsThePublishedCheckpointAtItsPin() {
        let sources = Zonos3Configuration().weightSources
        XCTAssertEqual(sources.map(\.role), ["checkpoint"])
        XCTAssertEqual(sources.first?.repo, "mlx-community/ZONOS3-Mini-swift-ptq")
        XCTAssertEqual(sources.first?.revision, "a95067477f04c1feff6c3a16a431a7e409bee72a")
        XCTAssertNotEqual(Zonos3Configuration.repo, "mlx-community/ZONOS3-Mini", "that repository is mlx-audio's conversion, a different layout")
        XCTAssertTrue(Zonos3Configuration.isPublished)
        XCTAssertEqual(Zonos3Configuration(revision: "abc").weightSources.first?.revision, "abc")
    }

    /// Every file the pipeline opens with the small decoder is downloaded; the 2.2 GB full decoder is not.
    func testDeclaredFilesMatchTheCheckpointLayout() {
        let layout = ["config.json", "tokenizer.json", "voices.json", "lm/model.safetensors", "codec/model.safetensors", "speaker_encoder/model.safetensors"]
        for file in layout {
            XCTAssertTrue(Zonos3Configuration.files.contains { WeightSourceProbe.matches(path: file, glob: $0) }, "not downloaded: \(file)")
        }
        XCTAssertFalse(Zonos3Configuration.files.contains { WeightSourceProbe.matches(path: "codec-full/model.safetensors", glob: $0) })
    }

    func testExplicitDirectoryProbe() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("zonos3-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let c = Zonos3Configuration(modelDirectory: dir)
        XCTAssertEqual(c.missingWeightSources(storeRoot: nil).count, 1)
        try Data().write(to: dir.appendingPathComponent(Zonos3Configuration.probe))
        XCTAssertTrue(c.missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(c.resolvedDirectory(storeRoot: nil), dir)
        XCTAssertEqual(c.prewarmPaths.map(\.lastPathComponent), ["model.safetensors", "model.safetensors"])
    }

    func testCodableExcludesEnvironmentURLs() throws {
        let c = Zonos3Configuration(revision: "r", modelDirectory: URL(fileURLWithPath: "/tmp/x"), modelsRootDirectory: URL(fileURLWithPath: "/tmp/y"))
        let json = String(decoding: try JSONEncoder().encode(c), as: UTF8.self)
        XCTAssertFalse(json.contains("/tmp/"))
        let back = try JSONDecoder().decode(Zonos3Configuration.self, from: Data(json.utf8))
        XCTAssertEqual(back.revision, "r")
        XCTAssertNil(back.modelDirectory)
        XCTAssertEqual(back.quant, .int4)
    }
}

final class CancellationConformanceTests: XCTestCase {
    /// CAN-1 / CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` unchanged, even unloaded.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = Zonos3Package(configuration: Zonos3Configuration())
        let report = await CancellationConformance.checkRun(package: package, request: TTSRequest(text: "Hello."))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — a checkpoint once per generated frame, on the RunProgress seam.
    func testCheckpointCadence() {
        let report = CancellationConformance.checkCadence(
            manifest: Zonos3Package.manifest,
            posture: .cadence([.init(phase: .generate, unit: .frame, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

final class RequestPlaneTests: XCTestCase {
    func testUnloadedPackageRejectsLegibly() async {
        let package = Zonos3Package(configuration: Zonos3Configuration())
        do {
            _ = try await package.run(TTSRequest(text: "Hello."))
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded, got \(error)") }
        } catch { XCTFail("expected PackageError, got \(error)") }
    }

    /// Eviction runs without weights, and an evict-only call (empty text) succeeds unloaded.
    func testEvictOnlyCallWorksUnloaded() async throws {
        let package = Zonos3Package(configuration: Zonos3Configuration())
        let response = try await package.run(TTSRequest(text: "", metaData: ["evictVoiceCacheKeys": .array([.string("a"), .string("b")])])) as? TTSResponse
        XCTAssertEqual(response?.audio.sampleRate, 44_100)
        XCTAssertEqual(response?.audio.data.count, 44, "a header-only WAV")
    }

    func testUnofferedTierIsRefusedByLoad() async {
        for q in [Quant.int8, .fp32, .bf16, .fp16] {
            let package = Zonos3Package(configuration: Zonos3Configuration(quant: q))
            do {
                try await package.load()
                XCTFail("loaded \(q)")
            } catch let error as Zonos3PackageError {
                guard case .unsupportedTier = error else { return XCTFail("\(error)") }
            } catch { XCTFail("\(error)") }
        }
    }

    func testMissingWeightsAreReportedByLoad() async {
        let package = Zonos3Package(configuration: Zonos3Configuration())
        do { try await package.load(); XCTFail("loaded with nothing on disk") }
        catch let error as Zonos3PackageError { guard case .missingWeights = error else { return XCTFail("\(error)") } }
        catch { XCTFail("\(error)") }
    }

    func testPlanDefaultsAreTheReferenceSampler() throws {
        let plan = try Zonos3Package.plan(TTSRequest(text: "  Hello there.  ", metaData: ["seed": .int(7)]))
        XCTAssertEqual(plan.text, "Hello there.")
        XCTAssertEqual(plan.seed, 7)
        XCTAssertEqual(plan.temperature, 0.75); XCTAssertEqual(plan.topK, 80); XCTAssertEqual(plan.topP, 0.69)
        XCTAssertEqual(plan.maxFrames, 2000)
        XCTAssertNil(plan.speakingRate); XCTAssertNil(plan.loudnessLUFS); XCTAssertNil(plan.fmaxHz)
        let o = plan.options
        XCTAssertEqual(o.sampling.repetitionPenalty, 2.4); XCTAssertEqual(o.sampling.repetitionWindow, 96); XCTAssertEqual(o.sampling.repetitionCodebooks, 6)
        XCTAssertEqual(o.silencePrefillFrames, 3)
    }

    func testPlanReadsTheMetaDataPlane() throws {
        let plan = try Zonos3Package.plan(TTSRequest(text: "Hi.", metaData: [
            "temperature": .double(0.5), "topK": .int(20), "topP": .double(0.9), "maxFrames": .int(300),
            "speakingRate": .int(18), "loudnessLUFS": .double(-20), "fmaxHz": .double(16000)]))
        XCTAssertEqual(plan.temperature, 0.5); XCTAssertEqual(plan.topK, 20); XCTAssertEqual(plan.topP, 0.9); XCTAssertEqual(plan.maxFrames, 300)
        XCTAssertEqual(plan.speakingRate, 18); XCTAssertEqual(plan.loudnessLUFS, -20); XCTAssertEqual(plan.fmaxHz, 16000)
        XCTAssertEqual(plan.options.conditions.speakingRate, 18)
    }

    func testAnUnseededPlanDrawsASeed() throws {
        let seeds = try (0..<4).map { _ in try Zonos3Package.plan(TTSRequest(text: "Hi.")).seed }
        XCTAssertGreaterThan(Set(seeds).count, 1)
    }

    func testUndeclaredFeaturesAreRefused() {
        let bad: [TTSRequest] = [
            TTSRequest(text: "   "),
            TTSRequest(text: "Hi.", emotion: .categorical("happy")),
            TTSRequest(text: "Hi.", targetDuration: 3),
            TTSRequest(text: "Hi.", additionalSpeakers: [TTSSpeakerVoice()]),
            TTSRequest(text: "Hi.", metaData: ["speakingRate": .double(0)]),
            TTSRequest(text: "Hi.", metaData: ["fmaxHz": .double(-1)]),
        ]
        for request in bad {
            XCTAssertThrowsError(try Zonos3Package.plan(request)) { error in
                guard case PackageError.unsupportedRequestFeature = error else { return XCTFail("\(error)") }
            }
        }
    }
}

final class SupportTests: XCTestCase {
    func testVoiceCacheIsLeastRecentlyUsed() {
        var cache = VoiceFeatureCache<Int>(capacity: 2)
        cache.insert(1, for: "a"); cache.insert(2, for: "b")
        XCTAssertEqual(cache.value(for: "a"), 1)   // touches a
        cache.insert(3, for: "c")                   // drops b
        XCTAssertEqual(cache.keys, ["a", "c"])
        XCTAssertNil(cache.value(for: "b"))
        cache.remove("a")
        XCTAssertEqual(cache.keys, ["c"])
        cache.removeAll()
        XCTAssertTrue(cache.isEmpty)
    }

    func testFingerprintFollowsTheBytes() {
        XCTAssertEqual(Zonos3Package.fingerprint(Data([1, 2, 3])), Zonos3Package.fingerprint(Data([1, 2, 3])))
        XCTAssertNotEqual(Zonos3Package.fingerprint(Data([1, 2, 3])), Zonos3Package.fingerprint(Data([1, 2, 4])))
    }

    func testVoicesFileIsValidated() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("zonos3-voices-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("voices.json")
        try JSONSerialization.data(withJSONObject: ["voices": [["id": "Savannah", "embedding": [Double](repeating: 0.5, count: 192)]]]).write(to: url)
        let voices = try Zonos3Package.readVoices(url, dim: 192)
        XCTAssertEqual(voices["savannah"]?.count, 192)
        try JSONSerialization.data(withJSONObject: ["voices": [["id": "x", "embedding": [1.0, 2.0]]]]).write(to: url)
        XCTAssertThrowsError(try Zonos3Package.readVoices(url, dim: 192))
    }

    func testWAVRoundTripAndResample() throws {
        let clip = wav(seconds: 0.5, sampleRate: 24_000)
        let same = try Zonos3AudioIO.decodeMono(clip, sampleRate: 24_000)
        XCTAssertEqual(same.count, 12_000)
        let up = try Zonos3AudioIO.decodeMono(clip, sampleRate: 44_100)
        XCTAssertEqual(Double(up.count), 22_050, accuracy: 64)
        XCTAssertThrowsError(try Zonos3AudioIO.decodeMono(Audio(format: .wav, data: Data([1, 2, 3])), sampleRate: 44_100))
    }

    func testSamplingClampsAndConditionBuckets() {
        var s = Sampling(); s.topK = 0; s.temperature = 0; s.repetitionCodebooks = -1
        XCTAssertEqual(outputPCM([0.1, -0.1], gain: 1), [0.1, -0.1])
        let loud = outputPCM([1.2, -1.2, 5.0], gain: 1)
        XCTAssertLessThan(loud[0], 1.0); XCTAssertGreaterThan(loud[0], 0.9); XCTAssertGreaterThan(loud[1], -1.0)
        XCTAssertLessThanOrEqual(loud[2], 1.0)   // the soft limiter bounds full scale
    }
}
