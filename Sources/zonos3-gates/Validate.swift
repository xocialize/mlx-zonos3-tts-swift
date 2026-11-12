// Validate.swift — the live lane: `MLXZonos3TTS` through `MLXServeEngine` (register → prepare → run → stream → refusals →
// cancel → evict), with `phys_footprint` read at each point. This is where the manifest's footprint constants come from.
//
//   zonos3-gates validate <checkpoint dir> <reference clip> <out dir>
//   zonos3-gates validate-store <models root> <reference clip> <out dir>     (downloads the published checkpoint)
import Darwin
import Foundation
import MLX
import MLXServeCore
import MLXToolKit
import MLXZonos3TTS
import Zonos3Core

func physGB() -> (now: Double, peak: Double) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return (Double(info.phys_footprint) / 1e9, Double(info.ledger_phys_footprint_peak) / 1e9)
}

func validate(checkpoint: URL?, store: URL?, reference: URL, out: URL) async throws -> Bool {
    var ok = true
    func check(_ name: String, _ pass: Bool, _ detail: String = "") {
        ok = ok && pass
        print("\(pass ? "PASS" : "FAIL") \(name)\(detail.isEmpty ? "" : ": " + detail)")
    }
    func f(_ v: Double) -> String { String(format: "%.2f", v) }
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let engine = MLXServeEngine()   // the default policy (.permissiveOnly, .advisory)
    if let store { await engine.useModelStore(ModelStore(root: store)) }
    let pid = try await engine.register(Zonos3Package.registration, configuration: Zonos3Configuration(modelDirectory: checkpoint), id: PackageID("zonos3-mini"))
    let advisories = await engine.licenseAdvisories
    check("registered under the default licence policy with no advisory", advisories.isEmpty, advisories.map(\.summary).joined(separator: "; "))
    if await engine.needsDownload(.tts, package: pid) {
        let t = Date()
        _ = try await engine.materialize(package: pid)
        print("materialized in \(f(Date().timeIntervalSince(t))) s")
        check("nothing left to download", !(await engine.needsDownload(.tts, package: pid)))
    }
    let before = physGB().now
    var t0 = Date()
    try await engine.prepare(.tts, package: pid)
    let resident = physGB().now
    let charged = Double(await engine.memory.residentBytes) / 1e9
    print("prepare \(f(Date().timeIntervalSince(t0))) s; phys before \(f(before)) GB, after prepare \(f(resident)) GB; engine charges \(f(charged)) GB")

    func samples(_ a: Audio) throws -> [Float] {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("v-\(UUID().uuidString).wav")
        try a.data.write(to: tmp); defer { try? FileManager.default.removeItem(at: tmp) }
        return try WAV.read(tmp).samples
    }
    func rms(_ x: [Float]) -> Double { x.isEmpty ? 0 : (x.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(x.count)).squareRoot() }
    let line = "We don't have much time. The bridge goes down at midnight, and every guard in the city knows our faces."
    let long = Array(repeating: "Take the second door on the left, keep your head down, and whatever you hear, don't stop walking. The letter arrived on a Tuesday, and by Friday everything we thought we knew about him was gone.", count: 6).joined(separator: " ")
    let clip = Audio(format: .wav, data: try Data(contentsOf: reference))

    @discardableResult
    func run(_ label: String, _ request: TTSRequest, minSeconds: Double = 2) async throws -> Data? {
        t0 = Date()
        guard let response = try await engine.run(request, package: pid) as? TTSResponse else { check(label, false, "no TTSResponse"); return nil }
        let seconds = Date().timeIntervalSince(t0)
        let s = try samples(response.audio)
        let dur = Double(s.count) / 44_100
        let p = physGB()
        check(label, response.audio.sampleRate == 44_100 && response.audio.channels == 1 && dur >= minSeconds && rms(s) > 0.01 && s.allSatisfy(\.isFinite),
              "\(f(dur)) s audio in \(f(seconds)) s (\(String(format: "%.1f", dur / seconds))x), rms \(String(format: "%.3f", rms(s))); phys \(f(p.now)) / peak \(f(p.peak)) GB")
        try response.audio.data.write(to: out.appendingPathComponent("\(label).wav"))
        return response.audio.data
    }
    try await run("warm-up", TTSRequest(text: "Warm up.", metaData: ["seed": .int(1)]), minSeconds: 0.3)
    let a = try await run("auto-voice", TTSRequest(text: line, metaData: ["seed": .int(7)]))
    let b = try await run("auto-voice-again", TTSRequest(text: line, metaData: ["seed": .int(7)]))
    check("same seed, same bytes", a != nil && a == b)
    let c = try await run("named-jett", TTSRequest(text: line, voice: VoiceSelector(.named("Jett")), metaData: ["seed": .int(7)]))
    check("a different voice is different audio", c != nil && c != a)
    try await run("no-voice", TTSRequest(text: line, voice: VoiceSelector(.named("none")), metaData: ["seed": .int(7)]))
    try await run("clone", TTSRequest(text: line, voice: VoiceSelector(.referenceAudio(clip)), metaData: ["seed": .int(7), "voiceCacheKey": .string("roxy")]))
    try await run("clone-cached", TTSRequest(text: line, voice: VoiceSelector(.referenceAudio(clip)), metaData: ["seed": .int(8), "voiceCacheKey": .string("roxy")]))
    try await run("fast", TTSRequest(text: line, metaData: ["seed": .int(7), "speakingRate": .double(24)]))
    try await run("slow", TTSRequest(text: line, metaData: ["seed": .int(7), "speakingRate": .double(9)]))
    try await run("french", TTSRequest(text: "Nous n'avons pas beaucoup de temps. Le pont sera fermé à minuit.", voice: VoiceSelector(.named("gaspard")), metaData: ["seed": .int(7)]))
    let afterShort = physGB().peak

    // streaming: chunk ordinals, exactly one final chunk, and the chunks are the response
    t0 = Date()
    let handle = engine.stream(TTSRequest(text: line, metaData: ["seed": .int(7)]), package: pid)
    var chunks: [TTSStreamChunk] = []
    var firstAt: Double? = nil
    for try await chunk in handle.chunks {
        if firstAt == nil { firstAt = Date().timeIntervalSince(t0) }
        chunks.append(chunk)
    }
    let streamed = chunks.flatMap(\.samples)
    let whole = a.map { try? samples(Audio(format: .wav, data: $0)) } ?? nil
    var streamDiff: Float = .infinity
    if let whole, whole.count == streamed.count { streamDiff = zip(whole, streamed).map { abs($0 - $1) }.max() ?? 0 }
    check("stream", chunks.map(\.index) == Array(0..<chunks.count) && chunks.filter(\.isFinal).count == 1 && chunks.last?.isFinal == true && streamDiff < 1e-4,
          "\(chunks.count) chunks, first after \(String(format: "%.3f", firstAt ?? -1)) s, \(f(Double(streamed.count) / 44_100)) s audio; vs the same-seed run() max_abs \(streamDiff) (16-bit rounding)")

    try await run("long", TTSRequest(text: long, metaData: ["seed": .int(7)]), minSeconds: 40)
    let afterLong = physGB().peak
    print("phys peak: short requests \(f(afterShort)) GB (activation \(f(afterShort - resident))), with the long request \(f(afterLong)) GB (activation \(f(afterLong - resident)))")

    // refusals — each must be refused, legibly
    let refusals: [(String, TTSRequest)] = [
        ("blank text", TTSRequest(text: "   ")),
        ("an unknown voice", TTSRequest(text: line, voice: VoiceSelector(.named("nobody")))),
        ("an emotion", TTSRequest(text: line, emotion: .categorical("happy"))),
        ("a target duration", TTSRequest(text: line, targetDuration: 4)),
        ("a second speaker", TTSRequest(text: line, additionalSpeakers: [TTSSpeakerVoice()])),
        ("a negative speaking rate", TTSRequest(text: line, metaData: ["speakingRate": .double(-3)])),
        ("a reference that is not audio", TTSRequest(text: line, voice: VoiceSelector(.referenceAudio(Audio(format: .wav, data: Data([1, 2, 3, 4])))))),
    ]
    for (label, bad) in refusals {
        do { _ = try await engine.run(bad, package: pid); check("refused: \(label)", false, "ADMITTED") }
        catch { check("refused: \(label)", true, String("\(error)".prefix(150))) }
    }

    // cancellation: cancel the long request mid-run
    let task = Task { try await engine.run(TTSRequest(text: long, metaData: ["seed": .int(9)]), package: pid) }
    try await Task.sleep(nanoseconds: 1_000_000_000)
    t0 = Date()
    task.cancel()
    do { _ = try await task.value; check("cancel mid-run", false, "ran to completion") }
    catch { check("cancel mid-run", true, "stopped \(String(format: "%.3f", Date().timeIntervalSince(t0))) s after cancel (\(String("\(error)".prefix(80))))") }
    try await run("after-cancel", TTSRequest(text: line, metaData: ["seed": .int(7)]))

    // eviction of a voice key, with an empty text, then of the package
    let evicted = try await engine.run(TTSRequest(text: "", metaData: ["evictVoiceCacheKeys": .string("roxy")]), package: pid) as? TTSResponse
    let evictedSamples = try evicted.map { try samples($0.audio) }
    check("evict-only call returns empty audio", evictedSamples?.isEmpty == true)
    await engine.evict(package: pid)
    MLX.Memory.clearCache()
    let p = physGB()
    print("evicted: phys \(f(p.now)) GB (lifetime peak \(f(p.peak)) GB); MLX active \(f(Double(MLX.Memory.activeMemory) / 1e9)) GB, cache \(f(Double(MLX.Memory.cacheMemory) / 1e9)) GB")
    // a second prepare must land where the first did (nothing from the first load is still held)
    try await engine.prepare(.tts, package: pid)
    let again = physGB().now
    check("re-prepare after evict lands at the same footprint", again < resident + 0.35, "phys \(f(again)) GB vs \(f(resident)) GB the first time")
    await engine.evict(package: pid)
    MLX.Memory.clearCache()
    print("evicted again: phys \(f(physGB().now)) GB; MLX active \(f(Double(MLX.Memory.activeMemory) / 1e9)) GB")
    print("FOOTPRINT resident \(f(resident)) GB, peak activation \(f(afterLong - resident)) GB (long request), \(f(afterShort - resident)) GB (short requests)")
    return ok
}
