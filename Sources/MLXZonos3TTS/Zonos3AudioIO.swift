// Zonos3AudioIO.swift — canonical `Audio` in and out. A reference clip is decoded with AVFoundation, mixed to mono and
// converted to the speaker encoder's 44.1 kHz input rate; speech goes out as 16-bit PCM WAV bytes (the fleet's `encodeWAV16`).
// Lifted from mlx-sam-audio-swift (SAMAudioAudioIO).

import AVFoundation
import Foundation
import MLXToolKit

enum Zonos3AudioIO {
    /// Mono float samples at `sampleRate`, resampled by AVAudioConverter when the file's rate differs.
    static func decodeMono(_ audio: Audio, sampleRate: Int) throws -> [Float] {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("zonos3-reference-\(UUID().uuidString).wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: tmp) } catch {
            throw PackageError.unsupportedRequestFeature("the reference clip is not readable audio (\(error.localizedDescription))")
        }
        let src = file.processingFormat
        let total = Int(file.length)
        guard total > 0, let buf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: 65_536) else {
            throw PackageError.unsupportedRequestFeature("the reference clip is empty")
        }
        // `AVAudioFile.read(into:)` may return fewer frames than asked (a float WAV with a PEAK chunk came back 788 frames
        // short of its 480 000 in one call), so read until the file's length is reached.
        var mono = [Float]()
        mono.reserveCapacity(total)
        while mono.count < total {
            try file.read(into: buf, frameCount: AVAudioFrameCount(min(65_536, total - mono.count)))
            guard buf.frameLength > 0 else { break }
            mono.append(contentsOf: mixToMono(buf))
        }
        guard mono.count == total else {
            throw PackageError.unsupportedRequestFeature("the reference clip could not be read in full (\(mono.count) of \(total) frames)")
        }
        guard Int(src.sampleRate) != sampleRate else { return mono }
        return try resample(mono, from: src.sampleRate, to: Double(sampleRate))
    }

    static func mixToMono(_ buf: AVAudioPCMBuffer) -> [Float] {
        let ch = Int(buf.format.channelCount), n = Int(buf.frameLength)
        guard let data = buf.floatChannelData, ch > 0 else { return [] }
        var mono = [Float](repeating: 0, count: n)
        for c in 0 ..< ch { for i in 0 ..< n { mono[i] += data[c][i] } }
        if ch > 1 { let inv = 1 / Float(ch); for i in 0 ..< n { mono[i] *= inv } }
        return mono
    }

    static func resample(_ samples: [Float], from: Double, to: Double) throws -> [Float] {
        guard let inFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: from, channels: 1, interleaved: false),
              let outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: to, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFmt, to: outFmt),
              let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw PackageError.unsupportedRequestFeature("the reference clip's sample rate (\(from) Hz) cannot be converted to \(to) Hz")
        }
        inBuf.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { inBuf.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        let capacity = AVAudioFrameCount((Double(samples.count) * to / from).rounded(.up)) + 64
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: capacity) else {
            throw PackageError.unsupportedRequestFeature("the reference clip is too long to convert")
        }
        var fed = false
        var failure: NSError?
        let status = converter.convert(to: outBuf, error: &failure) { _, state in
            if fed { state.pointee = .endOfStream; return nil }
            fed = true
            state.pointee = .haveData
            return inBuf
        }
        guard status != .error, failure == nil else {
            throw PackageError.unsupportedRequestFeature("sample-rate conversion failed (\(failure?.localizedDescription ?? "unknown"))")
        }
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
    }

    /// Mono float [-1, 1] → 16-bit PCM WAV (× 32767, clamped; non-finite samples become silence).
    static func encodeWAV16(samples: [Float], sampleRate: Int) -> Data {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let bytes = samples.count * 2
        d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + bytes)); d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(UInt32(bytes))
        var pcm = [Int16](repeating: 0, count: samples.count)
        for (i, s) in samples.enumerated() { pcm[i] = Int16(max(-1, min(1, s.isFinite ? s : 0)) * 32767) }
        pcm.withUnsafeBufferPointer { d.append(Data(buffer: $0)) }
        return d
    }
}
