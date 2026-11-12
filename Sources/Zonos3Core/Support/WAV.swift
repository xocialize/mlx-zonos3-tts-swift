// WAV.swift — minimal RIFF/WAVE reader and writer for the CLI: 16 / 24-bit PCM and 32-bit float in, mono float32 out.
import Foundation
import MLX

public enum WAV {
    /// Returns mono samples (channels averaged) and the file's sample rate.
    public static func read(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let d = try Data(contentsOf: url)
        guard d.count > 12, String(data: d[0..<4], encoding: .ascii) == "RIFF", String(data: d[8..<12], encoding: .ascii) == "WAVE" else {
            throw Zonos3Error.invalid("\(url.lastPathComponent) is not a RIFF/WAVE file")
        }
        func u16(_ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        var pos = 12, format = 0, channels = 0, rate = 0, bits = 0
        var out: [Float]? = nil
        while pos + 8 <= d.count {
            let id = String(data: d[pos..<(pos + 4)], encoding: .ascii) ?? "", size = u32(pos + 4), body = pos + 8
            if id == "fmt " {
                format = u16(body); channels = u16(body + 2); rate = u32(body + 4); bits = u16(body + 14)
                if format == 0xFFFE, size >= 26 { format = u16(body + 24) }
            } else if id == "data" {
                let end = min(d.count, body + size), bytes = bits / 8
                guard channels > 0, bytes > 0 else { throw Zonos3Error.invalid("WAV data before fmt") }
                let frames = (end - body) / (bytes * channels)
                var mono = [Float](repeating: 0, count: frames)
                d.withUnsafeBytes { raw in
                    for f in 0..<frames {
                        var acc: Float = 0
                        for c in 0..<channels {
                            let o = body + (f * channels + c) * bytes
                            switch (format, bits) {
                            case (3, 32): acc += raw.loadUnaligned(fromByteOffset: o, as: Float32.self)
                            case (1, 16): acc += Float(raw.loadUnaligned(fromByteOffset: o, as: Int16.self)) / 32768
                            case (1, 24):
                                let v = Int32(raw[o]) | Int32(raw[o + 1]) << 8 | Int32(Int8(bitPattern: raw[o + 2])) << 16
                                acc += Float(v) / 8_388_608
                            case (1, 32): acc += Float(raw.loadUnaligned(fromByteOffset: o, as: Int32.self)) / 2_147_483_648
                            default: break
                            }
                        }
                        mono[f] = acc / Float(channels)
                    }
                }
                guard [(3, 32), (1, 16), (1, 24), (1, 32)].contains(where: { $0 == (format, bits) }) else {
                    throw Zonos3Error.invalid("unsupported WAV encoding: format \(format), \(bits) bits")
                }
                out = mono
            }
            pos = body + size + (size & 1)
        }
        guard let samples = out else { throw Zonos3Error.invalid("\(url.lastPathComponent) has no data chunk") }
        return (samples, rate)
    }

    /// Mono 32-bit float.
    public static func write(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        var d = Data()
        func put<T>(_ v: T) { withUnsafeBytes(of: v) { d.append(contentsOf: $0) } }
        let bytes = samples.count * 4
        d.append("RIFF".data(using: .ascii)!); put(UInt32(36 + bytes)); d.append("WAVE".data(using: .ascii)!)
        d.append("fmt ".data(using: .ascii)!); put(UInt32(16)); put(UInt16(3)); put(UInt16(1)); put(UInt32(sampleRate))
        put(UInt32(sampleRate * 4)); put(UInt16(4)); put(UInt16(32))
        d.append("data".data(using: .ascii)!); put(UInt32(bytes))
        samples.withUnsafeBytes { d.append(contentsOf: $0) }
        try d.write(to: url)
    }
}
