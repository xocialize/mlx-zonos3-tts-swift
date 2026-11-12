// Weights.swift — the converted checkpoint (Tools/convert_gguf.py): dense tensors and MLX affine-quantized matrices
// (group 32; `<base>.weight` packed uint32 words + `<base>.scales` + `<base>.biases`). A port of zonos3_mlx/gguf_io.py's
// `Weight` / `matmul` / `dense`.
import Foundation
import MLX

public struct Weight {
    public let w: MLXArray
    public let scales: MLXArray?
    public let biases: MLXArray?
    public let bits: Int?
    public var quantized: Bool { scales != nil }
    public static let groupSize = 32
}

public enum Zonos3Error: Error, CustomStringConvertible {
    case missing(String), invalid(String)
    public var description: String {
        switch self {
        case .missing(let s): return "missing: \(s)"
        case .invalid(let s): return s
        }
    }
}

/// `x @ W.T` (W stored [out, in], as in torch Linear).
public func zmatmul(_ x: MLXArray, _ W: Weight) -> MLXArray {
    if let s = W.scales {
        return quantizedMM(x, W.w, scales: s, biases: W.biases, transpose: true, groupSize: Weight.groupSize, bits: W.bits)
    }
    return matmul(x, W.w.T.asType(.float32))
}

/// The float32 matrix of a weight (dequantized when quantized).
public func zdense(_ W: Weight) -> MLXArray {
    if let s = W.scales {
        return dequantized(W.w, scales: s, biases: W.biases, groupSize: Weight.groupSize, bits: W.bits)
    }
    return W.w.asType(.float32)
}

/// One converted file: matrices as `Weight`, tables as float32, split by the caller's `isTable` rule (upstream's loaders
/// differ per network).
public final class WeightFile {
    public private(set) var w: [String: Weight] = [:]
    public private(set) var t: [String: MLXArray] = [:]
    public let names: Set<String>

    public init(url: URL, quantized: [String: Int], isTable: (String, [Int]) -> Bool) throws {
        // Lazy loads bind to the stream current at creation: weights ride the CPU stream (Metal-watchdog rule 1).
        let raw = try Device.withDefaultDevice(.cpu) { try loadArrays(url: url) }
        var names = Set<String>()
        for (name, a) in raw {
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            names.insert(name)
            if let bits = quantized[name] {
                let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
                guard let s = raw[base + ".scales"], let b = raw[base + ".biases"] else { throw Zonos3Error.missing("\(base).scales/.biases") }
                w[name] = Weight(w: a, scales: s, biases: b, bits: bits)
            } else if isTable(name, a.shape) {
                t[name] = a.asType(.float32)
            } else {
                w[name] = Weight(w: a, scales: nil, biases: nil, bits: nil)
            }
        }
        self.names = names
        eval(Array(t.values) + w.values.flatMap { [$0.w] + [$0.scales, $0.biases].compactMap { $0 } })
    }

    public func table(_ name: String) throws -> MLXArray {
        guard let a = t[name] else { throw Zonos3Error.missing(name) }
        return a
    }
    public func weight(_ name: String) throws -> Weight {
        guard let a = w[name] else { throw Zonos3Error.missing(name) }
        return a
    }
}

/// config.json of a converted checkpoint (one entry per file: the GGUF metadata and the quantized tensors' bit widths).
public struct CheckpointConfig {
    public let root: URL
    private let files: [String: [String: Any]]

    public init(directory: URL) throws {
        root = directory
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let doc = try JSONSerialization.jsonObject(with: data) as? [String: Any], let files = doc["files"] as? [String: [String: Any]] else {
            throw Zonos3Error.invalid("config.json: no files table")
        }
        self.files = files
    }
    public func has(_ file: String) -> Bool { files[file] != nil }
    public func meta(_ file: String) throws -> [String: Any] {
        guard let m = files[file]?["meta"] as? [String: Any] else { throw Zonos3Error.missing("config.json files[\(file)].meta") }
        return m
    }
    public func quantized(_ file: String) -> [String: Int] {
        ((files[file]?["quantized"] as? [String: Any]) ?? [:]).compactMapValues { ($0 as? NSNumber)?.intValue }
    }
}

/// Typed reads of a GGUF metadata table (`<arch>.<key>`).
public struct Meta {
    let m: [String: Any]
    let arch: String
    public init(_ m: [String: Any], arch: String) { self.m = m; self.arch = arch }
    func raw(_ key: String) throws -> Any {
        guard let v = m["\(arch).\(key)"] else { throw Zonos3Error.missing("\(arch).\(key)") }
        return v
    }
    public func int(_ key: String) throws -> Int {
        guard let v = try raw(key) as? NSNumber else { throw Zonos3Error.invalid("\(arch).\(key) is not a number") }
        return v.intValue
    }
    public func double(_ key: String) throws -> Double {
        guard let v = try raw(key) as? NSNumber else { throw Zonos3Error.invalid("\(arch).\(key) is not a number") }
        return v.doubleValue
    }
    public func string(_ key: String) throws -> String {
        guard let v = try raw(key) as? String else { throw Zonos3Error.invalid("\(arch).\(key) is not a string") }
        return v
    }
    public func ints(_ key: String) throws -> [Int] {
        guard let v = try raw(key) as? [NSNumber] else { throw Zonos3Error.invalid("\(arch).\(key) is not a number list") }
        return v.map(\.intValue)
    }
    public func doubles(_ key: String) throws -> [Double] {
        guard let v = try raw(key) as? [NSNumber] else { throw Zonos3Error.invalid("\(arch).\(key) is not a number list") }
        return v.map(\.doubleValue)
    }
    public func strings(_ key: String) throws -> [String] {
        guard let v = try raw(key) as? [String] else { throw Zonos3Error.invalid("\(arch).\(key) is not a string list") }
        return v
    }
    public func has(_ key: String) -> Bool { m["\(arch).\(key)"] != nil }
}
