// VoiceEmbeddingCache.swift — what a clone keeps of a reference under a caller's `voiceCacheKey`: the 192-d speaker
// embedding, which depends on the clip bytes alone (MES Base §13 rule 6; the mlx-qwen3-tts-swift 0.7.0 API, AB-A-0233).
import Foundation

struct CachedVoice {
    /// Fingerprint of the reference bytes the embedding came from.
    let fingerprint: Int
    let embedding: [Float]
}

/// A small least-recently-used map from a caller's key to a value. Reading a key or inserting it makes it the most
/// recent; inserting past `capacity` drops the least recent. (The same type as mlx-qwen3-tts-swift 0.7.0's.)
struct VoiceFeatureCache<Value> {
    let capacity: Int
    private var entries: [String: Value] = [:]
    /// Keys, least recent first.
    private var order: [String] = []

    init(capacity: Int) {
        precondition(capacity > 0, "capacity must be positive")
        self.capacity = capacity
    }

    var keys: [String] { order }
    var isEmpty: Bool { entries.isEmpty }

    mutating func value(for key: String) -> Value? {
        guard let value = entries[key] else { return nil }
        touch(key)
        return value
    }

    mutating func insert(_ value: Value, for key: String) {
        entries[key] = value
        touch(key)
        while order.count > capacity { entries[order.removeFirst()] = nil }
    }

    mutating func remove(_ key: String) {
        entries[key] = nil
        order.removeAll { $0 == key }
    }

    mutating func removeAll() {
        entries.removeAll()
        order.removeAll()
    }

    private mutating func touch(_ key: String) {
        order.removeAll { $0 == key }
        order.append(key)
    }
}
