// Tokenizer.swift — the checkpoint's SuperBPE tokenizer (tokenizer.json: Split pre-tokenizer + byte-level BPE, no
// normalizer) and the frozen id convention of zonos3/tts/superbpe.py: [2] + (ids + 4) + [3].
// Written against the tokenizers file format rather than through swift-transformers: the Split pre-tokenizer's regex
// (a lookahead, Unicode classes) and `use_regex: false` byte-level stage are pinned here and gated id-for-id.
import CryptoKit
import Foundation

public final class SuperBPETokenizer {
    let vocab: [String: Int]
    let ranks: [String: Int]  // "a b" -> merge rank
    let split: NSRegularExpression
    let specials: [(String, Int)]
    static let byteChars: [Character] = {
        // GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100 onwards
        var keep = Array(33...126) + Array(161...172) + Array(174...255)
        var map = [Int: Int]()
        for b in keep { map[b] = b }
        var n = 0
        for b in 0..<256 where map[b] == nil { map[b] = 256 + n; n += 1 }
        return (0..<256).map { Character(UnicodeScalar(UInt32(map[$0]!))!) }
    }()
    private var cache: [String: [Int]] = [:]

    public init(url: URL, sha256 expected: String? = nil) throws {
        let data = try Data(contentsOf: url)
        if let expected {
            let got = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard got == expected else { throw Zonos3Error.invalid("SuperBPE SHA mismatch") }
        }
        guard let doc = try JSONSerialization.jsonObject(with: data) as? [String: Any], let model = doc["model"] as? [String: Any],
              model["type"] as? String == "BPE", let v = model["vocab"] as? [String: Any], let merges = model["merges"] as? [Any],
              doc["normalizer"] is NSNull || doc["normalizer"] == nil,
              let pre = doc["pre_tokenizer"] as? [String: Any], pre["type"] as? String == "Sequence", let stages = pre["pretokenizers"] as? [[String: Any]],
              stages.count == 2, stages[0]["type"] as? String == "Split", stages[0]["behavior"] as? String == "Isolated", stages[0]["invert"] as? Bool == false,
              let pattern = (stages[0]["pattern"] as? [String: Any])?["Regex"] as? String,
              stages[1]["type"] as? String == "ByteLevel", stages[1]["add_prefix_space"] as? Bool == false, stages[1]["use_regex"] as? Bool == false
        else { throw Zonos3Error.invalid("tokenizer.json is not the expected SuperBPE layout") }
        vocab = v.compactMapValues { ($0 as? NSNumber)?.intValue }
        var r = [String: Int]()
        for (i, m) in merges.enumerated() {
            if let pair = m as? [String], pair.count == 2 { r[pair[0] + " " + pair[1]] = i }
            else if let s = m as? String { r[s] = i }
        }
        ranks = r
        split = try NSRegularExpression(pattern: pattern)
        specials = ((doc["added_tokens"] as? [[String: Any]]) ?? []).compactMap { t in
            guard let c = t["content"] as? String, let id = (t["id"] as? NSNumber)?.intValue else { return nil }
            return (c, id)
        }
    }

    private func bpe(_ piece: String) -> [Int] {
        if let hit = cache[piece] { return hit }
        var word = piece.utf8.map { String(Self.byteChars[Int($0)]) }
        while word.count > 1 {
            var best = Int.max, at = -1
            for i in 0..<(word.count - 1) {
                if let rank = ranks[word[i] + " " + word[i + 1]], rank < best { best = rank; at = i }
            }
            if at < 0 { break }
            let a = word[at], b = word[at + 1]
            var merged: [String] = []
            merged.reserveCapacity(word.count)
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == a && word[i + 1] == b { merged.append(a + b); i += 2 } else { merged.append(word[i]); i += 1 }
            }
            word = merged
        }
        let ids = word.compactMap { vocab[$0] }
        cache[piece] = ids
        return ids
    }

    private func encodePlain(_ text: String) -> [Int] {
        // Split (Isolated): matches and the stretches between them are separate pieces
        let ns = text as NSString
        var ids: [Int] = []
        var pos = 0
        for m in split.matches(in: text, range: NSRange(location: 0, length: ns.length)) where m.range.length > 0 {
            if m.range.location > pos { ids += bpe(ns.substring(with: NSRange(location: pos, length: m.range.location - pos))) }
            ids += bpe(ns.substring(with: m.range))
            pos = m.range.location + m.range.length
        }
        if pos < ns.length { ids += bpe(ns.substring(from: pos)) }
        return ids
    }

    /// Token ids of `text` without special tokens (`encode(text, add_special_tokens=False)`).
    public func encode(_ text: String) -> [Int] {
        var pieces: [(String, Int?)] = [(text, nil)]
        for (content, id) in specials {
            pieces = pieces.flatMap { piece -> [(String, Int?)] in
                guard piece.1 == nil, piece.0.contains(content) else { return [piece] }
                var out: [(String, Int?)] = []
                let parts = piece.0.components(separatedBy: content)
                for (i, p) in parts.enumerated() {
                    if !p.isEmpty { out.append((p, nil)) }
                    if i < parts.count - 1 { out.append((content, id)) }
                }
                return out
            }
        }
        return pieces.flatMap { $0.1.map { [$0] } ?? encodePlain($0.0) }
    }

    /// The text column of a prompt: [2] + (ids + 4) + [3].
    public func textIDs(_ text: String) -> [Int] { [2] + encode(text).map { $0 + 4 } + [3] }
}
