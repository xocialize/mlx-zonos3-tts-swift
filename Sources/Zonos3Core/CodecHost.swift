// CodecHost.swift — streaming host half of the iSTFT codec decoder: overlap-add + FIR mixer, then the 44160 -> 44100 Hz
// sinc resampler. Ports of zonos3/codec/stream_host.py (`IstftHeadStream`, `ResamplerStream`; Zyphra, MIT) @ 4c66cea.
// Arithmetic is Double, outputs Float, as upstream. One difference: upstream runs the inverse FFT on the host in float64;
// here the windowed inverse-FFT frames arrive from the device in float32 (CodecDecoder.swift) and the overlap-add starts
// from them.
import Accelerate
import Foundation

final class IstftHeadStream {
    let nFFTs: [Int]
    let hop: Int
    let win2: [[Double]]  // window squared per branch
    let mixer: [[Double]]  // [2][K]
    let K: Int, h: Int
    var ola: [[Double]] = [[], []]
    var env: [[Double]] = [[], []]
    var olaBase = [0, 0]
    var mixIn: [[Double]]
    var mixBase = 0, mixNext = 0
    private(set) var frames = 0
    var done = false

    init(nFFTs: [Int], hop: Int, windows: [[Float]], mixer: [[Float]]) throws {
        self.nFFTs = nFFTs; self.hop = hop
        guard mixer.count == 2, mixer[0].count % 2 == 1 else { throw Zonos3Error.invalid("istft: mixer kernel must be odd") }
        for (n, w) in zip(nFFTs, windows) where w.count != n || n % hop != 0 { throw Zonos3Error.invalid("istft: window / n_fft / hop mismatch") }
        win2 = windows.map { $0.map { Double($0) * Double($0) } }
        self.mixer = mixer.map { $0.map(Double.init) }
        K = mixer[0].count; h = (K - 1) / 2
        mixIn = [[Double](repeating: 0, count: h), [Double](repeating: 0, count: h)]
    }

    /// `frames[i]`: the windowed inverse-FFT frames of branch i, [cols * n_i] row-major. Returns the 44160 Hz samples
    /// now final.
    func push(_ windowed: [[Float]], cols: Int) -> [Float] {
        precondition(!done, "istft: push after flush")
        if cols == 0 { return [] }
        for i in 0..<2 {
            let n = nFFTs[i], r = n / hop
            let span = (cols + r - 1) * hop
            var add = [Double](repeating: 0, count: span)
            var envAdd = [Double](repeating: 0, count: span)
            let fr = windowed[i], w2 = win2[i]
            for c in 0..<cols {
                let base = c * hop, src = c * n
                for t in 0..<n {
                    add[base + t] += Double(fr[src + t])
                    envAdd[base + t] += w2[t]
                }
            }
            let o = frames * hop - olaBase[i]
            let need = o + span
            if ola[i].count < need {
                let grow = need - ola[i].count
                ola[i].append(contentsOf: [Double](repeating: 0, count: grow))
                env[i].append(contentsOf: [Double](repeating: 0, count: grow))
            }
            for j in 0..<span { ola[i][o + j] += add[j]; env[i][o + j] += envAdd[j] }
        }
        frames += cols
        for i in 0..<2 { finalize(i, frames * hop) }
        return mixEmit()
    }

    /// End of stream: every sample m < frames*hop is final; the mixer sees h zeros after the last sample.
    func flush() -> [Float] {
        if done || frames == 0 { return [] }
        done = true
        let L = frames * hop
        for i in 0..<2 {
            finalize(i, olaBase[i] + ola[i].count)
            let keep = h + L - mixBase
            var m = mixIn[i]
            if m.count >= keep { m = Array(m[0..<keep]) } else { m.append(contentsOf: [Double](repeating: 0, count: keep - m.count)) }
            m.append(contentsOf: [Double](repeating: 0, count: h))
            mixIn[i] = m
        }
        return mixEmit()
    }

    /// Moves overlap-add positions [ola_base, pf) of branch i into the mixer input, dropping the centre padding.
    private func finalize(_ i: Int, _ pf: Int) {
        let n = pf - olaBase[i]
        if n <= 0 { return }
        let skip = min(n, max(0, nFFTs[i] / 2 - olaBase[i]))
        if skip < n {
            mixIn[i].reserveCapacity(mixIn[i].count + n - skip)
            for j in skip..<n { mixIn[i].append(ola[i][j] / env[i][j]) }
        }
        ola[i].removeFirst(n); env[i].removeFirst(n)
        olaBase[i] = pf
    }

    /// y[m] = sum_c sum_k mixer[c][k] * mix_in_c[m + k] for every m whose taps are available.
    private func mixEmit() -> [Float] {
        let m1 = min(mixIn[0].count, mixIn[1].count) + mixBase - K + 1
        if m1 <= mixNext { return [] }
        let take = m1 - mixBase + K - 1
        let outN = take - K + 1
        var y = [Double](repeating: 0, count: outN)
        var tmp = [Double](repeating: 0, count: outN)
        for c in 0..<2 {
            mixIn[c].withUnsafeBufferPointer { x in
                mixer[c].withUnsafeBufferPointer { k in
                    vDSP_convD(x.baseAddress!, 1, k.baseAddress!, 1, &tmp, 1, vDSP_Length(outN), vDSP_Length(K))  // correlation
                }
            }
            vDSP_vaddD(y, 1, tmp, 1, &y, 1, vDSP_Length(outN))
        }
        let drop = m1 - mixBase
        mixIn[0].removeFirst(drop); mixIn[1].removeFirst(drop)
        mixBase = m1; mixNext = m1
        return y.map { Float($0) }
    }
}

/// Streaming torchaudio sinc resampler: out[n*new + i] = sum_k kernel[i][k] * x[n*orig + k - width].
final class ResamplerStream {
    let orig: Int, new: Int, width: Int, kl: Int
    let kernel: [Double]  // [new * kl], row-major
    var buf: [Double] = []
    var base = 0  // absolute index of buf[0]
    var nIn = 0
    var nextBlock = 0

    init(kernel: [Float], orig: Int, new: Int, width: Int) throws {
        self.orig = orig; self.new = new; self.width = width; kl = 2 * width + orig
        guard kernel.count == new * kl else { throw Zonos3Error.invalid("resampler: kernel shape mismatch") }
        self.kernel = kernel.map(Double.init)
    }

    func push(_ x: [Float]) -> [Float] {
        buf.append(contentsOf: x.map(Double.init)); nIn += x.count
        return run(false)
    }
    func flush() -> [Float] { run(true) }

    private func run(_ flush: Bool) -> [Float] {
        let O = orig, N = new, W = width, KL = kl
        var total = 0
        let end: Int
        if flush {
            total = (N * nIn + O - 1) / O
            end = (total + N - 1) / N
        } else {
            let reach = nIn + W - KL
            end = reach >= 0 ? reach / O + 1 : 0
        }
        var out: [Float] = []
        if end > nextBlock {
            let lo = nextBlock * O - W
            let hi = (end - 1) * O - W + KL
            var seg = [Double](repeating: 0, count: hi - lo)
            let a = max(lo, 0), b = min(hi, nIn)
            if b > a { for j in a..<b { seg[j - lo] = buf[j - base] } }
            let blocks = end - nextBlock
            var y = [Double](repeating: 0, count: blocks * N)
            seg.withUnsafeBufferPointer { s in
                kernel.withUnsafeBufferPointer { k in
                    y.withUnsafeMutableBufferPointer { yp in
                        for blk in 0..<blocks {
                            // y[blk] = kernel [N, KL] @ window [KL]
                            cblas_dgemv(CblasRowMajor, CblasNoTrans, Int32(N), Int32(KL), 1.0, k.baseAddress!, Int32(KL),
                                        s.baseAddress! + blk * O, 1, 0.0, yp.baseAddress! + blk * N, 1)
                        }
                    }
                }
            }
            if flush { y = Array(y[0..<max(0, total - nextBlock * N)]) }
            out = y.map { Float($0) }
            nextBlock = end
        }
        let keepFrom = min(nIn, max(0, nextBlock * O - W))
        if keepFrom > base { buf.removeFirst(keepFrom - base); base = keepFrom }
        return out
    }
}

/// Decoder audio -> the served PCM: times the output gain (the decoder's headroom), then the soft limiter
/// (zonos3/tts/audio.py `output_pcm`; knee -1 dBFS).
public func outputPCM(_ samples: [Float], gain: Float) -> [Float] {
    let knee = Float(pow(10.0, -1.0 / 20.0)), width = 1 - knee
    return samples.map { s in
        let x = s * gain
        let a = abs(x)
        return a > knee ? (x < 0 ? -1 : 1) * (knee + width * tanh((a - knee) / width)) : x
    }
}
