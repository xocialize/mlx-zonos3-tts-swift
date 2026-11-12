# mlx-zonos3-tts-swift

Zyphra's [ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) — real-time multilingual text-to-speech at 44.1 kHz —
on Swift-MLX. About 9–10× real time and ~30 ms to first audio on an M5 Max, 3.3 GB peak with the small decoder.

Status: the core (`Zonos3Core`) is parity-gated against Zyphra's own MLX implementation; the MLXEngine package is not
written yet. Not published; weights are a local conversion.

```swift
let tts = try Zonos3TTS(directory: checkpointURL)
let voice = try tts.embedSpeaker(pcm44100: referenceSamples)       // 192-d, or pass nil for no voice
let result = try tts.synthesize(text: "Hello from ZONOS3 Mini.", speakerEmbedding: voice) { chunk in
    play(chunk); return true                                        // 44.1 kHz mono, as it becomes final
}
```

What is and is not ported, the gates and the numbers: [PORTING-SPEC.md](PORTING-SPEC.md),
[MEASUREMENTS.md](MEASUREMENTS.md). Voice cloning scores below the fleet's Character engines (cosine 0.79–0.80 against
VoxCPM2's 0.875 on the same voice); this is a fast multilingual speaker. Text is not normalized — write numbers and
abbreviations out.

Licences: [NOTICE.md](NOTICE.md). Weights Apache-2.0 (Zyphra); this code MIT.
