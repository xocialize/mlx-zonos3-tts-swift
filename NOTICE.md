# Notices

This package is a Swift-MLX port of ZONOS3-Mini.

- **Model weights** — Zyphra, Inc., Apache-2.0 (`Zyphra/ZONOS3-Mini`). Converted without change of values from the
  release GGUFs (`Tools/convert_gguf.py`).
- **Port source** — Zyphra's ZONOS3-Mini runtime, MIT, Copyright (c) 2026 Zyphra, Inc.
  (`github.com/Zyphra/ZONOS3-Mini` @ `4c66ceac`). The Swift files are translations of Zyphra's own
  `servers/mlx/src/zonos3_mlx` (backbone, codec decoder, speaker encoder, sampler, scheduler) and of
  `model/src/zonos3/{tts,codec/stream_host.py}` (prompt, delay, stream decode, host iSTFT / resampler). Nothing here
  derives from `model/src/zonos3/codec/s1dac/vendor` or `model/src/zonos3/models/third_party/antspeaker`.
- **Gated-delta Metal kernel** — `Sources/Zonos3Core/GatedDeltaKernel.swift`, from mlx-lm
  (`mlx_lm/models/gated_delta.py`), MIT, Copyright © 2025 Apple Inc., as adapted by Zyphra.
- **NPY / WAV helpers** — lifted from `xocialize/mlx-step-audio-editx-swift` via `mlx-sam-audio-swift`.
