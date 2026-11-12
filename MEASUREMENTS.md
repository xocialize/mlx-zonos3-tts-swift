# mlx-zonos3-tts-swift — measurements

M5 Max / 128 GB, macOS 27.3, release build, `zonos3-gates --synth`, PTQ backbone, MLX buffer pool uncapped.
All numbers below were read from runs on 2026-11-11.

## End to end (S2b): 10 English dub lines × 3 seeds, voice = Roxy (embedding)

Scored with the fleet's unchanged scorers (`WIP/dia-eval/harness/score_dia.py`).

| system | word err | fail > 0.30 | cos to Roxy (min) | OVRL | trimmed s |
|---|---|---|---|---|---|
| this port | 0.002 | 0 % | 0.794 (0.731) | 3.37 | 4.75 |
| upstream MLX server (E30 V0) | 0.005 | 0 % | 0.800 (0.670) | 3.42 | 4.73 |

Different sampling noise, so takes differ; the distributions agree.

## Speed (warm, in process, one request at a time)

| configuration | × real time | first audio |
|---|---|---|
| `keep2_plain`, 6 s lines | 8.9–9.6 | 26–33 ms |
| `keep2_plain`, 10–11 s lines | 9.8–10.1 | 29–49 ms |
| `full_mr`, 6 s lines | 5.5–5.6 | 68 ms |

Component speeds: teacher-forced LM 208 rows/s (15 rows = 1 s of audio); `keep2_plain` decode alone 35× real time;
speaker embedding 0.09 s warm (0.63 s first call) for a 10.2 s clip.

## Memory (`/usr/bin/time -l`, peak memory footprint of the CLI process)

| configuration | peak |
|---|---|
| backbone + `keep2_plain` + speaker encoder, 12 takes of 10–11 s | 3.28 GB |
| backbone + `full_mr` + speaker encoder, 6 takes of 6 s | 5.47 GB |

Those are uncapped-pool CLI peaks. The manifest is declared from the engine run below.

## Through MLXServeEngine (`zonos3-gates validate`, MLX cache 1 GiB) — the manifest's source

Three runs, 26 / 26 checks each time in the last two (the first predates the re-prepare check).

| point | phys_footprint |
|---|---|
| after `prepare` (resident) | 1.60 GB, all runs |
| lifetime peak after the short requests (named, cloned, fast, slow, French, streamed) | 2.75 – 2.79 GB |
| lifetime peak after a 67.8 s request | 2.76 – 2.79 GB (activation 1.16 – 1.19 GB; no growth with length) |
| after evict | 1.45 – 1.49 GB with MLX active 0.00 GB; a second prepare lands at 1.69 GB, a second evict at 0.56 GB |

Declared: 1.75 GB resident + 1.5 GB activation.

Speed through the engine: 8.2 – 9.3× real time on 6 s lines, 10.4× on the 67.8 s request; first streamed chunk after
92 – 95 ms (engine stream, first request of that kind); cancel returns 4 – 8 ms after the request. The first request
after load took 9.6 s (kernel compilation) — the package has no load-time warm-up.

The same seed returns the same bytes; `run()` and the stream of the same seed agree to 16-bit rounding (4.3e-5).

Engine-path takes scored with the fleet scorers (one take each): cloned from the 24 kHz reference through
AVAudioConverter, cosine to Roxy 0.792 / 0.813; `speakingRate` 24 vs 9 gives 3.93 s vs 5.70 s for the same line; word
error 0 on all six.
