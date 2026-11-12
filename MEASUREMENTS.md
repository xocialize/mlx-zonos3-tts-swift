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

Steady-state (post-load) footprint and a capped-pool figure are not measured yet; the manifest must not declare
from these peaks alone.
