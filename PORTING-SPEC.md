# mlx-zonos3-tts-swift — porting spec

Swift-MLX port of Zyphra's ZONOS3-Mini (E30, AB-D-0225). Oracle: Zyphra's own Python-MLX implementation
(`servers/mlx`, upstream `4c66ceac`) on the release GGUFs; goldens from `Tools/oracle_dump.py`.

## Scope of this cut

In: the 192-d embedding voice (or no voice), the three condition rows, default sampler, `keep2_plain` and `full_mr`
decoders, incremental decode with early first chunk, the speaker encoder.

Out (not ported): raw-reference prefix + its 4 GB encoder and SSL teacher, speaker CFG, LACI and duration targets,
activation steering, accent adapters, speech editing, phoneme input, **text normalization** (text is tokenized as
written; upstream normalizes 9 languages with `zyphra-textnorm`).

Seeds: upstream draws sampling noise from a `torch.Generator`; this port draws from MLXRandom. A seed is reproducible
here; it does not reproduce upstream's takes.

## Phases

| phase | gate (`zonos3-gates`) | status |
|---|---|---|
| S0 key contract | `--s0`: backbone 190 / 190 keys; `--s1-codec`: decoder 238 / 238 | PASSED 2026-11-11 |
| S1 backbone | `--s1-lm`: 183 prefill captures, worst rel 6.2e-4 (`ffn_out_10`); last-row logits rel 2.1e-4, argmax 10 / 10; 11 teacher-forced decode steps, argmax 110 / 110 | PASSED 2026-11-11 |
| S1 tokenizer | `--s1-tok`: 15 texts (11 scripts / edge cases) id-for-id | PASSED 2026-11-11 |
| S1 speaker encoder | `--s1-spk`: fbank max_abs 0; four stage features rel ≤ 6.1e-4; embedding cosine 1.000000 | PASSED 2026-11-11 |
| S1 codec decoder | `--s1-codec`: stream vs oracle stream SNR 61.4 dB; one chunk vs oracle whole decode 65.4 dB; chunking invariance max_abs 1.9e-3 | PASSED 2026-11-11 |
| S2 prompt | `--s2-prompt`: 32 rows identical, speaker / condition rows identical | PASSED 2026-11-11 |
| S2 generation | `--s2-gen`: oracle noise, teacher-forced: 1 flipped code of 980 (row 42); free-running identical up to that row | PASSED 2026-11-11 |
| S2b end to end | 30 takes scored with the fleet scorers — see MEASUREMENTS.md | PASSED 2026-11-11 |
| S7 engine wrap | `MLXZonos3TTS` on `Capability.tts` | pending |

Gates run as CLI modes on the GPU stream (quantized matmuls are Metal-only).

## Checkpoint layout (`Tools/convert_gguf.py`)

`model.safetensors` (backbone: Q4_0 experts, Q8_0 mixers and logits head as MLX affine, group 32, exact repack),
`codec/` (`keep2_plain`), `codec-full/` (`full_mr`), `speaker_encoder/`, `config.json` (every GGUF metadata key +
quantized tensors' bit widths), `tokenizer.json`.
