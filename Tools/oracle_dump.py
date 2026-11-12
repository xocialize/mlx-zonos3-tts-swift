"""E30 V1 oracle: goldens for the Swift port from Zyphra's own MLX implementation (zonos3_mlx @ 4c66cea).
Run with the upstream venv:  ref/ZONOS3-Mini/.venv/bin/python tools/oracle_dump.py <gguf dir> <out dir>
Everything is saved as .npy (float32 / int32) plus one meta.json."""
import sys, json, pathlib
import numpy as np, torch, mlx.core as mx

W, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
from zonos3_mlx.engine import Engine, plan_request
from zonos3_mlx.scheduler import Scheduler
from zonos3_mlx.models.lm import LMState
from zonos3_mlx.sampling import noise_for
from zonos3_mlx.audio_io import decode_pcm
from zonos3.tts.plan import Speaker
from zonos3.tts.superbpe import text_ids
from zonos3.tts.delay import shear_up

def save(name, a, dtype=np.float32):
    p = OUT / f"{name}.npy"; p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.ascontiguousarray(np.asarray(a), dtype=dtype))

REF = "/Volumes/Satechi/Development/mlxengine-audio/WIP/dia-eval/cues/ref_s1.wav"
TEXT = "We don't have much time. The bridge goes down at midnight, and every guard in the city knows our faces."
meta = {"text": TEXT, "upstream": "Zyphra/ZONOS3-Mini@4c66ceac", "backbone": "backbone-ptq.gguf", "decoder": "zac-keep2.gguf"}

engine = Engine.load(W / "backbone-ptq.gguf", zac=[str(W / "zac-keep2.gguf")])
c = engine.config

# ---- tokenizer
TEXTS = [TEXT, "Hello, world!", "It's 3:45 p.m. on May 5th, 2026 — costs $12.50.", "Nous n'avons pas beaucoup de temps.",
         "Die Brücke wird um Mitternacht gesperrt.", "あまり時間がありません。", "我们没有多少时间了。", "시간이 별로 없습니다.",
         "У нас мало времени.", "ليس لدينا الكثير من الوقت.", "हमारे पास ज़्यादा समय नहीं है।", "  leading and   multiple   spaces ",
         "UPPER lower MiXeD 12345", "emoji 😀 and tabs\tand\nnewlines", "naïve façade coöperate — “quotes” ‘single’ …"]
spec = c.text_tokenizer_spec
meta["tokenizer"] = [{"text": t, "ids": text_ids(t, spec)} for t in TEXTS]

# ---- speaker encoder
pcm = decode_pcm(pathlib.Path(REF).read_bytes())
save("spk/pcm_44100", pcm)
dump = {}
emb = engine.speaker_encoder.embed(pcm, dump=dump) if hasattr(engine, "speaker_encoder") else None
if emb is None:
    from zonos3_mlx.models.speaker_encoder import SpeakerEncoder
    emb = SpeakerEncoder.load(W / "spk-encoder.gguf").embed(pcm, dump=dump)
save("spk/embedding", emb)
meta["spk_dump"] = sorted(dump)
for k, v in dump.items(): save(f"spk/{k}", v)

# ---- plan
request = {"text": TEXT, "language": "en_us", "stream": False, "seed": 0, "speaker_mode": "embedding"}
plan = plan_request(engine, request, Speaker("embedding", embedding=torch.from_numpy(np.asarray(emb, dtype=np.float32))))
b = plan.cond
ids = b.input_ids.numpy().astype(np.int32)
save("lm/prompt_ids", ids, np.int32)
meta["plan"] = dict(normalized_text=plan.text, speaker_row=int(b.speaker_row), condition_values=[int(v) for v in b.condition_values or []],
                    condition_token_start=int(b.condition_token_start), prefill_frames=0 if plan.prefill_rows is None else int(len(plan.prefill_rows)),
                    max_tokens=int(plan.max_tokens), seed=int(plan.seed), twin=plan.twin is not None, prompt_rows=int(ids.shape[0]))
save("lm/speaker_embedding_row", b.speaker_embedding.numpy())

# ---- prefill captures + one decode step, on the GPU stream (quantized matmuls are Metal-only)
cap = {}
state = LMState()
out = engine.lm.forward(ids, state, speaker_rows=b.speaker_rows(), condition_rows=b.condition_rows(), capture=cap)
for k, v in cap.items(): save(f"lm/prefill/{k}", v)
meta["lm_captures"] = sorted(cap)
save("lm/prefill_logits_last", np.array(out.logits[-1]))

# ---- full generation with the scheduler (reference path), then the noise stream regenerated from the seed
sched = Scheduler(engine, batch=1); sched.start(warmup=False)
try:
    res = sched.synthesize(request, Speaker("embedding", embedding=torch.from_numpy(np.asarray(emb, dtype=np.float32))))
finally:
    sched.stop()
rows = np.stack([np.asarray(r) for r in res.rows]).astype(np.int32)
save("gen/rows", rows, np.int32)
save("gen/pcm", res.pcm)
meta["gen"] = dict(n_rows=int(len(rows)), eos_frame=None if res.eos_frame is None else int(res.eos_frame), samples=int(len(res.pcm)))
g = torch.Generator().manual_seed(int(plan.seed)); width = max(c.eoa_id, c.audio_pad_id) + 1
save("gen/noise", np.stack([noise_for(g, c.n_codebooks, width) for _ in range(len(rows))]))

# teacher-forced logits for the first 12 generated rows (fresh state): row k's logits predict rows[k]
state = LMState()
out = engine.lm.forward(ids, state, speaker_rows=b.speaker_rows(), condition_rows=b.condition_rows())
tf = [np.array(out.logits[-1])]
for k in range(11):
    out = engine.lm.forward(rows[k][None], state)
    tf.append(np.array(out.logits[-1]))
save("gen/tf_logits", np.stack(tf))

# ---- codec: the undelayed pre-EOS frames, whole decode and the incremental stream
dec = engine.decoder_of(None)
pad = c.audio_pad_id
eos = res.eos_frame if res.eos_frame is not None else len(rows) - (c.n_codebooks - 1)
full = np.concatenate([plan.prefill_rows.numpy()[:, : c.n_codebooks], rows[:, : c.n_codebooks]]) if plan.prefill_rows is not None else rows[:, : c.n_codebooks]
und = shear_up(torch.from_numpy(full.astype(np.int64)), pad).numpy()
P = 0 if plan.prefill_rows is None else len(plan.prefill_rows)
codes = und[: P + eos]
assert ((codes >= 0) & (codes < c.codebook_size)).all(), "non-codec ids in the decode window"
save("codec/codes", codes, np.int32)
whole = dec.decode(codes); save("codec/pcm_whole", whole)
st = dec.stream(); parts = []
sizes = [3, 1, 2, 4, 8] + [15] * 50; i = 0
for n in sizes:
    if i >= len(codes): break
    parts.append(st.decode(codes[i : i + n])); i += n
parts.append(st.flush())
save("codec/pcm_stream", np.concatenate(parts))
meta["codec"] = dict(frames=int(len(codes)), prefill=int(P), hop=int(dec.hop), gain=float(dec.output_gain(None)),
                     stream_chunks=[int(len(p)) for p in parts], whole_vs_stream_max_abs=float(np.abs(whole[: len(np.concatenate(parts))] - np.concatenate(parts)[: len(whole)]).max()))
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print(json.dumps({k: meta[k] for k in ("plan", "gen", "codec")}, indent=1))
