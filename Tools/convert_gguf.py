"""GGUF -> MLX safetensors for the Swift port (exact: no requantization, no float conversion).
  ref/ZONOS3-Mini/.venv/bin/python tools/convert_gguf.py <gguf dir> <out dir>
Layout written:
  model.safetensors            backbone; Q4_0/Q8_0 tensors as MLX affine: <name> (uint32 words), <name>.scales, <name>.biases
  codec/model.safetensors      keep2_plain decoder        codec-full/model.safetensors  full_mr decoder
  speaker_encoder/model.safetensors
  config.json                  every GGUF metadata key per file + per-tensor {bits} for quantized tensors
  tokenizer.json               the embedded SuperBPE tokenizer (sha256 pinned in config)
F16 stays float16, F32 float32 (F64 -> float32, as upstream's loader reads them)."""
import sys, json, pathlib, hashlib
import numpy as np, mlx.core as mx
from gguf import GGMLQuantizationType as T
from zonos3_mlx.gguf_io import GGUFFile, _repack, GROUP_SIZE

SRC, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
FILES = {"backbone-ptq.gguf": "model.safetensors", "zac-keep2.gguf": "codec/model.safetensors",
         "zac.gguf": "codec-full/model.safetensors", "spk-encoder.gguf": "speaker_encoder/model.safetensors"}
config = {"format": "zonos3-mlx-swift/1", "group_size": GROUP_SIZE, "source": "Zyphra/ZONOS3-Mini gguf/", "files": {}}
for gguf, rel in FILES.items():
    f = GGUFFile(SRC / gguf); arrays, quant, meta = {}, {}, {}
    for k in sorted(f.reader.fields):
        if k.startswith("GGUF."): continue
        v = f.meta(k)
        if k.endswith("tokenizer.json"):
            (OUT / "tokenizer.json").parent.mkdir(parents=True, exist_ok=True)
            (OUT / "tokenizer.json").write_text(v); meta["tokenizer_sha256"] = hashlib.sha256(v.encode()).hexdigest(); continue
        meta[k] = v.tolist() if hasattr(v, "tolist") else v
    for name in f.names():
        kind = f.tensor_type(name)
        if kind in (T.Q4_0, T.Q8_0):
            w = _repack(np.asarray(f._tensor(name).data, dtype=np.uint8), f.shape(name), kind)
            base = name[: -len(".weight")] if name.endswith(".weight") else name
            arrays[name] = w.w; arrays[base + ".scales"] = w.scales; arrays[base + ".biases"] = w.biases
            quant[name] = w.bits
        elif name == "output.weight" and kind == T.F16 and int(f.meta("general.file_type", 1)) not in (0, 1):
            # upstream quantizes a quantized file's F16 logits head to Q8_0 at load (Zonos3MLX.load); done here, same call
            w = f.weight_q8_0(name)
            arrays[name] = w.w; arrays["output.scales"] = w.scales; arrays["output.biases"] = w.biases; quant[name] = w.bits
        elif kind == T.F16:
            arrays[name] = mx.array(f.array(name))
        elif kind in (T.F32, T.F64):
            arrays[name] = mx.array(f.array(name).astype(np.float32))
        elif kind == T.I32:
            arrays[name] = mx.array(f.array(name))
        else:
            raise SystemExit(f"{gguf}:{name}: unsupported {kind.name}")
    p = OUT / rel; p.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(p), arrays, metadata={"format": "mlx"})
    config["files"][rel] = {"source": gguf, "meta": meta, "quantized": quant, "tensors": len(arrays)}
    print(f"{gguf} -> {rel}: {len(arrays)} arrays, {p.stat().st_size / 1e6:.1f} MB, quantized {len(quant)}")
(OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
