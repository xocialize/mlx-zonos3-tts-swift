"""The 27 voices Zyphra ships (assets/default_voices/<Language>/<Name>_<lang[-style]>_<G>_1.embedding.npy, float32 [192])
-> voices.json in a converted checkpoint: {"voices": [{"id", "name", "language", "style", "gender", "embedding"}]}.
  python Tools/pack_voices.py <upstream repo>/assets/default_voices <checkpoint dir>"""
import sys, json, pathlib
import numpy as np

src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
voices = []
for p in sorted(src.glob("*/*.embedding.npy")):
    name, tag, gender, _ = p.name[: -len(".embedding.npy")].split("_")
    lang, _, style = tag.partition("-")
    e = np.load(p).astype(np.float32).reshape(-1)
    assert e.shape == (192,) and np.isfinite(e).all(), p
    voices.append({"id": name.lower(), "name": name, "language": lang, "languageName": p.parent.name, "style": style or None,
                   "gender": {"M": "male", "F": "female"}[gender], "embedding": [float(v) for v in e]})
assert len({v["id"] for v in voices}) == len(voices)
(out / "voices.json").write_text(json.dumps({"source": "Zyphra/ZONOS3-Mini assets/default_voices", "voices": voices}, ensure_ascii=False))
print(len(voices), "voices:", ", ".join(v["id"] for v in voices))
