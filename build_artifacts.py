"""Assemble a transformers.js-style model directory under artifacts/.

under artifacts/
        config.json, tokenizer.json, tokenizer_config.json,
        special_tokens_map.json, vocab.txt, modules.json
        onnx/
          model.onnx            (fp32)
          model_quantized.onnx  (q8 int8)
"""
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "minishlab/potion-multilingual-128M"
ART = Path("artifacts")
ONNX = ART / "onnx"
ONNX.mkdir(parents=True, exist_ok=True)

META = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
]
for name in META:
    try:
        src = hf_hub_download(REPO, name)
        shutil.copy(src, ART / name)
        print(f"  metadata  {name}")
    except Exception as e:
        print(f"  skip      {name}  ({type(e).__name__})")

# ONNX models: copy the ones we already produced locally.
for name in ("model.onnx", "model_quantized.onnx"):
    src = Path("models") / name
    if src.exists():
        shutil.copy(src, ONNX / name)
        print(f"  onnx      onnx/{name}  ({src.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  MISSING   models/{name} — run download.py / quantize.py first")

print(f"\nartifacts/ assembled at {ART.resolve()}")
