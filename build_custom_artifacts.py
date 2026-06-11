"""Build static-embeddings-style artifacts.

    artifacts_custom/models/minishlab/potion-multilingual-128M/
        {fp32,fp16,fp8_e4m3,fp8_e5m2}.d{32,64,128,256}.npy(.zst)
        tokenizer.json(.zst)
        README.md

fp8 arrays are stored as raw uint8 bytes (the .npy header can't describe fp8);
the consumer knows the dtype from the filename and decodes via a 256-entry LUT.
"""
from pathlib import Path
from textwrap import dedent

import numpy as np
import ml_dtypes
import onnx
from onnx import numpy_helper
import zstandard
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "minishlab/potion-multilingual-128M"
OUT = Path("artifacts_custom/models/minishlab/potion-multilingual-128M")
OUT.mkdir(parents=True, exist_ok=True)

MATRYOSHKA_DIMS = [32, 64, 128, 256]
# name -> (numpy/ml_dtypes dtype, store-as-uint8-bytes?)
PRECISIONS = {
    "fp32":     (np.float32,               False),
    "fp16":     (np.float16,               False),
    "fp8_e4m3": (ml_dtypes.float8_e4m3fn,  True),
    "fp8_e5m2": (ml_dtypes.float8_e5m2,    True),
}


def extract_embedding_table() -> np.ndarray:
    """Pull embedding_bag.weight [500353, 256] fp32 straight out of model.onnx."""
    model = onnx.load("models/model.onnx", load_external_data=False)
    for init in model.graph.initializer:
        if init.name == "embedding_bag.weight":
            arr = numpy_helper.to_array(init).astype(np.float32)
            print(f"extracted {init.name}: {arr.shape} {arr.dtype}")
            return arr
    raise RuntimeError("embedding_bag.weight not found in models/model.onnx")


def zst_compress(path: Path) -> None:
    cctx = zstandard.ZstdCompressor(level=10)
    out = path.with_suffix(path.suffix + ".zst")
    with open(path, "rb") as fin, open(out, "wb") as fout:
        cctx.copy_stream(fin, fout)


def save_variant(table_dim: np.ndarray, name: str, dtype, as_u8: bool) -> int:
    casted = table_dim.astype(dtype)
    to_store = casted.view(np.uint8) if as_u8 else casted
    path = OUT / f"{name}.npy"
    np.save(path, to_store)
    zst_compress(path)
    return path.stat().st_size


def normalized_mean_pool(rows: np.ndarray) -> np.ndarray:
    pooled = rows.mean(axis=0)
    return pooled / (np.linalg.norm(pooled) + 1e-12)


def roundtrip(arr_f32: np.ndarray, dtype) -> np.ndarray:
    """fp32 -> dtype -> fp32, to measure quantization loss."""
    return arr_f32.astype(dtype).astype(np.float32)


def cosine_per_vector(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((num / den).mean())


# ---- build ----------------------------------------------------------------

table = extract_embedding_table()                  # [500353, 256] fp32
vocab_size, full_dim = table.shape

print("\nwriting variants:")
rows_md = []
for dim in MATRYOSHKA_DIMS:
    truncated = table[:, :dim]
    for name, (dtype, as_u8) in PRECISIONS.items():
        size = save_variant(truncated, f"{name}.d{dim}", dtype, as_u8)
        print(f"  {name}.d{dim}.npy  {size/1e6:8.1f} MB")

# tokenizer (+ zst)
tok_src = hf_hub_download(REPO, "tokenizer.json")
tok_path = OUT / "tokenizer.json"
tok_path.write_bytes(Path(tok_src).read_bytes())
zst_compress(tok_path)
tokenizer = Tokenizer.from_file(str(tok_path))
print(f"\ntokenizer.json copied ({tok_path.stat().st_size/1e6:.2f} MB)")

# ---- README stats ---------------------------------------------------------

norms = np.linalg.norm(table, axis=1)

# Mean-pooled quantization loss: pool in fp32, compare quantized roundtrip.
phrases = [
    "The committee approved the proposal after a long, heated discussion.",
    "Careful tuning of hyperparameters affects neural network performance.",
    "Despite the heavy rain, the concert continued as planned.",
    "Remote work changed how teams collaborate across time zones.",
    "வேகமான பழுப்பு நரி சோம்பேறி நாயின் மேல் குதிக்கிறது.",
    "Привет, как дела сегодня утром?",
]
loss_dtypes = {"fp16": np.float16, "fp8_e4m3": ml_dtypes.float8_e4m3fn, "fp8_e5m2": ml_dtypes.float8_e5m2}
pooled_cos = {k: [] for k in loss_dtypes}
for phrase in phrases:
    ids = tokenizer.encode(phrase, add_special_tokens=False).ids
    rows = table[np.array(ids, dtype=np.int64)]
    base = normalized_mean_pool(rows)
    for k, dt in loss_dtypes.items():
        rt = normalized_mean_pool(roundtrip(rows, dt))
        pooled_cos[k].append(float(np.dot(base, rt)))
avg_pooled = {k: sum(v) / len(v) for k, v in pooled_cos.items()}

# Per-vector quantization loss over the full table.
pv = {}
for k, dt in loss_dtypes.items():
    rt = roundtrip(table, dt)
    pv[k] = {
        "cos": cosine_per_vector(table, rt),
        "mse": float(np.mean((table - rt) ** 2)),
        "mae": float(np.mean(np.abs(table - rt))),
    }

readme = dedent(f"""
    # minishlab/potion-multilingual-128M (custom static-embedding artifacts)

    Raw embedding-table artifacts for static embedding. 
    The table was extracted from the model's `embedding_bag.weight`; 
    tokenization uses `tokenizer.json` only.

    Files: `{{fp32,fp16,fp8_e4m3,fp8_e5m2}}.d{{32,64,128,256}}.npy(.zst)`.
    fp8 files store raw uint8 bytes (decode via a 256-entry LUT in the consumer).

    ## Model Stats

    | item          | metric  | value |
    | ------------- | ------- | ----- |
    | vocab         | size    | {vocab_size:,} |
    | embedding     | dims    | {full_dim} |
    | vector length | mean    | {norms.mean():.2f} |
    | vector length | median  | {np.median(norms):.2f} |
    | vector length | stddev  | {norms.std():.2f} |
    | values        | mean    | {table.mean():.2f} |
    | values        | stddev  | {table.std():.2f} |

    ## Mean-Pooled Quantization Loss

    Pooling done in fp32; quantized roundtrip compared to the original pooled
    vector (cosine similarity). This reflects real usage.

    | Precision | Cosine Similarity |
    | --------- | ----------------- |
    | fp16      | {avg_pooled['fp16']:.5f} |
    | fp8 e4m3  | {avg_pooled['fp8_e4m3']:.5f} |
    | fp8 e5m2  | {avg_pooled['fp8_e5m2']:.5f} |

    ## Per-Vector Quantization Loss (full table)

    | Precision | Cosine | MSE | MAE |
    | --------- | ------ | --- | --- |
    | fp16      | {pv['fp16']['cos']:.5f} | {pv['fp16']['mse']:.5f} | {pv['fp16']['mae']:.5f} |
    | fp8 e4m3  | {pv['fp8_e4m3']['cos']:.5f} | {pv['fp8_e4m3']['mse']:.5f} | {pv['fp8_e4m3']['mae']:.5f} |
    | fp8 e5m2  | {pv['fp8_e5m2']['cos']:.5f} | {pv['fp8_e5m2']['mse']:.5f} | {pv['fp8_e5m2']['mae']:.5f} |
    """).strip()

(OUT / "README.md").write_text(readme + "\n")
print(f"\nREADME.md written\n\nartifacts at {OUT.resolve()}")
