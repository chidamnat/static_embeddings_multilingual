# Step 1 - setup
```
uv init
uv venv
```

# Step 2 - download static embeddings multilingual model from HF
uv run download.py


## Onnx & transformers.js (option 1)
#### Step 3a - quantization
uv run quantize.py

#### Step 3b - build artifacts for transformers js format (onnx)
uv run build_artifacts.py

#### Step 3c - Recall metrics
uv run recall.py

#### Step 3d - verify quantized vs fp32 results for comparison
uv run recall.py


## npy & zstd custom format
#### Step 4a - Custom artifacts
uv run build_custom_artifacts.py