import onnx, numpy as np, onnxruntime as ort

# (a) structural: embedding table must come back 8-bit
m = onnx.load("models/model_quantized.onnx", load_external_data=False)
names = {t.name: onnx.TensorProto.DataType.Name(t.data_type) for t in m.graph.initializer}
print("quantized table:", {k: v for k, v in names.items() if "embedding_bag" in k})
assert "embedding_bag.weight_quantized" in names, "table NOT quantized — subgraph pass skipped it"

# (b) functional: compare fp32 vs q8 outputs
so = ort.SessionOptions(); so.log_severity_level = 3
fp32 = ort.InferenceSession("models/model.onnx", so, providers=["CPUExecutionProvider"])
q8   = ort.InferenceSession("models/model_quantized.onnx", so, providers=["CPUExecutionProvider"])

input_ids = np.array([5,6,7,8,9,10,11,12,13,14, 100,101,102,103], dtype=np.int64)
offsets   = np.array([0, 10], dtype=np.int64)
feeds = {"input_ids": input_ids, "offsets": offsets}

a, b = fp32.run(None, feeds)[0], q8.run(None, feeds)[0]
for i in range(a.shape[0]):
    cos = float(np.dot(a[i], b[i]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-9))
    print(f"seq{i}: cosine(fp32,q8) = {cos:.4f}")