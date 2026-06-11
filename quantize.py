import onnx, os, sys
from onnxruntime.quantization import QuantType, QuantizationMode
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.registry import IntegerOpsRegistry

SRC = "models/model.onnx"
OUT = "models/model_quantized.onnx"
PER_CHANNEL = "--per-channel" in sys.argv

model = onnx.load(SRC)
quantizer = ONNXQuantizer(
    model,
    per_channel=PER_CHANNEL,
    reduce_range=False,
    mode=QuantizationMode.IntegerOps,
    static=False,
    weight_qType=QuantType.QInt8,
    activation_qType=QuantType.QUInt8,
    tensors_range=None,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    op_types_to_quantize=list(IntegerOpsRegistry.keys()),  # includes Gather
    extra_options=dict(EnableSubgraph=True, MatMulConstBOnly=True),  # recurse into Loop body
)

quantizer.quantize_model()
onnx.save(quantizer.model.model, OUT)
print(f"fp32 {os.path.getsize(SRC)/1e6:.1f} MB  ->  q8 {os.path.getsize(OUT)/1e6:.1f} MB"
      f"  (per_channel={PER_CHANNEL})")