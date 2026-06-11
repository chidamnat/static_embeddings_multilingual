from huggingface_hub import hf_hub_download
import shutil, pathlib

p = hf_hub_download(
    repo_id="minishlab/potion-multilingual-128M",
    filename="onnx/model.onnx",
)

pathlib.Path("models").mkdir(exist_ok=True)
shutil.copy(p, "models/model.onnx")
print("saved models/model.onnx")