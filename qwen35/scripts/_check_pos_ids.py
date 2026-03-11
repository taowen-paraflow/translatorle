"""Check if position_ids needs to be [4,B,S] or [3,B,S] in transformers 5.x."""
import inspect
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

src = inspect.getsource(Qwen3_5TextModel.forward)
# Print lines around position_ids handling
lines = src.split("\n")
for i, line in enumerate(lines):
    if "position_ids" in line:
        print(f"L{i}: {line}")
