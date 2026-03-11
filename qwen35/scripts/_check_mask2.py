"""Check how Qwen3.5 handles attention mask in forward pass."""
import inspect
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5ForCausalLM,
)

# Get the forward source of TextModel to see mask handling
src = inspect.getsource(Qwen3_5TextModel.forward)
# Print lines related to mask
for i, line in enumerate(src.split("\n")):
    if "mask" in line.lower() or "causal" in line.lower():
        print(f"  L{i}: {line}")

print("\n--- _update_linear_attn_mask ---")
if hasattr(Qwen3_5TextModel, "_update_linear_attn_mask"):
    src2 = inspect.getsource(Qwen3_5TextModel._update_linear_attn_mask)
    print(src2[:2000])

# Check if there's a FlexAttention or some other mechanism
print("\n--- Checking attention layer forward ---")
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    src3 = inspect.getsource(Qwen3_5Attention.forward)
    for i, line in enumerate(src3.split("\n")):
        if "mask" in line.lower():
            print(f"  Attn L{i}: {line}")
except Exception as e:
    print(f"Error: {e}")

# Check decoder layer
print("\n--- DecoderLayer forward mask lines ---")
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    src4 = inspect.getsource(Qwen3_5DecoderLayer.forward)
    for i, line in enumerate(src4.split("\n")):
        if "mask" in line.lower():
            print(f"  DecL{i}: {line}")
except Exception as e:
    print(f"Error: {e}")
