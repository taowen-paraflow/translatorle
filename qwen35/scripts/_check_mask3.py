"""Check create_causal_mask function."""
import inspect
from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

# Check if create_causal_mask is in the module
if hasattr(mod, 'create_causal_mask'):
    src = inspect.getsource(mod.create_causal_mask)
    print("create_causal_mask source:")
    print(src[:3000])
else:
    # Search for it
    print("create_causal_mask not in module, searching...")
    src = inspect.getsource(mod)
    for i, line in enumerate(src.split("\n")):
        if "create_causal_mask" in line:
            print(f"  L{i}: {line}")

# Check the full forward source of TextModel
print("\n\n--- TextModel.forward (first 3000 chars) ---")
src2 = inspect.getsource(mod.Qwen3_5TextModel.forward)
print(src2[:3000])
