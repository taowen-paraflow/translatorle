"""Check causal mask methods in transformers 5.x Qwen3.5."""
import inspect
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5TextModel,
)

# Check mask-related methods
text_methods = [m for m in dir(Qwen3_5TextModel) if "mask" in m.lower() or "causal" in m.lower()]
print("Qwen3_5TextModel mask/causal methods:", text_methods)

causal_methods = [m for m in dir(Qwen3_5ForCausalLM) if "mask" in m.lower() or "causal" in m.lower()]
print("Qwen3_5ForCausalLM mask/causal methods:", causal_methods)

# Check forward params
sig = inspect.signature(Qwen3_5TextModel.forward)
print("\nQwen3_5TextModel.forward params:", list(sig.parameters.keys()))

# Check specific methods
for cls_name, cls in [("TextModel", Qwen3_5TextModel), ("ForCausalLM", Qwen3_5ForCausalLM)]:
    for method_name in ["_update_causal_mask", "_prepare_4d_causal_mask"]:
        has = hasattr(cls, method_name)
        print(f"{cls_name}.{method_name}: {has}")
        if has:
            m = getattr(cls, method_name)
            sig = inspect.signature(m)
            print(f"  signature: {sig}")
