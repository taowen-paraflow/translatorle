"""Find create_causal_mask origin and check position_ids handling."""
import inspect
from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

# Where does create_causal_mask come from?
src = inspect.getsource(mod)
for i, line in enumerate(src.split("\n")):
    if "create_causal_mask" in line and ("import" in line or "from" in line or "def " in line):
        print(f"L{i}: {line}")

# Check if it's defined in the module or imported
print("\ncreate_causal_mask type:", type(getattr(mod, 'create_causal_mask', None)))

# Get its source
try:
    fn = mod.create_causal_mask
    src_file = inspect.getfile(fn)
    print(f"defined in: {src_file}")
    src_fn = inspect.getsource(fn)
    print(f"\n--- create_causal_mask source (first 2000 chars) ---")
    print(src_fn[:2000])
except Exception as e:
    print(f"Error getting source: {e}")

# Check rotary_emb signature
print("\n--- rotary_emb ---")
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    # Check the __init__ to see rotary_emb type
    init_src = inspect.getsource(Qwen3_5TextModel.__init__)
    for i, line in enumerate(init_src.split("\n")):
        if "rotary" in line.lower():
            print(f"  L{i}: {line}")
except Exception as e:
    print(f"Error: {e}")
