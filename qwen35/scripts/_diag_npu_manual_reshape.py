"""Test: Manual reshape to static, then compile on NPU without NPUW_LLM.

Strategy:
1. Load the stateful model
2. Convert stateful to stateless (like NPUW_LLM does)
3. Manually reshape: set KV cache dim 2 to a fixed size, keep GDN states
4. Compile on NPU
"""
import openvino as ov
import time

core = ov.Core()
model = core.read_model("models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml")
print(f"Model loaded: {len(model.inputs)} inputs, {len(model.outputs)} outputs")
print(f"ReadValue ops: {sum(1 for op in model.get_ops() if op.get_type_name() == 'ReadValue')}")

# First, convert stateful to stateless (like NPUW_LLM does)
# This turns ReadValue/Assign into Parameters/Results
from openvino._offline_transformations import stateful_to_stateless_transformation
stateful_to_stateless_transformation(model)

print(f"\nAfter StatefulToStateless:")
print(f"Inputs: {len(model.inputs)}")
for inp in model.inputs:
    name = inp.get_any_name()
    shape = inp.get_partial_shape()
    print(f"  {name}: {shape}")

# Now manually reshape
KV_CACHE_SIZE = 2048  # total KV cache size
INPUT_SIZE = 1        # single token decode

new_shapes = {}
for inp in model.inputs:
    name = inp.get_any_name()
    shape = inp.get_partial_shape()

    if "input_ids" in name or "inputs_embeds" in name:
        if len(shape) == 3:
            new_shapes[name] = ov.PartialShape([1, INPUT_SIZE, shape[2]])
        else:
            new_shapes[name] = ov.PartialShape([1, INPUT_SIZE])
    elif "attention_mask" in name:
        new_shapes[name] = ov.PartialShape([1, KV_CACHE_SIZE])
    elif "position_ids" in name:
        if len(shape) == 3:
            new_shapes[name] = ov.PartialShape([3, 1, INPUT_SIZE])
        else:
            new_shapes[name] = ov.PartialShape([1, INPUT_SIZE])
    elif "beam_idx" in name:
        new_shapes[name] = ov.PartialShape([1])
    elif "key" in name or "value" in name:
        # KV cache: set dim 2 (seq_len) to kvcache_size - input_size
        new_shape = list(shape)
        new_shape[0] = 1  # batch
        new_shape[2] = KV_CACHE_SIZE - INPUT_SIZE  # past sequence length
        new_shapes[name] = ov.PartialShape(new_shape)
    elif "conv" in name:
        # Conv state: all static, just set batch to 1
        new_shape = list(shape)
        new_shape[0] = 1
        new_shapes[name] = ov.PartialShape(new_shape)
    elif "recurrent" in name:
        # Recurrent state: all static, just set batch to 1
        new_shape = list(shape)
        new_shape[0] = 1
        new_shapes[name] = ov.PartialShape(new_shape)
    else:
        new_shape = list(shape)
        new_shape[0] = 1
        new_shapes[name] = ov.PartialShape(new_shape)

print(f"\nReshaping to static shapes:")
for name, shape in new_shapes.items():
    if "recurrent" in name or "conv" in name:
        print(f"  {name}: {shape}")
    elif "key" in name or "value" in name:
        print(f"  {name}: {shape}")

try:
    model.reshape(new_shapes)
    print("\nReshape succeeded!")
except Exception as e:
    print(f"\nReshape FAILED: {e}")
    import sys
    sys.exit(1)

# Verify shapes after reshape
print(f"\nAfter reshape, all shapes:")
for inp in model.inputs:
    name = inp.get_any_name()
    shape = inp.get_partial_shape()
    is_static = all(d.is_static for d in shape)
    if not is_static:
        print(f"  WARNING dynamic: {name}: {shape}")

# Now compile on NPU
config = {
    "NPU_USE_NPUW": "YES",
    "NPUW_FOLD": "NO",
}

print(f"\nCompiling on NPU...")
t0 = time.time()
try:
    compiled = core.compile_model(model, "NPU", config)
    elapsed = time.time() - t0
    print(f"SUCCESS! Compilation took {elapsed:.1f}s")
except Exception as e:
    elapsed = time.time() - t0
    print(f"FAILED after {elapsed:.1f}s")
    print(f"Error: {e}")
