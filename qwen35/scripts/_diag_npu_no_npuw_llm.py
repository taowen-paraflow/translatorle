"""Test: Can we compile the NPU model without NPUW_LLM?

The model is already static (seq_len=1), so NPUW_LLM's reshape_to_static
should not be needed. Let's see if plain NPU compilation works.
"""
import openvino as ov
import time

core = ov.Core()
model = core.read_model("models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml")
print(f"Model loaded: {len(model.inputs)} inputs, {len(model.outputs)} outputs")

# Try plain NPU compilation without NPUW_LLM
config = {
    "NPU_USE_NPUW": "YES",  # Still use NPUW for subgraph partitioning
    # But NO "NPUW_LLM": "YES" - skip the LLM pipeline
    "NPUW_FOLD": "NO",  # Needed for non-uniform layers
}

print(f"\nCompiling on NPU with config: {config}")
t0 = time.time()
try:
    compiled = core.compile_model(model, "NPU", config)
    elapsed = time.time() - t0
    print(f"SUCCESS! Compilation took {elapsed:.1f}s")
    print(f"Compiled inputs: {len(compiled.inputs)}")
    for inp in compiled.inputs:
        print(f"  {inp.get_any_name()}: {inp.get_partial_shape()}")
except Exception as e:
    elapsed = time.time() - t0
    print(f"FAILED after {elapsed:.1f}s")
    print(f"Error: {e}")
