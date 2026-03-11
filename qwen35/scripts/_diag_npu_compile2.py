"""Diagnostic: test NPU model on CPU first, then try NPU compilation strategies."""
import time
import openvino as ov

core = ov.Core()
model_path = "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml"

# Test 1: CPU compilation (should always work)
print("=== Test 1: CPU compilation ===")
try:
    model = core.read_model(model_path)
    t0 = time.time()
    compiled = core.compile_model(model, "CPU")
    print(f"  SUCCESS in {time.time()-t0:.1f}s")
    print(f"  Inputs: {len(compiled.inputs)}")
    print(f"  Outputs: {len(compiled.outputs)}")
    req = compiled.create_infer_request()
    states = req.query_state()
    print(f"  Stateful variables: {len(states)}")
    for s in states[:3]:
        print(f"    {s.name}: {s.state.shape}")
    del compiled, req
except Exception as e:
    print(f"  FAILED: {e}")

# Test 2: NPU with NPUW_LLM, explicit GDN states renamed to avoid confusion
print("\n=== Test 2: NPU with NPUW_LLM ===")
try:
    model2 = core.read_model(model_path)
    config = {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_LLM_BATCH_DIM": "0",
        "NPUW_LLM_SEQ_LEN_DIM": "2",
        "NPUW_LLM_MAX_PROMPT_LEN": "1",
        "NPUW_LLM_MIN_RESPONSE_LEN": "128",
        "NPUW_FOLD": "NO",
        "NPUW_LLM_PREFILL_HINT": "STATIC",
    }
    t0 = time.time()
    compiled = core.compile_model(model2, "NPU", config)
    print(f"  SUCCESS in {time.time()-t0:.1f}s")
    del compiled
except Exception as e:
    err_str = str(e)
    if len(err_str) > 200:
        err_str = err_str[:200] + "..."
    print(f"  FAILED: {err_str}")

# Test 3: Check what NPUW_LLM does to the model by examining shapes
print("\n=== Test 3: Model structure analysis ===")
model3 = core.read_model(model_path)
# Print all ReadValue shapes and their connected consumers
for op in model3.get_ops():
    if op.get_type_name() == "ReadValue":
        shape = op.get_output_partial_shape(0)
        # Check if dim 2 is dynamic
        is_dim2_dynamic = shape.rank.get_length() >= 3 and shape[2].is_dynamic
        print(f"  ReadValue {op.get_friendly_name()}: {shape} (dim2_dynamic={is_dim2_dynamic})")
