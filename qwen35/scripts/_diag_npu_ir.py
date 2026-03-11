"""Diagnostic: inspect NPU model IR structure."""
import openvino as ov

core = ov.Core()
model = core.read_model(r"models\qwen35\Qwen3.5-0.8B-npu\openvino_model.xml")

print("=== Model Inputs ===")
for inp in model.inputs:
    name = inp.get_any_name()
    shape = inp.get_partial_shape()
    print("  {}: {}".format(name, shape))

print()
print("=== Model Outputs ===")
for out in model.outputs:
    name = out.get_any_name()
    shape = out.get_partial_shape()
    print("  {}: {}".format(name, shape))

print()
print("=== Stateful Variables (ReadValue/Assign) ===")
state_count = 0
for op in model.get_ops():
    if op.get_type_name() == "ReadValue":
        state_count += 1
        name = op.get_friendly_name()
        shape = op.get_output_partial_shape(0)
        print("  ReadValue: {} -> {}".format(name, shape))
print("Total: {} stateful variables".format(state_count))

print()
print("=== Try CPU compile (verify IR correctness) ===")
try:
    compiled = core.compile_model(model, "CPU")
    print("CPU compile: OK")
    del compiled
except Exception as e:
    print("CPU compile FAILED: {}".format(e))

print()
print("=== Try NPU compile without NPUW_LLM ===")
try:
    compiled = core.compile_model(model, "NPU", {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "NO",
    })
    print("NPU (no NPUW_LLM) compile: OK")
    del compiled
except Exception as e:
    err = str(e)[:500]
    print("NPU (no NPUW_LLM) compile FAILED: {}".format(err))
