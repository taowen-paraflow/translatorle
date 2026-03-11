"""Inspect the NPU IR inputs/outputs."""
import openvino as ov

core = ov.Core()
model = core.read_model("models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml")

print("=== INPUTS ===")
for i, inp in enumerate(model.inputs):
    name = inp.get_any_name()
    shape = inp.get_partial_shape()
    dtype = inp.get_element_type()
    print(f"  [{i:2d}] {name:50s} {str(shape):30s} {dtype}")

print(f"\nTotal inputs: {len(model.inputs)}")

print("\n=== OUTPUTS ===")
for i, out in enumerate(model.outputs):
    name = out.get_any_name()
    shape = out.get_partial_shape()
    dtype = out.get_element_type()
    print(f"  [{i:2d}] {name:50s} {str(shape):30s} {dtype}")

print(f"\nTotal outputs: {len(model.outputs)}")

# Check for stateful variables
compiled = core.compile_model(model, "CPU")
req = compiled.create_infer_request()
states = req.query_state()
print(f"\n=== STATEFUL VARIABLES: {len(states)} ===")
for s in states:
    print(f"  {s.name}")
