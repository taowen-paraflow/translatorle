"""Diagnostic: inspect NPU model IR shapes."""
import openvino as ov

core = ov.Core()
model = core.read_model("models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml")

print("=== INPUTS ===")
for i, inp in enumerate(model.inputs):
    name = inp.get_any_name()
    shape = inp.get_partial_shape()
    dtype = inp.get_element_type()
    print(f"  [{i}] {name}: {shape} {dtype}")

print(f"\n=== OUTPUTS ({len(model.outputs)}) ===")
for i, out in enumerate(model.outputs):
    name = out.get_any_name()
    shape = out.get_partial_shape()
    dtype = out.get_element_type()
    print(f"  [{i}] {name}: {shape} {dtype}")

print(f"\n=== SINKS (stateful) ===")
for s in model.get_sinks():
    print(f"  {s}")

print(f"\nTotal inputs: {len(model.inputs)}, outputs: {len(model.outputs)}, sinks: {len(model.get_sinks())}")

# Check for any 0-dimensions
print("\n=== Checking for 0-dimensions ===")
for i, inp in enumerate(model.inputs):
    shape = inp.get_partial_shape()
    for d in range(shape.rank.get_length()):
        dim = shape[d]
        if dim.is_static and dim.get_length() == 0:
            print(f"  WARNING: Input [{i}] {inp.get_any_name()} has 0 at dim {d}: {shape}")

# Check ReadValue shapes
print("\n=== ReadValue ops ===")
for op in model.get_ops():
    if op.get_type_name() == "ReadValue":
        print(f"  {op.get_friendly_name()}: {op.get_output_partial_shape(0)} {op.get_output_element_type(0)}")
