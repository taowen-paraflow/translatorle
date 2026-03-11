"""Diagnose: what shapes does StatefulToStateless produce?"""
import openvino as ov

core = ov.Core()
model = core.read_model("models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml")

print("=== BEFORE StatefulToStateless ===")
print(f"Inputs: {len(model.inputs)}")
for inp in model.inputs:
    print(f"  {inp.get_any_name()}: {inp.get_partial_shape()}")

print(f"\nReadValue ops: {sum(1 for op in model.get_ops() if op.get_type_name() == 'ReadValue')}")

# Use the C++ pass via the Manager
from openvino import _pyopenvino as _ov
manager = _ov.passes.Manager()
# Try to find and run StatefulToStateless
try:
    s2s = ov.pass_pack.StatefulToStateless()
except:
    pass

# Alternative: use ov.pass directly
try:
    from openvino._pyopenvino import passes
    print("\nAvailable passes:", [x for x in dir(passes) if not x.startswith('_')])
except:
    pass

# Try the transformation via offline API
try:
    from openvino._offline_transformations import apply_make_stateful_transformation
    print("\nHave apply_make_stateful_transformation")
except ImportError:
    pass

# Let's just manually simulate what StatefulToStateless does:
# It creates Parameters with the same shapes as ReadValue outputs
print("\n=== Simulated post-StatefulToStateless shapes ===")
print("(These are the shapes reshape_to_static would see)")
for op in model.get_ops():
    if op.get_type_name() == "ReadValue":
        vid = op.get_variable_id()
        shape = op.get_output_partial_shape(0)
        rank = len(shape)
        if rank >= 3:
            dim2 = shape[2]
            dim2_static = dim2.is_static
            print(f"  {vid}: {shape}  dim2={dim2} static={dim2_static}")
        else:
            print(f"  {vid}: {shape} (rank < 3)")
