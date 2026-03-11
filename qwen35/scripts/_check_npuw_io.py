"""Check what inputs/outputs the NPUW_LLM compiled model exposes."""
import openvino as ov
from qwen35.config import NPUW_LLM_OV_CONFIG, NPUW_MODEL_DIR

core = ov.Core()
m = core.read_model(str(NPUW_MODEL_DIR / "openvino_model.xml"))

print("=== Original model inputs ===")
for inp in m.inputs:
    print(f"  {inp.get_any_name()}: {inp.get_partial_shape()} {inp.get_element_type()}")
print(f"  Total: {len(m.inputs)}")

print("=== Original model outputs ===")
for out in m.outputs:
    print(f"  {out.get_any_name()}: {out.get_partial_shape()} {out.get_element_type()}")
print(f"  Total: {len(m.outputs)}")

print()
print("Compiling with NPUW_LLM...")
c = core.compile_model(m, "NPU", NPUW_LLM_OV_CONFIG)

print("=== Compiled model inputs ===")
for inp in c.inputs:
    print(f"  {inp.get_any_name()}: {inp.get_partial_shape()} {inp.get_element_type()}")
print(f"  Total: {len(c.inputs)}")

print("=== Compiled model outputs ===")
for out in c.outputs:
    print(f"  {out.get_any_name()}: {out.get_partial_shape()} {out.get_element_type()}")
print(f"  Total: {len(c.outputs)}")
