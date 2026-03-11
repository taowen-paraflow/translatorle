"""Debug script: inspect hybrid model inputs/outputs."""
import openvino as ov

core = ov.Core()
model = core.read_model("models/qwen35/Qwen3.5-0.8B-hybrid/openvino_model.xml")

print("=== INPUTS ===")
for i, inp in enumerate(model.inputs):
    names = inp.get_names()
    shape = inp.get_partial_shape()
    print(f"  [{i}] {names} shape={shape}")

print()
print("=== OUTPUTS ===")
for i, out in enumerate(model.outputs):
    names = out.get_names()
    shape = out.get_partial_shape()
    print(f"  [{i}] {names} shape={shape}")

# Count categories
input_names = set()
for inp in model.inputs:
    input_names.update(inp.get_names())
output_names = set()
for out in model.outputs:
    output_names.update(out.get_names())

recurrent_in = sorted(n for n in input_names if "recurrent" in n)
conv_in = sorted(n for n in input_names if "conv" in n)
kv_in = sorted(n for n in input_names if "key" in n or "value" in n)
gdn_inter = sorted(n for n in output_names if "gdn_intermediate" in n)
recurrent_out = sorted(n for n in output_names if "recurrent" in n)
conv_out = sorted(n for n in output_names if "conv" in n)

print()
print(f"Recurrent inputs: {len(recurrent_in)} -> {recurrent_in[:3]}...")
print(f"Conv inputs: {len(conv_in)} -> {conv_in[:3]}...")
print(f"KV inputs: {len(kv_in)} -> {kv_in[:3]}...")
print(f"GDN intermediates: {len(gdn_inter)} -> {gdn_inter[:4]}...")
print(f"Recurrent outputs: {len(recurrent_out)} -> {recurrent_out[:3]}...")
print(f"Conv outputs: {len(conv_out)} -> {conv_out[:3]}...")
