#!/usr/bin/env python3
"""Inspect the NPUW model's input/output names and shapes."""
import openvino as ov

core = ov.Core()
model = core.read_model("models/qwen35/Qwen3.5-0.8B-npuw/openvino_model.xml")

print("=== INPUTS ===")
for inp in model.inputs:
    name = inp.get_any_name()
    shape = inp.get_partial_shape()
    print("  " + name + ": " + str(shape))

print("\n=== OUTPUTS ===")
for out in model.outputs:
    names = list(out.get_names())
    shape = out.get_partial_shape()
    print("  " + str(names) + ": " + str(shape))

print("\n=== GDN State Mapping Check ===")
input_names = set()
for inp in model.inputs:
    input_names.add(inp.get_any_name())

gdn_outputs = []
kv_outputs = []
for out in model.outputs:
    for n in out.get_names():
        if "present" in n and ("conv" in n or "recurrent" in n):
            gdn_outputs.append(n)
        elif "present" in n and ("key" in n or "value" in n):
            kv_outputs.append(n)

gdn_inputs = [n for n in sorted(input_names) if "past" in n and ("conv" in n or "recurrent" in n)]

print("GDN outputs (" + str(len(gdn_outputs)) + "): " + str(sorted(gdn_outputs)[:5]))
print("GDN inputs  (" + str(len(gdn_inputs)) + "): " + str(gdn_inputs[:5]))

print("\n=== NPUW_LLM Name Mapping Simulation ===")
for out_name in sorted(gdn_outputs)[:5]:
    hf_mapped = out_name.replace("present", "past_key_values")
    fallback_mapped = out_name.replace("present", "past")
    hf_match = hf_mapped in input_names
    fb_match = fallback_mapped in input_names
    print("  Output: " + out_name)
    print("    HF map -> " + hf_mapped + " : " + ("MATCH" if hf_match else "NO MATCH"))
    print("    Fallback -> " + fallback_mapped + " : " + ("MATCH" if fb_match else "NO MATCH"))

print("\n=== KV Cache Mapping ===")
for out_name in sorted(kv_outputs)[:3]:
    hf_mapped = out_name.replace("present", "past_key_values")
    fb_mapped = out_name.replace("present", "past")
    hf_match = hf_mapped in input_names
    print("  Output: " + out_name)
    print("    HF map -> " + hf_mapped + " : " + ("MATCH" if hf_match else "NO MATCH"))

print("\n=== Stateful ops ===")
reads = [op for op in model.get_ordered_ops() if op.get_type_name() == "ReadValue"]
assigns = [op for op in model.get_ordered_ops() if op.get_type_name() == "Assign"]
print("ReadValue: " + str(len(reads)) + ", Assign: " + str(len(assigns)))
