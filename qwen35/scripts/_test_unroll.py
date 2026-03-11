#!/usr/bin/env python3
"""Test multiple approaches to unroll Loop nodes in Qwen3.5 IR."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov
import openvino.passes as ov_passes

print("OpenVINO version:", ov.__version__)
core = ov.Core()

# ============================================================
# Approach 1: LowLatency2 first (dynamic trip), then reshape
# ============================================================
print("\n=== Approach 1: LowLatency2 first, then reshape ===")
model = core.read_model("models/qwen35/Qwen3.5-0.8B-ov/openvino_model.xml")
loop_count = len([op for op in model.get_ordered_ops() if op.get_type_name() in ("Loop", "TensorIterator")])
print("Loops before:", loop_count)

# Apply LowLatency2 with dynamic shapes (adds ReadValue/Assign)
mgr = ov_passes.Manager()
mgr.register_pass(ov_passes.LowLatency2())
mgr.run_passes(model)

loop_count = len([op for op in model.get_ordered_ops() if op.get_type_name() in ("Loop", "TensorIterator")])
rv_count = len([op for op in model.get_ordered_ops() if op.get_type_name() == "ReadValue"])
print("After LowLatency2: Loops={}, ReadValue={}".format(loop_count, rv_count))

# Now reshape to static
print("Reshaping to static shapes...")
new_shapes = {}
for inp in model.inputs:
    name = inp.get_any_name()
    ps = inp.get_partial_shape()
    if "inputs_embeds" in name:
        new_shapes[name] = [1, 1, ps[2].get_length()]
    elif "attention_mask" in name:
        new_shapes[name] = [1, 2]
    elif "position_ids" in name:
        new_shapes[name] = [3, 1, 1]
    elif "beam_idx" in name:
        new_shapes[name] = [1]
    elif ".conv" in name:
        new_shapes[name] = [1] + [d.get_length() for d in ps[1:]]
    elif ".recurrent" in name:
        new_shapes[name] = [1] + [d.get_length() for d in ps[1:]]
    elif ".key" in name or ".value" in name:
        new_shapes[name] = [1, ps[1].get_length(), 1, ps[3].get_length()]
    else:
        new_shapes[name] = [1]  # beam_idx etc

try:
    model.reshape(new_shapes)
    print("Reshape OK")
except Exception as e:
    print("Reshape failed:", e)

# ConstantFolding
mgr2 = ov_passes.Manager()
mgr2.register_pass(ov_passes.ConstantFolding())
mgr2.run_passes(model)

loop_count = len([op for op in model.get_ordered_ops() if op.get_type_name() in ("Loop", "TensorIterator")])
print("After reshape + ConstantFolding: Loops={}".format(loop_count))

if loop_count > 0:
    loop = [op for op in model.get_ordered_ops() if op.get_type_name() in ("Loop", "TensorIterator")][0]
    src = loop.input(0).get_source_output().get_node()
    print("  Trip count:", src.get_type_name())
    if src.get_type_name() == "Constant":
        print("  Trip count value:", src.get_data())

# ============================================================
# Approach 2: Combined manager with ConstantFolding + LowLatency2
# ============================================================
print("\n=== Approach 2: Reshape first, then combined CF+LL2 in one Manager ===")
model2 = core.read_model("models/qwen35/Qwen3.5-0.8B-ov/openvino_model.xml")

# Reshape first
new_shapes2 = {}
for inp in model2.inputs:
    name = inp.get_any_name()
    ps = inp.get_partial_shape()
    if "inputs_embeds" in name:
        new_shapes2[name] = [1, 1, ps[2].get_length()]
    elif "attention_mask" in name:
        new_shapes2[name] = [1, 2]
    elif "position_ids" in name:
        new_shapes2[name] = [3, 1, 1]
    elif "beam_idx" in name:
        new_shapes2[name] = [1]
    elif ".conv" in name:
        new_shapes2[name] = [1] + [d.get_length() for d in ps[1:]]
    elif ".recurrent" in name:
        new_shapes2[name] = [1] + [d.get_length() for d in ps[1:]]
    elif ".key" in name or ".value" in name:
        new_shapes2[name] = [1, ps[1].get_length(), 1, ps[3].get_length()]
    else:
        new_shapes2[name] = [1]
model2.reshape(new_shapes2)
print("Reshape done")

# Apply both in one manager
try:
    mgr3 = ov_passes.Manager()
    mgr3.register_pass(ov_passes.ConstantFolding())
    mgr3.register_pass(ov_passes.LowLatency2())
    mgr3.run_passes(model2)

    loop_count = len([op for op in model2.get_ordered_ops() if op.get_type_name() in ("Loop", "TensorIterator")])
    rv_count = len([op for op in model2.get_ordered_ops() if op.get_type_name() == "ReadValue"])
    print("After combined CF+LL2: Loops={}, ReadValue={}".format(loop_count, rv_count))
except Exception as e:
    print("Combined approach failed:", e)

# ============================================================
# Approach 3: Just test the NPUW IR on CPU
# ============================================================
print("\n=== Approach 3: Test NPUW model (single-step) on CPU ===")
npuw_model = core.read_model("models/qwen35/Qwen3.5-0.8B-npuw/openvino_model.xml")
loop_count = len([op for op in npuw_model.get_ordered_ops() if op.get_type_name() in ("Loop", "TensorIterator")])
print("NPUW model Loops:", loop_count, "(should be 0)")

compiled = core.compile_model(npuw_model, "CPU")
request = compiled.create_infer_request()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("models/qwen35/Qwen3.5-0.8B-npuw", trust_remote_code=True)
embed_table = np.load("models/qwen35/Qwen3.5-0.8B-npuw/embed_tokens.npy").astype(np.float32)

prompt = "Hello, what is your name?"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"][0]
print("Prompt:", prompt, "->", len(input_ids), "tokens")

# Initialize states
conv_states = {}
recurrent_states = {}
kv_caches = {}
for inp in npuw_model.inputs:
    name = inp.get_any_name()
    if "cache_params.past.conv" in name:
        ps = inp.get_partial_shape()
        conv_states[name] = np.zeros([1] + [d.get_length() for d in ps[1:]], dtype=np.float32)
    elif "cache_params.past.recurrent" in name:
        ps = inp.get_partial_shape()
        recurrent_states[name] = np.zeros([1] + [d.get_length() for d in ps[1:]], dtype=np.float32)
    elif "past_key_values" in name:
        ps = inp.get_partial_shape()
        kv_caches[name] = np.zeros((1, ps[1].get_length(), 0, ps[3].get_length()), dtype=np.float32)

# Token-by-token inference
generated = []
past_length = 0
all_tokens = list(input_ids)

for step in range(len(input_ids) + 10):
    token_id = all_tokens[step] if step < len(all_tokens) else generated[-1]
    embeds = embed_table[token_id:token_id+1].reshape(1, 1, -1)

    inp = {
        "inputs_embeds": embeds,
        "attention_mask": np.ones((1, past_length + 1), dtype=np.int64),
        "position_ids": np.full((3, 1, 1), past_length, dtype=np.int64),
        "beam_idx": np.array([0], dtype=np.int64),
    }
    inp.update(conv_states)
    inp.update(recurrent_states)
    inp.update(kv_caches)

    result = request.infer(inp)

    logits = result["logits"][0, -1, :]
    has_nan = np.any(np.isnan(logits))
    next_token = int(np.argmax(logits))
    text = tokenizer.decode([next_token])

    if step < 3 or step == len(input_ids) - 1 or has_nan or step >= len(input_ids):
        print("  Step {}: token='{}' -> next='{}' NaN={} logits=[{:.2f},{:.2f}]".format(
            step, tokenizer.decode([token_id]), text, has_nan,
            float(np.min(logits)), float(np.max(logits))))

    if has_nan:
        # Check which states have NaN
        for i in range(3):
            c = result.get("cache_params.present.conv." + str(i))
            r = result.get("cache_params.present.recurrent." + str(i))
            if c is not None:
                print("    conv.{}: NaN={}".format(i, np.any(np.isnan(c))))
            if r is not None:
                print("    recur.{}: NaN={}".format(i, np.any(np.isnan(r))))
        break

    # Update states
    for name in conv_states:
        out_name = name.replace("past", "present")
        conv_states[name] = np.array(result[out_name])
    for name in recurrent_states:
        out_name = name.replace("past", "present")
        recurrent_states[name] = np.array(result[out_name])
    for name in kv_caches:
        idx = name.replace("past_key_values.", "").split(".")[0]
        kv_type = name.split(".")[-1]
        out_name = "present." + idx + "." + kv_type
        kv_caches[name] = np.array(result[out_name])

    past_length += 1
    if step >= len(input_ids) - 1:
        generated.append(next_token)
        all_tokens.append(next_token)

print("\nGenerated:", tokenizer.decode(generated, skip_special_tokens=True))
print("Done!")
