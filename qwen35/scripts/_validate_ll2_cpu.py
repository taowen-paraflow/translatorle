#!/usr/bin/env python3
"""Validate the LowLatency2 model on CPU.

Uses the qwen35 venv which should have full OpenVINO (with CPU plugin).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov

print("OpenVINO version:", ov.__version__)
core = ov.Core()
print("Available devices:", core.available_devices)

MODEL_DIR = "models/qwen35/Qwen3.5-0.8B-ll2"

# Check model structure
model = core.read_model(f"{MODEL_DIR}/openvino_model.xml")
ops = model.get_ordered_ops()
loops = sum(1 for op in ops if op.get_type_name() in ("Loop", "TensorIterator"))
rvs = sum(1 for op in ops if op.get_type_name() == "ReadValue")
assigns = sum(1 for op in ops if op.get_type_name() == "Assign")
print(f"Model: Loops={loops}, RV={rvs}, Assign={assigns}")
print(f"Inputs: {len(model.inputs)}")
for inp in model.inputs:
    print(f"  {inp.get_any_name()}: {inp.get_partial_shape()}")
print(f"Outputs: {len(model.outputs)}")
for out in model.outputs:
    print(f"  {out.get_any_name()}: {out.get_partial_shape()}")

if "CPU" not in core.available_devices:
    print("ERROR: CPU device not available")
    sys.exit(1)

# Load tokenizer and embeddings
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
embed_table = np.load(f"{MODEL_DIR}/embed_tokens.npy").astype(np.float32)

# Compile
compiled = core.compile_model(model, "CPU")
request = compiled.create_infer_request()

prompt = "Hello, what is your name?"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"][0]
print(f"\nPrompt: '{prompt}' -> {len(input_ids)} tokens")

# Token-by-token inference
generated = []
past_length = 0
all_tokens = list(input_ids)

for step in range(len(input_ids) + 15):
    token_id = all_tokens[step] if step < len(all_tokens) else generated[-1]
    embeds = embed_table[token_id:token_id+1].reshape(1, 1, -1)

    inp = {
        "inputs_embeds": embeds,
        "attention_mask": np.ones((1, past_length + 1), dtype=np.int64),
        "position_ids": np.full((3, 1, 1), past_length, dtype=np.int64),
        "beam_idx": np.array([0], dtype=np.int32),
    }

    result = request.infer(inp)

    logits = None
    for out in compiled.outputs:
        name = out.get_any_name()
        if "logits" in name:
            logits = result[out]
            break

    if logits is None:
        print("ERROR: no logits output")
        break

    logits_1d = logits[0, -1, :]
    has_nan = np.any(np.isnan(logits_1d))
    next_token = int(np.argmax(logits_1d))
    text = tokenizer.decode([next_token])

    if step < 3 or step == len(input_ids) - 1 or has_nan or step >= len(input_ids):
        print(f"  Step {step}: '{tokenizer.decode([token_id])}' -> '{text}' NaN={has_nan} logits=[{float(np.min(logits_1d)):.2f},{float(np.max(logits_1d)):.2f}]")

    if has_nan:
        print("ERROR: NaN in logits!")
        # Check states
        states = request.query_state()
        print(f"  Number of states: {len(states)}")
        for s in states[:5]:
            data = s.state.data
            print(f"    {s.name}: shape={data.shape} NaN={np.any(np.isnan(data))}")
        break

    past_length += 1
    if step >= len(input_ids) - 1:
        generated.append(next_token)
        all_tokens.append(next_token)

    if len(generated) > 0 and generated[-1] == tokenizer.eos_token_id:
        break

print(f"\nGenerated: '{tokenizer.decode(generated, skip_special_tokens=True)}'")
print("Done!")
