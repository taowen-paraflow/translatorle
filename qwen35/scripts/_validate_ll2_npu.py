#!/usr/bin/env python3
"""Validate the LowLatency2 model on NPU.

The LL2 model has 0 Loops, 84 ReadValue/Assign (fully stateful).
This is the target deployment path.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov

print("OpenVINO version:", ov.__version__)
core = ov.Core()
print("Available devices:", core.available_devices)

MODEL_DIR = "models/qwen35/Qwen3.5-0.8B-ll2"

# Load model
model = core.read_model(f"{MODEL_DIR}/openvino_model.xml")
ops = model.get_ordered_ops()
loops = sum(1 for op in ops if op.get_type_name() in ("Loop", "TensorIterator"))
rvs = sum(1 for op in ops if op.get_type_name() == "ReadValue")
print(f"Model: Loops={loops}, ReadValue={rvs}")

print(f"\nInputs:")
for inp in model.inputs:
    print(f"  {inp.get_any_name()}: {inp.get_partial_shape()}")
print(f"Outputs:")
for out in model.outputs:
    print(f"  {out.get_any_name()}: {out.get_partial_shape()}")

# Try NPU compilation
device = "NPU"
if device not in core.available_devices:
    print(f"ERROR: {device} not available")
    sys.exit(1)

print(f"\nCompiling on {device}...")
config = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_MAX_PROMPT_LEN": "1",
    "NPUW_FOLD": "NO",
}

t0 = time.time()
try:
    compiled = core.compile_model(model, device, config)
    print(f"Compilation OK in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"Compilation FAILED in {time.time()-t0:.1f}s: {e}")

    # Try without NPUW_LLM
    print("\nRetrying without NPUW_LLM...")
    config2 = {
        "NPU_USE_NPUW": "YES",
        "NPUW_FOLD": "NO",
    }
    try:
        compiled = core.compile_model(model, device, config2)
        print(f"Compilation (no LLM) OK in {time.time()-t0:.1f}s")
    except Exception as e2:
        print(f"Compilation (no LLM) also FAILED: {e2}")

        # Try direct NPU (no NPUW)
        print("\nRetrying direct NPU...")
        try:
            compiled = core.compile_model(model, device, {})
            print(f"Direct NPU compilation OK")
        except Exception as e3:
            print(f"Direct NPU also FAILED: {e3}")
            sys.exit(1)

# Quick inference test
print("\n=== Quick inference test ===")
from transformers import AutoTokenizer

# Use the original model dir for tokenizer since LL2 might not have copied all files
orig_dir = "models/qwen35/Qwen3.5-0.8B-ov"
tokenizer = AutoTokenizer.from_pretrained(orig_dir, trust_remote_code=True)
embed_table = np.load(f"{orig_dir}/embed_tokens.npy").astype(np.float32)

request = compiled.create_infer_request()

prompt = "Hello, what is your name?"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"][0]
print(f"Prompt: '{prompt}' -> {len(input_ids)} tokens")

generated = []
past_length = 0
all_tokens = list(input_ids)

t0 = time.time()
for step in range(len(input_ids) + 10):
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
        print(f"  Step {step}: '{tokenizer.decode([token_id])}' -> '{text}' NaN={has_nan}")

    if has_nan:
        print("ERROR: NaN!")
        break

    past_length += 1
    if step >= len(input_ids) - 1:
        generated.append(next_token)
        all_tokens.append(next_token)

    if len(generated) > 0 and generated[-1] == tokenizer.eos_token_id:
        break

elapsed = time.time() - t0
decode_tokens = len(generated)
print(f"\nGenerated: '{tokenizer.decode(generated, skip_special_tokens=True)}'")
if decode_tokens > 0:
    print(f"Speed: {decode_tokens/elapsed:.1f} tok/s (total {elapsed:.1f}s)")
print("Done!")
