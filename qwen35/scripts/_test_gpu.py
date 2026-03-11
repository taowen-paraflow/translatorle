#!/usr/bin/env python3
"""Test GPU inference for Qwen3.5."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen35.inference import Qwen35OVModel

def test(device, path, prompt, max_new=30):
    print("Loading %s from %s..." % (device, path))
    model = Qwen35OVModel.from_pretrained(path, device=device)
    print("Loaded. token_by_token=%s" % model._token_by_token)

    messages = [{"role": "user", "content": prompt}]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = model.tokenizer(text, return_tensors="pt")
    n_in = inputs["input_ids"].shape[1]

    model.reset()
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.time() - t0

    generated = out[0][n_in:]
    n_out = len(generated)
    tok_s = n_out / elapsed if elapsed > 0 else 0
    result = model.tokenizer.decode(generated, skip_special_tokens=True)
    print("%s: %d tok in %.1fs = %.1f tok/s" % (device, n_out, elapsed, tok_s))
    print("Output: %s" % result)
    print()

if __name__ == "__main__":
    prompt = "What is 2+3?"
    test("GPU", "models/qwen35/Qwen3.5-0.8B-ov", prompt, 30)
    test("CPU", "models/qwen35/Qwen3.5-0.8B-ov", prompt, 30)
    test("NPU", "models/qwen35/Qwen3.5-0.8B-npu", prompt, 30)
