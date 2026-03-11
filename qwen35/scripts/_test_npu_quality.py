#!/usr/bin/env python3
"""Test NPU output quality with various prompts."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen35.inference import Qwen35OVModel

def test_prompt(model, prompt, max_new=30, use_chat=False):
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

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
    mode = "chat" if use_chat else "raw"
    print(f"[{mode}] ({n_out}tok {tok_s:.1f}t/s) {prompt!r}")
    print(f"  -> {result}")
    print()

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "NPU"
    if device == "NPU":
        path = "models/qwen35/Qwen3.5-0.8B-npu"
    else:
        path = "models/qwen35/Qwen3.5-0.8B-ov"

    print(f"Loading {device} model from {path}...")
    model = Qwen35OVModel.from_pretrained(path, device=device)
    print(f"Model loaded. token_by_token={model._token_by_token}, "
          f"static_cache={model._is_static_cache}, "
          f"explicit_gdn={model._has_explicit_gdn_states}")
    print()

    # Raw completion
    test_prompt(model, "The capital of France is", max_new=20)
    test_prompt(model, "1+1=", max_new=10)
    test_prompt(model, "def fibonacci(n):", max_new=30)

    # Chat mode
    test_prompt(model, "What is 2+3?", max_new=20, use_chat=True)
    test_prompt(model, "Say hello in French", max_new=20, use_chat=True)
