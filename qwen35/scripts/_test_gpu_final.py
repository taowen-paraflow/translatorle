#!/usr/bin/env python3
"""Final GPU quality and speed test."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen35.inference import Qwen35OVModel

def bench(device, path, prompts, max_new=50):
    print("Loading %s..." % device)
    model = Qwen35OVModel.from_pretrained(path, device=device)

    for prompt in prompts:
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
        print("[%s %.1ft/s] %s" % (device, tok_s, prompt))
        print("  -> %s" % result[:200])
        print()

if __name__ == "__main__":
    prompts = [
        "Say hello in French",
        "The capital of France is",
        "Write a haiku about snow",
    ]
    bench("GPU", "models/qwen35/Qwen3.5-0.8B-ov", prompts)
    bench("CPU", "models/qwen35/Qwen3.5-0.8B-ov", prompts)
