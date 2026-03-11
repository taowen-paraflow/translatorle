#!/usr/bin/env python3
"""Quick NPU inference test with chat template."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen35.inference import Qwen35OVModel

def test(device, model_path):
    print(f"\n{'='*60}")
    print(f"Testing {device}")
    print(f"{'='*60}")
    model = Qwen35OVModel.from_pretrained(model_path, device=device)

    messages = [{"role": "user", "content": "请用中文简要介绍法国巴黎"}]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = model.tokenizer(text, return_tensors="pt")
    n_in = inputs["input_ids"].shape[1]
    print(f"Input tokens: {n_in}")

    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    elapsed = time.time() - t0

    generated = out[0][n_in:]
    n_out = len(generated)
    tok_s = n_out / elapsed if elapsed > 0 else 0
    result = model.tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Output ({n_out} tok in {elapsed:.1f}s = {tok_s:.1f} tok/s):")
    print(result)
    print()

if __name__ == "__main__":
    test("NPU", "models/qwen35/Qwen3.5-0.8B-npu")
    test("CPU", "models/qwen35/Qwen3.5-0.8B-ov")
