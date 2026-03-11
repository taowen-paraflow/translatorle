"""Compare hybrid model: CPU vs NPU output quality."""
import sys
sys.path.insert(0, ".")
import time
from qwen35.inference import Qwen35OVModel

prompts = [
    "The capital of France is",
    "Hello",
    "1+1=",
]

for device, ov_config in [("CPU", None), ("NPU", {"NPU_USE_NPUW": "YES", "NPUW_FOLD": "NO"})]:
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    model = Qwen35OVModel.from_pretrained(
        "models/qwen35/Qwen3.5-0.8B-hybrid",
        device=device,
        ov_config=ov_config,
    )
    print(f"  is_hybrid: {model._is_hybrid}")

    for prompt in prompts:
        inputs = model.tokenizer(prompt, return_tensors="pt")
        n_input = inputs["input_ids"].shape[1]
        t0 = time.time()
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        elapsed = time.time() - t0
        n_new = len(outputs[0]) - n_input
        text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  [{prompt!r}] -> {text!r}  ({n_new} tok in {elapsed:.1f}s = {n_new/elapsed:.1f} t/s)")

    del model
