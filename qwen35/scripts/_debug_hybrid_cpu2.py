"""Debug: run hybrid model on CPU with fixed sorting."""
import sys
sys.path.insert(0, ".")

from qwen35.inference import Qwen35OVModel

# Load on CPU to verify correctness without NPU FP16 issues
model = Qwen35OVModel.from_pretrained(
    "models/qwen35/Qwen3.5-0.8B-hybrid",
    device="CPU",
)
print(f"is_hybrid: {model._is_hybrid}")
print(f"is_static_cache: {model._is_static_cache}")
print(f"recurrent_inputs order: {model._explicit_recurrent_inputs[:5]}")

import torch
prompt = "Hello"
inputs = model.tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nCPU Output: {text}")
