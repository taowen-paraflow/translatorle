"""Test hybrid model on CPU to verify correctness (no NPU precision issues)."""
import sys
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

from qwen35.inference import Qwen35OVModel

# Load hybrid model on CPU
model = Qwen35OVModel.from_pretrained(
    "models/qwen35/Qwen3.5-0.8B-hybrid",
    device="CPU",
    ov_config={},
)
print(f"is_hybrid={model._is_hybrid}")
print(f"recurrent_inputs={model._explicit_recurrent_inputs[:3]}...")
print(f"intermediate_outputs={len(model._gdn_intermediate_outputs)} outputs")

inputs = model.tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"CPU output: {text}")
