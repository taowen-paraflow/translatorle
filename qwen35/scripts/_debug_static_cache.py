"""Debug static-cache inference step by step."""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen35.inference import Qwen35OVModel

def debug_model(model_path, device, label):
    print(f"\n{'='*60}")
    print(f"Model: {label} ({model_path})")
    print(f"{'='*60}")
    model = Qwen35OVModel.from_pretrained(model_path, device=device)

    prompt = "The capital of France is"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"  = {[model.tokenizer.decode([t]) for t in input_ids[0].tolist()]}")
    print(f"Static cache: {model._is_static_cache}")
    print(f"Token by token: {model._token_by_token}")

    # Generate step by step
    all_tokens = input_ids[0].tolist()
    for step in range(10):
        if step == 0:
            ids = input_ids
        else:
            ids = torch.tensor([[all_tokens[-1]]], dtype=torch.long)

        from qwen35.inference import Qwen35CacheState
        cache = None if step == 0 else Qwen35CacheState()

        output = model.forward(input_ids=ids, cache_params=cache)
        logits = output.logits[0, -1]  # [vocab_size]
        top5 = torch.topk(logits, 5)
        next_token = top5.indices[0].item()
        all_tokens.append(next_token)

        decoded = model.tokenizer.decode([next_token])
        top5_decoded = [(model.tokenizer.decode([t.item()]), f"{v.item():.2f}") for t, v in zip(top5.indices, top5.values)]
        print(f"Step {step}: next={decoded!r:10s}  logit={top5.values[0].item():.2f}  top5={top5_decoded}")

    result = model.tokenizer.decode(all_tokens, skip_special_tokens=True)
    print(f"\nFull output: {result}")

# Test both models
debug_model("models/qwen35/Qwen3.5-0.8B-ov", "CPU", "Standard (stateful)")
debug_model("models/qwen35/Qwen3.5-0.8B-npu", "CPU", "NPU static-cache")
