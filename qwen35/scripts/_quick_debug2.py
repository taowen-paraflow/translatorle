import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
logging.basicConfig(level=logging.WARNING)

import numpy as np, torch
from qwen35.inference import Qwen35OVModel, Qwen35CacheState

prompt = "The capital of France is"

for label, path in [("Standard", "models/qwen35/Qwen3.5-0.8B-ov"), ("Static", "models/qwen35/Qwen3.5-0.8B-npu")]:
    print(f"\n--- {label} ---", flush=True)
    m = Qwen35OVModel.from_pretrained(path, device="CPU")
    toks = m.tokenizer(prompt, return_tensors="pt")["input_ids"]

    # Prefill
    out = m.forward(input_ids=toks)
    logits = out.logits[0, -1]
    top5 = torch.topk(logits, 5)
    print(f"  Prefill top5: {[(m.tokenizer.decode([t.item()]), f'{v.item():.3f}') for t, v in zip(top5.indices, top5.values)]}", flush=True)

    # Decode 3 steps
    t1 = top5.indices[0].item()
    for step in range(3):
        out = m.forward(input_ids=torch.tensor([[t1]]), cache_params=Qwen35CacheState())
        logits = out.logits[0, -1]
        top5 = torch.topk(logits, 5)
        print(f"  Step {step} (input={m.tokenizer.decode([t1])!r:10s}) top5: {[(m.tokenizer.decode([t.item()]), f'{v.item():.3f}') for t, v in zip(top5.indices, top5.values)]}", flush=True)
        t1 = top5.indices[0].item()

    del m

print("\nDone.", flush=True)
