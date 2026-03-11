import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
logging.basicConfig(level=logging.WARNING)

import numpy as np, torch
from qwen35.inference import Qwen35OVModel, Qwen35CacheState

# Test static-cache model
print("Loading static-cache model...", flush=True)
m = Qwen35OVModel.from_pretrained("models/qwen35/Qwen3.5-0.8B-npu", device="CPU")
print(f"is_static_cache={m._is_static_cache}, max_cache_len={getattr(m, '_max_cache_len', 'N/A')}", flush=True)

prompt = "The capital of France is"
toks = m.tokenizer(prompt, return_tensors="pt")["input_ids"]
print(f"Tokens: {toks[0].tolist()}", flush=True)

# Prefill
out = m.forward(input_ids=toks)
logits = out.logits[0, -1]
t1 = torch.argmax(logits).item()
print(f"After prefill: past_length={m._past_length}, next_token={t1} = {m.tokenizer.decode([t1])!r}", flush=True)

# Decode 5 tokens
for step in range(5):
    out = m.forward(input_ids=torch.tensor([[t1]]), cache_params=Qwen35CacheState())
    logits = out.logits[0, -1]
    t1 = torch.argmax(logits).item()
    print(f"  Decode {step}: past_length={m._past_length}, next_token={t1} = {m.tokenizer.decode([t1])!r}", flush=True)

print("Done.", flush=True)
