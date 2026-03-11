#!/usr/bin/env python3
"""Debug: compare first-token logits between CPU and NPU models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from qwen35.inference import Qwen35OVModel

def get_first_logits(device, path, token_ids):
    """Run a single forward pass and return logits."""
    model = Qwen35OVModel.from_pretrained(path, device=device)
    model.reset()

    import torch
    ids = torch.tensor([token_ids], dtype=torch.long)

    # Run forward
    output = model.forward(input_ids=ids)
    logits = output.logits[0, -1, :].numpy()  # last token position
    return logits, model

if __name__ == "__main__":
    # Simple token sequence: "Hello" -> token IDs
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "models/qwen35/Qwen3.5-0.8B-ov", trust_remote_code=True
    )
    token_ids = tok.encode("The capital of France is")
    print("Token IDs:", token_ids)
    print("Decoded:", [tok.decode([t]) for t in token_ids])

    # Get logits from both
    print("\nRunning CPU...")
    cpu_logits, cpu_model = get_first_logits(
        "CPU", "models/qwen35/Qwen3.5-0.8B-ov", token_ids
    )

    print("Running NPU...")
    npu_logits, npu_model = get_first_logits(
        "NPU", "models/qwen35/Qwen3.5-0.8B-npu", token_ids
    )

    # Compare
    print("\n=== Logits Comparison ===")
    cpu_top5 = np.argsort(cpu_logits)[-5:][::-1]
    npu_top5 = np.argsort(npu_logits)[-5:][::-1]

    print("CPU top-5 tokens:")
    for t in cpu_top5:
        print("  %d (%s): %.4f" % (t, repr(tok.decode([t])), cpu_logits[t]))

    print("NPU top-5 tokens:")
    for t in npu_top5:
        print("  %d (%s): %.4f" % (t, repr(tok.decode([t])), npu_logits[t]))

    # Numerical diff
    diff = np.abs(cpu_logits - npu_logits)
    print("\nMax abs diff: %.6f" % diff.max())
    print("Mean abs diff: %.6f" % diff.mean())
    print("CPU logits range: [%.4f, %.4f]" % (cpu_logits.min(), cpu_logits.max()))
    print("NPU logits range: [%.4f, %.4f]" % (npu_logits.min(), npu_logits.max()))

    # Check if argmax matches
    cpu_pred = cpu_logits.argmax()
    npu_pred = npu_logits.argmax()
    print("\nCPU prediction: %d (%s)" % (cpu_pred, repr(tok.decode([cpu_pred]))))
    print("NPU prediction: %d (%s)" % (npu_pred, repr(tok.decode([npu_pred]))))
