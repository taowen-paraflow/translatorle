"""Extract model.norm / mtp.norm weight ratio for C++ MTP head correction.

When using the head block (model.norm + lm_head) for MTP draft tokens,
the MTP output already has mtp.norm applied. To get correct logits, we
multiply the MTP output by (model_norm_weight / mtp_norm_weight) before
feeding it to the head block, which cancels the mtp.norm and replaces
it with model.norm.

Run with qwen35 venv:
  powershell.exe -Command 'cd C:\Apps\translatorle; uv run --project qwen35 python -m qwen35.scripts.extract_norm_correction'
"""
import glob
import os
import sys

import numpy as np
from safetensors import safe_open
from huggingface_hub import snapshot_download


def main():
    model_id = "Qwen/Qwen3.5-0.8B"
    print(f"Loading weights from {model_id}")
    model_path = snapshot_download(model_id)

    model_norm_w = None
    mtp_norm_w = None
    for sf_path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(sf_path, framework="pt") as f:
            for key in f.keys():
                if key in ("model.norm.weight", "model.language_model.norm.weight"):
                    model_norm_w = f.get_tensor(key).float().numpy()
                if key == "mtp.norm.weight":
                    mtp_norm_w = f.get_tensor(key).float().numpy()

    assert model_norm_w is not None, "model.norm.weight not found"
    assert mtp_norm_w is not None, "mtp.norm.weight not found"

    print(f"model.norm.weight: shape={model_norm_w.shape}, "
          f"mean={float(model_norm_w.mean()):.4f}")
    print(f"mtp.norm.weight: shape={mtp_norm_w.shape}, "
          f"mean={float(mtp_norm_w.mean()):.4f}")

    ratio = model_norm_w / mtp_norm_w
    print(f"ratio: range=[{float(ratio.min()):.4f}, {float(ratio.max()):.4f}], "
          f"mean={float(ratio.mean()):.4f}")

    # Save to all hybrid model dirs
    dirs = [
        "models/qwen35/Qwen3.5-0.8B-hybrid",
        "models/qwen35/Qwen3.5-0.8B-hybrid-attn-int4sym-gdn-int8sym-head-int4sym",
    ]
    for d in dirs:
        if os.path.isdir(d):
            out = os.path.join(d, "mtp_norm_correction.npy")
            np.save(out, ratio.astype(np.float32))
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()
