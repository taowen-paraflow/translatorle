#!/usr/bin/env python3
"""Export Qwen3.5-0.8B without stateful transformation (for LowLatency2 testing)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from qwen35.export import export_model
from qwen35.config import Qwen35Config

cfg = Qwen35Config()
hf_model = "Qwen/Qwen3.5-0.8B"
out_dir = Path("models/qwen35/Qwen3.5-0.8B-nonstateful")

# We'll modify export to skip the stateful step
import qwen35.export as exp

# Monkey-patch to skip stateful
orig_patch = exp.patch_stateful_hybrid_ssm if hasattr(exp, 'patch_stateful_hybrid_ssm') else None
from qwen35.stateful import patch_stateful_hybrid_ssm as _orig_stateful

def noop_stateful(model):
    print("SKIPPING stateful transformation (for LL2 test)")
    return

# Patch it
import qwen35.stateful
qwen35.stateful.patch_stateful_hybrid_ssm = noop_stateful
exp.patch_stateful_hybrid_ssm = noop_stateful

export_model(hf_model, out_dir)
print(f"Non-stateful model saved to {out_dir}")
