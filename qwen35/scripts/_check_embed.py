"""Check embedding table and model weights for NaN."""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from qwen35.config import NPUW_MODEL_DIR

# Check embeddings
e = np.load(str(NPUW_MODEL_DIR / "embed_tokens.npy"))
print(f"embed dtype: {e.dtype}, shape: {e.shape}")
print(f"  has NaN: {np.any(np.isnan(e))}")
print(f"  has Inf: {np.any(np.isinf(e))}")
e32 = e.astype(np.float32)
print(f"  range: [{np.nanmin(e32):.4f}, {np.nanmax(e32):.4f}]")
print(f"  embed[9419] sum: {e32[9419].sum():.4f}")

# Quick model inference test with manual forward
import openvino as ov
core = ov.Core()
model = core.read_model(str(NPUW_MODEL_DIR / "openvino_model.xml"))

# Check model constants for NaN
print("\nChecking model weights for NaN...")
for op in model.get_ordered_ops():
    if op.get_type_name() == "Constant":
        data = op.get_data()
        if np.any(np.isnan(data)):
            print(f"  NaN in constant: {op.get_friendly_name()}, shape={data.shape}")
        if np.any(np.isinf(data)):
            print(f"  Inf in constant: {op.get_friendly_name()}, shape={data.shape}")
print("Weight check done.")
