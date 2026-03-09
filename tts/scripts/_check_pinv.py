import numpy as np
from pathlib import Path
p = Path(__file__).resolve().parent.parent.parent / "models" / "tts" / "talker_lm_head_pinv.npy"
x = np.load(str(p))
print(f"Shape: {x.shape}, dtype: {x.dtype}")
