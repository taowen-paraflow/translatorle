"""Quantize Qwen3.5 FP16 OpenVINO IR to INT4_ASYM using nncf.compress_weights().

Usage (root venv):
    uv run python -m qwen35.scripts.quantize_int4 [--model-size 0.8B|4B]
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import nncf
import openvino as ov

_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.config import MODELS_DIR, ARCH_CONFIGS

# Files to copy verbatim (not the .bin/.xml which get regenerated)
COPY_FILES = [
    "config.json",
    "embed_tokens.npy",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
]


def main():
    parser = argparse.ArgumentParser(description="Quantize Qwen3.5 FP16 IR to INT4_ASYM")
    parser.add_argument(
        "--model-size",
        default="0.8B",
        choices=list(ARCH_CONFIGS.keys()),
        help="Model size (default: %(default)s)",
    )
    args = parser.parse_args()

    arch = ARCH_CONFIGS[args.model_size]
    src_dir = MODELS_DIR / arch["ov_dir_name"]
    dst_dir = MODELS_DIR / arch["ov_dir_name"].replace("-ov", "-ov-int4")

    src_xml = src_dir / "openvino_model.xml"
    if not src_xml.exists():
        raise FileNotFoundError(f"Source model not found: {src_xml}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load FP16 model
    print(f"Loading FP16 model from {src_xml} ...")
    core = ov.Core()
    model = core.read_model(str(src_xml))
    print(f"  Model loaded: {len(model.inputs)} inputs, {len(model.outputs)} outputs")

    # 2. Compress weights to INT4_ASYM
    print("Compressing weights to INT4_ASYM (group_size=128) ...")
    t0 = time.time()
    compressed = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        group_size=128,
    )
    elapsed = time.time() - t0
    print(f"  Compression done in {elapsed:.1f}s")

    # 3. Save quantized model
    dst_xml = dst_dir / "openvino_model.xml"
    print(f"Saving quantized model to {dst_xml} ...")
    ov.save_model(compressed, str(dst_xml))
    print("  Saved.")

    # 4. Copy auxiliary files
    print("Copying auxiliary files ...")
    for fname in COPY_FILES:
        src = src_dir / fname
        dst = dst_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Copied {fname}")
        else:
            print(f"  SKIP (not found): {fname}")

    # 5. Report sizes
    print("\n=== Output files ===")
    total = 0
    for f in sorted(dst_dir.iterdir()):
        sz = f.stat().st_size
        total += sz
        if sz > 1_000_000:
            print(f"  {f.name:30s}  {sz / 1e9:.2f} GB")
        else:
            print(f"  {f.name:30s}  {sz / 1e3:.1f} KB")
    print(f"  {'TOTAL':30s}  {total / 1e9:.2f} GB")
    print(f"\nDone in {elapsed:.1f}s. Output: {dst_dir}")


if __name__ == "__main__":
    main()
