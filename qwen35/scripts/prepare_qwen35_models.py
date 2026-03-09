#!/usr/bin/env python3
"""All-in-one Qwen3.5 model preparation: download + export to OpenVINO.

Usage:
    uv run python -m qwen35.scripts.prepare_qwen35_models [--hf-model ID_OR_PATH] [--skip-existing]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure parent package is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.config import MODELS_DIR

log = logging.getLogger("prepare_qwen35")


def resolve_hf_model(hf_model: str) -> Path:
    """Resolve HF model path: if it's a directory use it, otherwise download from Hub."""
    p = Path(hf_model)
    if p.is_dir():
        log.info("Using local model directory: %s", p)
        return p
    log.info("Downloading model from HuggingFace Hub: %s", hf_model)
    from huggingface_hub import snapshot_download
    local = snapshot_download(hf_model)
    return Path(local)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Qwen3.5 models for OpenVINO inference"
    )
    parser.add_argument(
        "--hf-model",
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip export if output already exists",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 compression (keep FP32)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1: Download model
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOAD MODEL")
    print("=" * 60)
    hf_model_dir = resolve_hf_model(args.hf_model)

    # Step 2: Export to OpenVINO
    model_name = Path(args.hf_model).name
    output_dir = MODELS_DIR / f"{model_name}-ov"

    if args.skip_existing and (output_dir / "openvino_model.xml").exists():
        print(f"\n[Step 2] Skipping export — {output_dir} already exists")
    else:
        print("\n" + "=" * 60)
        print("STEP 2: EXPORT TO OPENVINO")
        print("=" * 60)
        from qwen35.export import export_model
        export_model(
            model_dir=str(hf_model_dir),
            output_dir=str(output_dir),
            compress_to_fp16=not args.no_fp16,
        )

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.1f}s. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
