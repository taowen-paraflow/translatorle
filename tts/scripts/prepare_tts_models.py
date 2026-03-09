#!/usr/bin/env python3
"""All-in-one Qwen3-TTS model preparation: download, export, surgery, quantize.

Usage:
    uv run python -m tts.scripts.prepare_tts_models [--hf-model PATH] [--skip-existing]
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

# Ensure parent package is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_TTS_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _TTS_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from tts.config import MODELS_DIR

log = logging.getLogger("prepare_tts")


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


def step15_copy_tokenizer(hf_model_dir: Path, models_dir: Path, skip_existing: bool) -> None:
    """Copy tokenizer and config files from HF model to models_dir for inference."""
    print("\n[Step 15/15] Copying tokenizer and config files...")
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "config.json",
        "generation_config.json",
        "added_tokens.json",
    ]
    copied = 0
    for fname in files_to_copy:
        src = hf_model_dir / fname
        dst = models_dir / fname
        if src.exists():
            if skip_existing and dst.exists():
                continue
            shutil.copy2(src, dst)
            copied += 1
    print(f"  Copied {copied} files to {models_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Qwen3-TTS models for OpenVINO inference"
    )
    parser.add_argument(
        "--hf-model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="HuggingFace model ID or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose output already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    hf_model_dir = resolve_hf_model(args.hf_model)

    t0 = time.time()

    # Steps 1-9: Talker
    print("\n" + "=" * 60)
    print("TALKER EXPORT (Steps 1-9)")
    print("=" * 60)
    from tts.scripts._export_talker import export_talker
    export_talker(hf_model_dir, MODELS_DIR, args.skip_existing)

    # Steps 10-13: Code Predictor
    print("\n" + "=" * 60)
    print("CODE PREDICTOR EXPORT (Steps 10-13)")
    print("=" * 60)
    from tts.scripts._export_code_predictor import export_code_predictor
    export_code_predictor(hf_model_dir, MODELS_DIR, args.skip_existing)

    # Step 14: Speech Decoder
    print("\n" + "=" * 60)
    print("SPEECH DECODER EXPORT (Step 14)")
    print("=" * 60)
    from tts.scripts._export_decoder import export_decoder
    export_decoder(hf_model_dir, MODELS_DIR, args.skip_existing)

    # Step 15: Copy tokenizer files
    print("\n" + "=" * 60)
    print("TOKENIZER FILES (Step 15)")
    print("=" * 60)
    step15_copy_tokenizer(hf_model_dir, MODELS_DIR, args.skip_existing)

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.1f}s. Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
