#!/usr/bin/env python3
"""Prepare Qwen3.5-VL models for OpenVINO deployment.

Exports the vision encoder, copies the text decoder IR (which already accepts
``inputs_embeds``), and extracts the embedding table.

Output layout:
    models/qwen35/Qwen3.5-0.8B-vl/
    +-- vision_encoder.xml / .bin       Vision encoder (ViT)
    +-- openvino_model.xml / .bin       Decoder (inputs_embeds, shared with text-only)
    +-- embed_tokens.npy                Embedding table [vocab_size, hidden_size]
    +-- config.json
    +-- tokenizer.json
    +-- tokenizer_config.json

Usage:
    uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_vl_models
    uv run --project qwen35 python -m qwen35.scripts.prepare_qwen35_vl_models --hf-model Qwen/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure parent package is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.config import MODELS_DIR


def _free_memory():
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Qwen3.5-VL models for OpenVINO deployment",
    )
    parser.add_argument(
        "--hf-model",
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: models/qwen35/<model_name>-vl)",
    )
    args = parser.parse_args()

    model_id = args.hf_model
    model_name = Path(model_id).name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else MODELS_DIR / f"{model_name}-vl"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  Qwen3.5-VL Model Preparation for OpenVINO")
    print("=" * 60)
    print(f"  Model ID:     {model_id}")
    print(f"  Output dir:   {output_dir}")
    print("=" * 60)
    print()

    total_start = time.perf_counter()

    # ================================================================
    # Step 1: Load the full VL model
    # ================================================================
    print("=" * 60)
    print("Step 1: Loading Qwen3.5 VL model")
    print("=" * 60)
    t0 = time.perf_counter()

    import torch

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    model = None
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float32, trust_remote_code=True,
        )
    except (ImportError, ValueError, KeyError):
        pass

    if model is None:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_id, dtype=torch.float32, trust_remote_code=True,
        )

    model.eval()
    print(f"  Model class: {type(model).__name__}")

    # Auto-detect hidden_size from config
    config = model.config
    text_config = (
        config.text_config if hasattr(config, "text_config") else config
    )
    hidden_size = text_config.hidden_size
    print(f"  hidden_size: {hidden_size}")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")
    print()

    # ================================================================
    # Step 2: Export vision encoder
    # ================================================================
    print("=" * 60)
    print("Step 2: Exporting vision encoder")
    print("=" * 60)
    t0 = time.perf_counter()

    from qwen35.export_vl import export_vision_encoder

    export_vision_encoder(model, output_dir)

    print(f"  Time: {time.perf_counter() - t0:.1f}s")
    print()

    # ================================================================
    # Step 3: Copy text decoder IR (already has inputs_embeds)
    # ================================================================
    print("=" * 60)
    print("Step 3: Copying text decoder IR")
    print("=" * 60)
    t0 = time.perf_counter()

    # The text decoder IR already accepts inputs_embeds (no surgery needed).
    # Both text-only and VL share the same decoder IR.
    text_ov_dir = MODELS_DIR / f"{model_name}-ov"

    src_xml = text_ov_dir / "openvino_model.xml"
    src_bin = text_ov_dir / "openvino_model.bin"
    dst_xml = output_dir / "openvino_model.xml"
    dst_bin = output_dir / "openvino_model.bin"

    if src_xml.exists() and src_bin.exists():
        print(f"  Found existing text-only OV model at: {text_ov_dir}")
        shutil.copy2(str(src_xml), str(dst_xml))
        shutil.copy2(str(src_bin), str(dst_bin))
        # Copy config / tokenizer files alongside the IR
        for f in text_ov_dir.iterdir():
            if f.is_file() and not f.name.startswith("openvino_model"):
                shutil.copy2(str(f), str(output_dir / f.name))
        print(f"  Copied decoder IR to: {output_dir}")
    else:
        print(f"  No existing text-only OV model at: {text_ov_dir}")
        print(f"  Re-exporting text decoder from {model_id} ...")

        # Resolve HF model to a local directory
        hf_model_dir = Path(model_id)
        if not hf_model_dir.is_dir():
            from huggingface_hub import snapshot_download

            hf_model_dir = Path(snapshot_download(model_id))

        from qwen35.export import export_model

        export_model(
            model_dir=str(hf_model_dir),
            output_dir=str(output_dir),
            compress_to_fp16=True,
        )

    print(f"  Time: {time.perf_counter() - t0:.1f}s")
    print()

    # ================================================================
    # Step 4: Extract or copy embed_tokens.npy
    # ================================================================
    print("=" * 60)
    print("Step 4: Preparing embed_tokens.npy")
    print("=" * 60)
    t0 = time.perf_counter()

    embed_path = output_dir / "embed_tokens.npy"
    src_embed = text_ov_dir / "embed_tokens.npy"
    if src_embed.exists():
        shutil.copy2(str(src_embed), str(embed_path))
        print(f"  Copied embed_tokens.npy from text-only model")
    else:
        from qwen35.export_vl import extract_embed_tokens
        extract_embed_tokens(model, embed_path)

    print(f"  Time: {time.perf_counter() - t0:.1f}s")
    print()

    # Free VL model memory (no longer needed)
    del model
    _free_memory()
    print("  [Freed VL model memory]")
    print()

    # ================================================================
    # Step 5: Copy tokenizer and config files
    # ================================================================
    print("=" * 60)
    print("Step 5: Copying tokenizer and config files")
    print("=" * 60)
    t0 = time.perf_counter()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"  Saved tokenizer to: {output_dir}")

    # Copy config.json if not already present
    config_dst = output_dir / "config.json"
    if not config_dst.exists():
        local_model = Path(model_id)
        if local_model.is_dir() and (local_model / "config.json").exists():
            shutil.copy2(str(local_model / "config.json"), str(config_dst))
            print("  Copied config.json from local model")
        else:
            try:
                from huggingface_hub import hf_hub_download

                downloaded = hf_hub_download(
                    repo_id=model_id, filename="config.json"
                )
                shutil.copy2(downloaded, str(config_dst))
                print("  Downloaded config.json from HuggingFace")
            except Exception as e:
                print(f"  WARNING: Could not get config.json: {e}")

    # Also try to copy preprocessor_config.json (needed for image processing)
    preproc_dst = output_dir / "preprocessor_config.json"
    if not preproc_dst.exists():
        local_model = Path(model_id)
        if local_model.is_dir() and (local_model / "preprocessor_config.json").exists():
            shutil.copy2(
                str(local_model / "preprocessor_config.json"), str(preproc_dst)
            )
            print("  Copied preprocessor_config.json from local model")
        else:
            try:
                from huggingface_hub import hf_hub_download

                downloaded = hf_hub_download(
                    repo_id=model_id, filename="preprocessor_config.json"
                )
                shutil.copy2(downloaded, str(preproc_dst))
                print("  Downloaded preprocessor_config.json from HuggingFace")
            except Exception:
                # Not all models have this file; it is optional
                pass

    print(f"  Time: {time.perf_counter() - t0:.1f}s")
    print()

    # ================================================================
    # Summary
    # ================================================================
    total_elapsed = time.perf_counter() - total_start
    print("=" * 60)
    print(f"  All done! Total time: {total_elapsed:.1f}s")
    print("=" * 60)
    print()
    print("Output files:")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            rel = item.relative_to(output_dir)
            size_mb = item.stat().st_size / 1024 / 1024
            print(f"  {rel} ({size_mb:.1f} MB)")
    print()


if __name__ == "__main__":
    main()
