"""Quantize the ASR decoder model from FP16 to INT4 for faster NPU inference.

Loads the IR-surgery'd stateful decoder (with inputs_embeds) from
  models/decoder_stateful_embeds/openvino_model.xml
and compresses weights to INT4_SYM using NNCF, then saves the result to
  models/decoder_stateful_int4/openvino_model.xml

Also copies embed_tokens.npy and tokenizer/config files so the INT4 directory
is self-contained and ready for inference.

Usage (from WSL, targeting Windows Python):
    powershell.exe -Command '
        $env:PYTHONIOENCODING = "utf-8";
        cd C:\\Apps\\translatorle;
        C:\\Users\\taowen\\.local\\bin\\uv.exe run python asr/scripts/quantize_decoder.py
    '
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path


def main():
    # -- Resolve paths -------------------------------------------------------
    # asr/scripts/quantize_decoder.py -> asr/scripts -> asr -> translatorle
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"

    src_dir = models_dir / "decoder_stateful_embeds"
    dst_dir = models_dir / "decoder_stateful_int4"

    src_xml = src_dir / "openvino_model.xml"
    src_bin = src_dir / "openvino_model.bin"
    embed_npy = models_dir / "embed_tokens.npy"

    print()
    print("=" * 60)
    print("  ASR Decoder INT4 Quantization")
    print("=" * 60)
    print(f"  Source: {src_dir}")
    print(f"  Target: {dst_dir}")
    print("=" * 60)
    print()

    # -- Validate source files exist -----------------------------------------
    if not src_xml.exists():
        print(f"ERROR: Source model not found: {src_xml}")
        return
    if not src_bin.exists():
        print(f"ERROR: Source weights not found: {src_bin}")
        return

    src_bin_size = src_bin.stat().st_size
    print(f"Source .bin size: {src_bin_size / 1024 / 1024:.1f} MB")

    # -- Load the FP16 model -------------------------------------------------
    print()
    print("Step 1: Loading FP16 model...")
    t0 = time.perf_counter()

    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(src_xml))

    params = list(model.get_parameters())
    print(f"  Parameters ({len(params)}):")
    for p in params:
        print(f"    {p.get_friendly_name()}: {p.get_partial_shape()}")

    print(f"  Load time: {time.perf_counter() - t0:.1f}s")

    # -- Compress weights to INT4 --------------------------------------------
    print()
    print("Step 2: Compressing weights with NNCF (INT4_SYM, group_size=128)...")
    t0 = time.perf_counter()

    import nncf

    compressed_model = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        group_size=128,
        ratio=1.0,
    )

    print(f"  Compression time: {time.perf_counter() - t0:.1f}s")

    # -- Save compressed model -----------------------------------------------
    print()
    print("Step 3: Saving INT4 model...")
    t0 = time.perf_counter()

    os.makedirs(str(dst_dir), exist_ok=True)

    dst_xml = dst_dir / "openvino_model.xml"
    ov.save_model(compressed_model, str(dst_xml))

    dst_bin = dst_dir / "openvino_model.bin"
    dst_bin_size = dst_bin.stat().st_size
    print(f"  Saved: {dst_xml}")
    print(f"  Save time: {time.perf_counter() - t0:.1f}s")

    # -- Copy auxiliary files ------------------------------------------------
    print()
    print("Step 4: Copying auxiliary files...")

    # Copy all non-IR files from source dir (tokenizer, config, etc.)
    copied_files = []
    for fname in sorted(os.listdir(str(src_dir))):
        if fname.startswith("openvino_model"):
            continue  # Skip IR files (we wrote our own)
        src_file = src_dir / fname
        dst_file = dst_dir / fname
        if src_file.is_file():
            shutil.copy2(str(src_file), str(dst_file))
            copied_files.append(fname)
    print(f"  Copied {len(copied_files)} files: {', '.join(copied_files)}")

    # Copy embed_tokens.npy into the INT4 directory
    if embed_npy.exists():
        dst_embed = dst_dir / "embed_tokens.npy"
        shutil.copy2(str(embed_npy), str(dst_embed))
        embed_size = dst_embed.stat().st_size / 1024 / 1024
        print(f"  Copied embed_tokens.npy ({embed_size:.1f} MB)")
    else:
        print(f"  WARNING: embed_tokens.npy not found at {embed_npy}")

    # -- Summary -------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Before (FP16): {src_bin_size / 1024 / 1024:>8.1f} MB  {src_bin}")
    print(f"  After  (INT4): {dst_bin_size / 1024 / 1024:>8.1f} MB  {dst_bin}")

    ratio = dst_bin_size / src_bin_size
    reduction = (1 - ratio) * 100
    print(f"  Compression:   {reduction:.1f}% reduction ({ratio:.2f}x)")
    print()
    print("  Output files:")
    for f in sorted(os.listdir(str(dst_dir))):
        fpath = dst_dir / f
        if fpath.is_file():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"    {f}: {size_mb:.1f} MB")
    print()
    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
