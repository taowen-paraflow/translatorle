"""Re-export HY-MT model with INT4_SYM quantization for NPU.

The original model was exported with optimum-intel's default load_in_4bit=True,
which uses INT4_ASYM (asymmetric quantization with zero points). Intel NPU does
NOT support INT4_ASYM efficiently, causing fallback to CPU and ~1.2 tok/s.

Since nncf.compress_weights() cannot re-compress already-quantized INT4 weights,
this script re-exports from the FP32 standalone Qwen3 checkpoint with explicit
INT4_SYM settings.

Steps:
  1. Load hy_mt_qwen3_standalone with optimum-intel, export=True
  2. Use quantization_config with INT4_SYM, group_size=128, ratio=1.0
  3. Save to models/hy_mt_int4sym/
  4. Convert tokenizer to OpenVINO format
  5. Print before/after .bin file sizes and verify no zero_point artifacts

Usage:
    uv run python hymt/scripts/requantize_mt.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    standalone_dir = project_root / "models" / "hy_mt_qwen3_standalone"
    src_dir = project_root / "models" / "hy_mt_ov"
    dst_dir = project_root / "models" / "hy_mt_int4sym"

    if not standalone_dir.exists():
        print(f"ERROR: Standalone checkpoint not found: {standalone_dir}")
        print("  Run prepare_mt_models.py first to create the Qwen3 checkpoint.")
        return

    print("=" * 60)
    print("  HY-MT Re-export: INT4_SYM for NPU")
    print("=" * 60)
    print(f"  Source (FP32): {standalone_dir}")
    print(f"  Target:        {dst_dir}")
    print()

    os.makedirs(str(dst_dir), exist_ok=True)

    # --- Step 1: Export with INT4_SYM ---
    print("[1/3] Exporting with INT4_SYM quantization...")
    print("  mode=INT4_SYM, group_size=128, ratio=1.0, sym=True")
    print("  This may take several minutes...")
    t0 = time.perf_counter()

    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig

    quant_config = OVWeightQuantizationConfig(
        bits=4,
        sym=True,
        group_size=128,
        ratio=1.0,
    )

    ov_model = OVModelForCausalLM.from_pretrained(
        str(standalone_dir),
        export=True,
        quantization_config=quant_config,
    )
    ov_model.save_pretrained(str(dst_dir))

    t_export = time.perf_counter() - t0
    print(f"  Exported in {t_export:.1f}s")
    print()

    # --- Step 2: Save HF tokenizer + convert to OV format ---
    print("[2/3] Converting tokenizer to OpenVINO format...")
    t0 = time.perf_counter()

    from transformers import AutoTokenizer
    from openvino_tokenizers import convert_tokenizer
    import openvino as ov

    tokenizer = AutoTokenizer.from_pretrained(str(standalone_dir))
    tokenizer.save_pretrained(str(dst_dir))

    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, str(dst_dir / "openvino_tokenizer.xml"))
    ov.save_model(ov_detokenizer, str(dst_dir / "openvino_detokenizer.xml"))

    t_tok = time.perf_counter() - t0
    print(f"  Done in {t_tok:.1f}s")
    print()

    # --- Step 3: Verify and report ---
    print("[3/3] Verification...")

    # List output files
    print("  Output files:")
    for f in sorted(os.listdir(str(dst_dir))):
        fpath = dst_dir / f
        if fpath.is_file():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"    {f}: {size_mb:.1f} MB")
    print()

    # Size comparison
    src_bin = src_dir / "openvino_model.bin"
    dst_bin = dst_dir / "openvino_model.bin"
    src_size = src_bin.stat().st_size / (1024 * 1024)
    dst_size = dst_bin.stat().st_size / (1024 * 1024)

    print("=" * 60)
    print("  Size Comparison")
    print("=" * 60)
    print(f"  Original (INT4_ASYM): {src_size:>10.1f} MB")
    print(f"  New      (INT4_SYM):  {dst_size:>10.1f} MB")
    diff = src_size - dst_size
    pct = (1 - dst_size / src_size) * 100 if src_size > 0 else 0
    print(f"  Reduction:            {diff:>10.1f} MB ({pct:.1f}%)")
    print()

    # Check for asymmetric artifacts
    print("  Checking for asymmetric artifacts in new model...")
    dst_xml = dst_dir / "openvino_model.xml"
    with open(str(dst_xml), "r", encoding="utf-8") as f:
        xml_content = f.read()
    zp_count = xml_content.count("zero_point")
    sub_count = xml_content.count("Subtract")
    print(f"  zero_point references: {zp_count}")
    print(f"  Subtract operations:   {sub_count}")
    if zp_count == 0 and sub_count == 0:
        print("  OK: Model is purely symmetric (INT4_SYM) -- no zero points!")
    else:
        print("  WARNING: Asymmetric artifacts still present!")
    print()

    total = t_export + t_tok
    print(f"  Total time: {total:.1f}s")
    print("  Done!")
    print()


if __name__ == "__main__":
    main()
