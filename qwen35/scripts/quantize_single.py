"""Quantize Qwen3.5 single-IR model using nncf.compress_weights().

Applies INT4 weight compression to the single stateful IR (openvino_model.xml).
Optionally quantizes embed_tokens.npy to INT8 per-row symmetric.

Usage (root venv):
    uv run python -m qwen35.scripts.quantize_single
    uv run python -m qwen35.scripts.quantize_single --model-dir models/qwen35/Qwen3.5-0.8B-paro-ov --mode int4_sym
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import nncf
import numpy as np
import openvino as ov

MODE_MAP = {
    "int4_sym": nncf.CompressWeightsMode.INT4_SYM,
    "int4_asym": nncf.CompressWeightsMode.INT4_ASYM,
    "int8_sym": nncf.CompressWeightsMode.INT8_SYM,
    "int8_asym": nncf.CompressWeightsMode.INT8_ASYM,
}


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1e9:.2f} GB"
    elif size_bytes >= 1_000_000:
        return f"{size_bytes / 1e6:.1f} MB"
    else:
        return f"{size_bytes / 1e3:.1f} KB"


def find_paro_rotation_ops(model, group_size=128):
    """Find PARO rotation matrix/scale ops to protect from quantization."""
    names = []
    for op in model.get_ops():
        op_type = op.get_type_name()
        if op_type not in ("MatMul", "Multiply"):
            continue
        for input_idx in range(op.get_input_size()):
            source = op.input(input_idx).get_source_output().get_node()
            while source.get_type_name() == "Convert":
                source = source.input(0).get_source_output().get_node()
            if source.get_type_name() != "Constant":
                continue
            shape = list(source.get_output_shape(0))
            if (len(shape) == 3
                and shape[1] == group_size
                and shape[2] == group_size
                and shape[0] >= 1):
                names.append(op.get_friendly_name())
                break
            if (op_type == "Multiply"
                and len(shape) == 2
                and shape[0] == 1
                and shape[1] >= group_size
                and shape[1] % group_size == 0):
                names.append(op.get_friendly_name())
                break
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3.5 single-IR model with nncf.compress_weights()"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/qwen35/Qwen3.5-0.8B-paro-ov"),
        help="Source model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--mode",
        default="int4_sym",
        choices=["int4_sym", "int4_asym", "int8_sym", "int8_asym"],
        help="Quantization mode (default: %(default)s)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="NNCF group_size for INT4 (default: %(default)s)",
    )
    args = parser.parse_args()

    src_dir = args.model_dir.resolve()
    if not src_dir.exists():
        print(f"ERROR: Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    src_xml = src_dir / "openvino_model.xml"
    if not src_xml.exists():
        print(f"ERROR: {src_xml} not found", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is not None:
        dst_dir = args.output_dir.resolve()
    else:
        dst_dir = src_dir.parent / f"{src_dir.name}-{args.mode.replace('_', '')}"

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source:  {src_dir}")
    print(f"Output:  {dst_dir}")
    print(f"Mode:    {args.mode}")
    print(f"Group:   {args.group_size}")
    print()

    core = ov.Core()

    # Compress main model
    print("Loading model...", flush=True)
    model = core.read_model(str(src_xml))

    orig_bin = src_dir / "openvino_model.bin"
    orig_size = orig_bin.stat().st_size if orig_bin.exists() else 0

    kwargs = {"mode": MODE_MAP[args.mode]}
    if "int4" in args.mode:
        kwargs["group_size"] = args.group_size

    rotation_ops = find_paro_rotation_ops(model)
    if rotation_ops:
        kwargs["ignored_scope"] = nncf.IgnoredScope(names=rotation_ops)
        print(f"Protecting {len(rotation_ops)} PARO rotation ops from quantization")

    print(f"Compressing ({args.mode})...", flush=True)
    t0 = time.time()
    compressed = nncf.compress_weights(model, **kwargs)
    elapsed = time.time() - t0

    dst_xml = dst_dir / "openvino_model.xml"
    ov.save_model(compressed, str(dst_xml))

    comp_bin = dst_dir / "openvino_model.bin"
    comp_size = comp_bin.stat().st_size if comp_bin.exists() else 0
    ratio = orig_size / comp_size if comp_size > 0 else 0
    print(f"Model: {format_size(orig_size)} -> {format_size(comp_size)} ({ratio:.1f}x, {elapsed:.1f}s)")

    # Copy auxiliary files
    print("\nCopying auxiliary files...")
    for src_file in sorted(src_dir.iterdir()):
        if src_file.is_dir():
            continue
        if src_file.suffix in (".xml", ".bin") and src_file.stem == "openvino_model":
            continue

        # Quantize embed_tokens.npy to INT8
        if src_file.name == "embed_tokens.npy":
            print(f"  Quantizing {src_file.name} to INT8...", end="", flush=True)
            embed_fp16 = np.load(str(src_file))
            embed_fp32 = embed_fp16.astype(np.float32)
            scales = np.max(np.abs(embed_fp32), axis=1, keepdims=True) / 127.0
            scales = np.where(scales == 0, 1.0, scales)
            embed_int8 = np.round(embed_fp32 / scales).astype(np.int8)
            scales_fp16 = scales.squeeze().astype(np.float16)
            np.save(str(dst_dir / "embed_tokens_int8.npy"), embed_int8)
            np.save(str(dst_dir / "embed_tokens_scales.npy"), scales_fp16)
            int8_size = (dst_dir / "embed_tokens_int8.npy").stat().st_size
            scales_size = (dst_dir / "embed_tokens_scales.npy").stat().st_size
            ratio = src_file.stat().st_size / (int8_size + scales_size)
            print(f" {format_size(src_file.stat().st_size)} -> {format_size(int8_size + scales_size)} ({ratio:.1f}x)")
            continue

        # Skip FP16 embed if INT8 versions already exist in source
        if src_file.name in ("embed_tokens_int8.npy", "embed_tokens_scales.npy"):
            shutil.copy2(str(src_file), str(dst_dir / src_file.name))
            print(f"  Copied {src_file.name} ({format_size(src_file.stat().st_size)})")
            continue

        shutil.copy2(str(src_file), str(dst_dir / src_file.name))
        print(f"  Copied {src_file.name} ({format_size(src_file.stat().st_size)})")

    print(f"\nDone. Output: {dst_dir}")


if __name__ == "__main__":
    main()
