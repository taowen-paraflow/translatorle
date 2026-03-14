"""Quantize Qwen3.5 hybrid subgraph IRs using nncf.compress_weights().

Applies per-subgraph-type quantization to the 13 hybrid IR files:
  - 6 GDN blocks (GPU, contain Loop nodes for recurrence)
  - 6 Attention blocks (NPU, standard SDPA)
  - 1 Head block (GPU, RMSNorm + lm_head 248k vocab projection)

Non-quantized subgraphs (embed, tokenizer IRs) and auxiliary files
(embed_tokens.npy, config.json, tokenizer files) are copied verbatim.

Usage (root venv):
    uv run python -m qwen35.scripts.quantize_hybrid
    uv run python -m qwen35.scripts.quantize_hybrid --attn-mode int4_sym --gdn-mode int8_sym
    uv run python -m qwen35.scripts.quantize_hybrid --model-dir models/qwen35/Qwen3.5-0.8B-hybrid --output-dir models/qwen35/Qwen3.5-0.8B-hybrid-int4

Notes:
    - NPU requires INT4_SYM (not ASYM) -- ASYM falls back to slow path on NPU
    - GDN blocks contain Loop nodes with FP32 recurrence -- INT4 may affect quality
    - For <=1B models, INT4 on NPU may be slower (dequant overhead > bandwidth saving)
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import nncf
import numpy as np
import openvino as ov

# NNCF compression mode mapping
MODE_MAP = {
    "int4_sym": nncf.CompressWeightsMode.INT4_SYM,
    "int4_asym": nncf.CompressWeightsMode.INT4_ASYM,
    "int8_sym": nncf.CompressWeightsMode.INT8_SYM,
    "int8_asym": nncf.CompressWeightsMode.INT8_ASYM,
}

# Number of subgraphs per type
NUM_GDN_BLOCKS = 6
NUM_ATTN_BLOCKS = 6


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1e9:.2f} GB"
    elif size_bytes >= 1_000_000:
        return f"{size_bytes / 1e6:.1f} MB"
    else:
        return f"{size_bytes / 1e3:.1f} KB"


def get_ir_size(directory: Path, stem: str) -> int:
    """Get combined size of .xml + .bin for a given IR stem."""
    total = 0
    for ext in (".xml", ".bin"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            total += p.stat().st_size
    return total


def find_paro_rotation_ops(model, group_size=128):
    """Find MatMul/BatchMatMul ops with PARO rotation matrix constants.

    PARO rotation matrices appear as Constant inputs to MatMul ops with
    shape (num_groups, group_size, group_size) — e.g., (8, 128, 128).
    Also matches Multiply ops for channel_scales with shape (1, dim).

    Returns list of op friendly names to exclude from NNCF compression.
    """
    names = []
    for op in model.get_ops():
        op_type = op.get_type_name()
        if op_type not in ("MatMul", "Multiply"):
            continue
        for input_idx in range(op.get_input_size()):
            source = op.input(input_idx).get_source_output().get_node()
            # Walk through Convert nodes (inserted by FP16 compression)
            while source.get_type_name() == "Convert":
                source = source.input(0).get_source_output().get_node()
            if source.get_type_name() != "Constant":
                continue
            shape = list(source.get_output_shape(0))
            # Match rotation matrix: (num_groups, group_size, group_size)
            if (len(shape) == 3
                and shape[1] == group_size
                and shape[2] == group_size
                and shape[0] >= 1):
                names.append(op.get_friendly_name())
                break
            # Match channel_scales: (1, dim) where dim is multiple of group_size
            if (op_type == "Multiply"
                and len(shape) == 2
                and shape[0] == 1
                and shape[1] >= group_size
                and shape[1] % group_size == 0):
                names.append(op.get_friendly_name())
                break
    return names


def compress_ir(
    core: ov.Core,
    src_dir: Path,
    dst_dir: Path,
    stem: str,
    mode_str: str,
    group_size: int,
) -> tuple[int, int, float]:
    """Compress a single IR file and return (original_size, compressed_size, elapsed).

    Returns sizes in bytes and elapsed time in seconds.
    """
    src_xml = src_dir / f"{stem}.xml"
    dst_xml = dst_dir / f"{stem}.xml"

    orig_size = get_ir_size(src_dir, stem)

    if mode_str == "fp16":
        # Just copy
        for ext in (".xml", ".bin"):
            src = src_dir / f"{stem}{ext}"
            dst = dst_dir / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(str(src), str(dst))
        return orig_size, orig_size, 0.0

    # Load, compress, save
    model = core.read_model(str(src_xml))
    mode = MODE_MAP[mode_str]

    t0 = time.time()
    kwargs = {"mode": mode}
    if "int4" in mode_str:
        kwargs["group_size"] = group_size

    # Protect PARO rotation matrices from quantization
    rotation_ops = find_paro_rotation_ops(model)
    if rotation_ops:
        kwargs["ignored_scope"] = nncf.IgnoredScope(names=rotation_ops)
        print(f" (protecting {len(rotation_ops)} PARO rotation ops)", end="", flush=True)

    compressed = nncf.compress_weights(model, **kwargs)
    elapsed = time.time() - t0

    ov.save_model(compressed, str(dst_xml))
    comp_size = get_ir_size(dst_dir, stem)

    return orig_size, comp_size, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen3.5 hybrid subgraph IRs with nncf.compress_weights()"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/qwen35/Qwen3.5-0.8B-hybrid"),
        help="Source hybrid model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-generated from quantization settings)",
    )
    parser.add_argument(
        "--attn-mode",
        default="int4_sym",
        choices=["fp16", "int4_sym", "int8_sym", "int8_asym"],
        help="Quantization for attention blocks (default: %(default)s)",
    )
    parser.add_argument(
        "--gdn-mode",
        default="fp16",
        choices=["fp16", "int4_sym", "int8_sym", "int8_asym"],
        help="Quantization for GDN blocks (default: %(default)s).",
    )
    parser.add_argument(
        "--head-mode",
        default="fp16",
        choices=["fp16", "int4_sym", "int4_asym", "int8_sym", "int8_asym"],
        help="Quantization for head block (default: %(default)s)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="NNCF group_size for INT4 modes (default: %(default)s)",
    )
    args = parser.parse_args()

    src_dir = args.model_dir.resolve()
    if not src_dir.exists():
        print(f"ERROR: Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    # Auto-generate output dir name from quantization settings
    if args.output_dir is not None:
        dst_dir = args.output_dir.resolve()
    else:
        # Build a suffix that reflects the quantization choices
        parts = []
        if args.attn_mode != "fp16":
            parts.append(f"attn-{args.attn_mode.replace('_', '')}")
        if args.gdn_mode != "fp16":
            parts.append(f"gdn-{args.gdn_mode.replace('_', '')}")
        if args.head_mode != "fp16":
            parts.append(f"head-{args.head_mode.replace('_', '')}")
        if not parts:
            print("ERROR: All modes are fp16 -- nothing to quantize.", file=sys.stderr)
            sys.exit(1)
        suffix = "-".join(parts)
        dst_dir = src_dir.parent / f"{src_dir.name}-{suffix}"

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source:  {src_dir}")
    print(f"Output:  {dst_dir}")
    print(f"Settings:")
    print(f"  Attention blocks: {args.attn_mode}")
    print(f"  GDN blocks:       {args.gdn_mode}")
    print(f"  Head block:       {args.head_mode}")
    print(f"  Group size:       {args.group_size}")
    print()

    core = ov.Core()

    # Collect all subgraph stems and their modes
    subgraphs = []
    for i in range(NUM_GDN_BLOCKS):
        subgraphs.append((f"gdn_block_{i}", args.gdn_mode, "GDN"))
        # Also quantize chunkwise GDN prefill blocks (same weights, different graph)
        if (src_dir / f"gdn_prefill_block_{i}.xml").exists():
            subgraphs.append((f"gdn_prefill_block_{i}", args.gdn_mode, "GDN"))
        # Also quantize S1 (no-Loop) GDN decode blocks
        if (src_dir / f"gdn_s1_block_{i}.xml").exists():
            subgraphs.append((f"gdn_s1_block_{i}", args.gdn_mode, "GDN"))
    for i in range(NUM_ATTN_BLOCKS):
        subgraphs.append((f"attn_block_{i}", args.attn_mode, "Attn"))
    subgraphs.append(("head", args.head_mode, "Head"))

    # Track sizes for summary
    results = []  # (stem, category, mode, orig_size, comp_size, elapsed)
    total_elapsed = 0.0

    # Process each subgraph
    for stem, mode_str, category in subgraphs:
        src_xml = src_dir / f"{stem}.xml"
        if not src_xml.exists():
            print(f"  SKIP (not found): {stem}.xml")
            continue

        action = "copy (fp16)" if mode_str == "fp16" else f"compress ({mode_str})"
        print(f"[{category:4s}] {stem:20s} -> {action} ...", end="", flush=True)

        orig_size, comp_size, elapsed = compress_ir(
            core, src_dir, dst_dir, stem, mode_str, args.group_size
        )
        total_elapsed += elapsed

        if mode_str == "fp16":
            print(f" {format_size(orig_size)}")
        else:
            ratio = orig_size / comp_size if comp_size > 0 else 0
            print(f" {format_size(orig_size)} -> {format_size(comp_size)} ({ratio:.1f}x, {elapsed:.1f}s)")

        results.append((stem, category, mode_str, orig_size, comp_size, elapsed))

    # Copy all other files (non-subgraph IRs and auxiliary files)
    print("\nCopying auxiliary files ...")
    subgraph_stems = {stem for stem, _, _ in subgraphs}
    copied_files = []

    for src_file in sorted(src_dir.iterdir()):
        # Skip directories (e.g. cache/)
        if src_file.is_dir():
            continue

        # Skip subgraph IR files (already handled)
        file_stem = src_file.stem  # e.g. "gdn_block_0" from "gdn_block_0.xml"
        if file_stem in subgraph_stems and src_file.suffix in (".xml", ".bin"):
            continue

        # Quantize embed_tokens.npy to INT8 per-row symmetric
        if src_file.name == "embed_tokens.npy":
            print(f"  Quantizing {src_file.name} to INT8 per-row ...", end="", flush=True)
            embed_fp16 = np.load(str(src_file))  # [vocab, dim] float16
            orig_size = src_file.stat().st_size
            embed_fp32 = embed_fp16.astype(np.float32)
            scales = np.max(np.abs(embed_fp32), axis=1, keepdims=True) / 127.0  # [vocab, 1]
            scales = np.where(scales == 0, 1.0, scales)  # avoid division by zero
            embed_int8 = np.round(embed_fp32 / scales).astype(np.int8)  # [vocab, dim]
            scales_fp16 = scales.squeeze().astype(np.float16)  # [vocab]
            np.save(str(dst_dir / "embed_tokens_int8.npy"), embed_int8)
            np.save(str(dst_dir / "embed_tokens_scales.npy"), scales_fp16)
            int8_size = (dst_dir / "embed_tokens_int8.npy").stat().st_size
            scales_size = (dst_dir / "embed_tokens_scales.npy").stat().st_size
            total_size = int8_size + scales_size
            ratio = orig_size / total_size if total_size > 0 else 0
            print(
                f" {format_size(orig_size)} -> {format_size(total_size)} ({ratio:.1f}x)"
                f"\n    embed_tokens_int8.npy:   {format_size(int8_size)}"
                f"\n    embed_tokens_scales.npy: {format_size(scales_size)}"
            )
            copied_files.append("embed_tokens_int8.npy")
            copied_files.append("embed_tokens_scales.npy")
            continue

        dst_file = dst_dir / src_file.name
        shutil.copy2(str(src_file), str(dst_file))
        copied_files.append(src_file.name)
        print(f"  Copied {src_file.name} ({format_size(src_file.stat().st_size)})")

    # Print summary
    print("\n" + "=" * 72)
    print("QUANTIZATION SUMMARY")
    print("=" * 72)
    print(f"{'Subgraph':<22s} {'Mode':<10s} {'Original':>10s} {'Quantized':>10s} {'Ratio':>7s}")
    print("-" * 72)

    total_orig = 0
    total_comp = 0
    # Group by category for subtotals
    category_stats = {}  # category -> (orig_total, comp_total)

    for stem, category, mode_str, orig_size, comp_size, elapsed in results:
        ratio_str = f"{orig_size / comp_size:.1f}x" if comp_size > 0 and mode_str != "fp16" else "---"
        print(
            f"  {stem:<20s} {mode_str:<10s} {format_size(orig_size):>10s} "
            f"{format_size(comp_size):>10s} {ratio_str:>7s}"
        )
        total_orig += orig_size
        total_comp += comp_size

        if category not in category_stats:
            category_stats[category] = [0, 0]
        category_stats[category][0] += orig_size
        category_stats[category][1] += comp_size

    print("-" * 72)

    # Category subtotals
    for category in ["GDN", "Attn", "Head"]:
        if category in category_stats:
            co, cc = category_stats[category]
            ratio_str = f"{co / cc:.1f}x" if cc > 0 and co != cc else "---"
            print(
                f"  {category + ' total':<20s} {'':10s} {format_size(co):>10s} "
                f"{format_size(cc):>10s} {ratio_str:>7s}"
            )

    print("-" * 72)
    overall_ratio = f"{total_orig / total_comp:.1f}x" if total_comp > 0 else "---"
    print(
        f"  {'TOTAL':<20s} {'':10s} {format_size(total_orig):>10s} "
        f"{format_size(total_comp):>10s} {overall_ratio:>7s}"
    )
    print()
    print(f"Compression time: {total_elapsed:.1f}s")
    print(f"Output directory:  {dst_dir}")


if __name__ == "__main__":
    main()
