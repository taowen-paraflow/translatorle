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
    parser.add_argument(
        "--npu",
        action="store_true",
        help="Export Loop-free IR for NPU (token-by-token prefill, all-static)",
    )
    parser.add_argument(
        "--npuw",
        action="store_true",
        help="Export NPUW_LLM-compatible IR for NPU (dynamic KV, Loop-free GDN)",
    )
    parser.add_argument(
        "--ll2",
        action="store_true",
        help="Export non-stateful Loop IR for LowLatency2 pipeline",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Export hybrid NPU+CPU IR (NPU inference + CPU FP32 GDN state update)",
    )
    parser.add_argument(
        "--multisub",
        action="store_true",
        help="Export multi-subgraph IRs for NPU (6 subgraphs x 4 layers, FP32 between subgraphs)",
    )
    parser.add_argument(
        "--npu-v2",
        action="store_true",
        help="Export NPU v2 subgraph IRs (host-side rotary precomputation, 6 subgraphs x 4 layers)",
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

    # Step 3 (optional): Export NPU model
    if args.npu:
        npu_output_dir = MODELS_DIR / f"{model_name}-npu"
        if args.skip_existing and (npu_output_dir / "openvino_model.xml").exists():
            print(f"\n[Step 3] Skipping NPU export — {npu_output_dir} already exists")
        else:
            print("\n" + "=" * 60)
            print("STEP 3: EXPORT NPU MODEL (Loop-free)")
            print("=" * 60)
            from qwen35.export import export_model_npu
            export_model_npu(
                model_dir=str(hf_model_dir),
                output_dir=str(npu_output_dir),
                compress_to_fp16=not args.no_fp16,
            )

    # Step 4 (optional): Export NPUW_LLM-compatible model
    if args.npuw:
        npuw_output_dir = MODELS_DIR / f"{model_name}-npuw"
        if args.skip_existing and (npuw_output_dir / "openvino_model.xml").exists():
            print(f"\n[Step 4] Skipping NPUW export — {npuw_output_dir} already exists")
        else:
            print("\n" + "=" * 60)
            print("STEP 4: EXPORT NPUW_LLM MODEL (dynamic KV + Loop-free GDN)")
            print("=" * 60)
            from qwen35.export import export_model_npuw
            export_model_npuw(
                model_dir=str(hf_model_dir),
                output_dir=str(npuw_output_dir),
                compress_to_fp16=not args.no_fp16,
            )

    # Step 5 (optional): Export non-stateful Loop IR for LowLatency2 pipeline
    if args.ll2:
        ll2_output_dir = MODELS_DIR / f"{model_name}-loop"
        if args.skip_existing and (ll2_output_dir / "openvino_model.xml").exists():
            print(f"\n[Step 5] Skipping Loop IR export — {ll2_output_dir} already exists")
        else:
            print("\n" + "=" * 60)
            print("STEP 5: EXPORT NON-STATEFUL LOOP IR (for LowLatency2)")
            print("=" * 60)
            from qwen35.export import export_model_loop_ir
            export_model_loop_ir(
                model_dir=str(hf_model_dir),
                output_dir=str(ll2_output_dir),
                compress_to_fp16=not args.no_fp16,
            )

    # Step 6 (optional): Export hybrid NPU+CPU model
    if args.hybrid:
        hybrid_output_dir = MODELS_DIR / f"{model_name}-hybrid"
        if args.skip_existing and (hybrid_output_dir / "openvino_model.xml").exists():
            print(f"\n[Step 6] Skipping hybrid export — {hybrid_output_dir} already exists")
        else:
            print("\n" + "=" * 60)
            print("STEP 6: EXPORT HYBRID NPU+CPU MODEL")
            print("=" * 60)
            from qwen35.export import export_model_hybrid
            export_model_hybrid(
                model_dir=str(hf_model_dir),
                output_dir=str(hybrid_output_dir),
                compress_to_fp16=not args.no_fp16,
            )

    # Step 7 (optional): Export multi-subgraph NPU model
    if args.multisub:
        multisub_output_dir = MODELS_DIR / f"{model_name}-multisub"
        if args.skip_existing and (multisub_output_dir / "subgraph_0.xml").exists():
            print(f"\n[Step 7] Skipping multi-subgraph export — {multisub_output_dir} already exists")
        else:
            print("\n" + "=" * 60)
            print("STEP 7: EXPORT MULTI-SUBGRAPH NPU MODEL (6 subgraphs x 4 layers)")
            print("=" * 60)
            from qwen35.export import export_model_multisubgraph
            export_model_multisubgraph(
                model_dir=str(hf_model_dir),
                output_dir=str(multisub_output_dir),
                compress_to_fp16=not args.no_fp16,
            )

    # Step 8 (optional): Export NPU v2 model (host-side rotary precomputation)
    if args.npu_v2:
        npu_v2_output_dir = MODELS_DIR / f"{model_name}-npu-v2"
        if args.skip_existing and (npu_v2_output_dir / "subgraph_0.xml").exists():
            print(f"\n[Step 8] Skipping NPU v2 export — {npu_v2_output_dir} already exists")
        else:
            print("\n" + "=" * 60)
            print("STEP 8: EXPORT NPU V2 MODEL (host-side rotary, 6 subgraphs x 4 layers)")
            print("=" * 60)
            from qwen35.export_npu_v2 import export_npu_v2
            export_npu_v2(
                model_dir=str(hf_model_dir),
                output_dir=str(npu_v2_output_dir),
                compress_to_fp16=not args.no_fp16,
            )

    elapsed = time.time() - t0
    destinations = [str(output_dir)]
    if args.npu:
        destinations.append(str(MODELS_DIR / f"{model_name}-npu"))
    if args.npuw:
        destinations.append(str(MODELS_DIR / f"{model_name}-npuw"))
    if args.ll2:
        destinations.append(str(MODELS_DIR / f"{model_name}-loop"))
    if args.hybrid:
        destinations.append(str(MODELS_DIR / f"{model_name}-hybrid"))
    if args.multisub:
        destinations.append(str(MODELS_DIR / f"{model_name}-multisub"))
    if args.npu_v2:
        destinations.append(str(MODELS_DIR / f"{model_name}-npu-v2"))
    print(f"\nAll done in {elapsed:.1f}s. Models saved to: {', '.join(destinations)}")


if __name__ == "__main__":
    main()
