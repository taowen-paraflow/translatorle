"""Prepare HY-MT1.5-1.8B model for OpenVINO NPU deployment.

This single script performs ALL model preparation steps:
  1. Download HY-MT1.5-1.8B from HuggingFace (or use local path)
  2. Load the HunYuanDenseV1ForCausalLM model
  3. Remap weights to Qwen3ForCausalLM format (query_layernorm -> q_norm, etc.)
  4. Save standalone Qwen3ForCausalLM checkpoint
  5. Export via optimum-intel as a stateful OpenVINO model with weight compression
  6. Convert tokenizer to OpenVINO format for openvino_genai.LLMPipeline

Output layout under models/:
    models/
    +-- hy_mt_ov/                  Final OpenVINO IR + OV tokenizer (★ inference)
    +-- hy_mt_qwen3_standalone/    Intermediate HF checkpoint (Qwen3ForCausalLM)
    +-- HY-MT1.5-1.8B/            Downloaded original weights (if --download)

Usage:
    uv run python hymt/scripts/prepare_mt_models.py
    uv run python hymt/scripts/prepare_mt_models.py --model-id C:\\path\\to\\HY-MT1.5-1.8B
    uv run python hymt/scripts/prepare_mt_models.py --skip-download --model-id C:\\path\\to\\HY-MT1.5-1.8B
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timer():
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0


def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

def _resolve_paths(project_root: Path):
    m = project_root / "models"
    return {
        "models_dir": m,
        "download_dir": m / "HY-MT1.5-1.8B",
        "standalone_dir": m / "hy_mt_qwen3_standalone",
        "export_dir": m / "hy_mt_ov",
    }


# ============================================================================
# Step 1 -- Download model
# ============================================================================

def step1_download(paths: dict):
    print("=" * 60)
    print("Step 1: Downloading HY-MT1.5-1.8B from HuggingFace")
    print("=" * 60)
    elapsed = _timer()

    from huggingface_hub import snapshot_download

    dst = str(paths["download_dir"])
    print(f"  Destination: {dst}")
    snapshot_download(repo_id="tencent/HY-MT1.5-1.8B", local_dir=dst)
    print(f"  Time: {elapsed():.1f}s")
    print()
    return dst


# ============================================================================
# Step 2 -- Load model
# ============================================================================

def step2_load_model(model_id: str):
    print("=" * 60)
    print("Step 2: Loading HunYuanDenseV1ForCausalLM")
    print("=" * 60)
    elapsed = _timer()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.float32,
    )
    model.eval()
    print(f"  Loaded from: {model_id}")
    print(f"  Time: {elapsed():.1f}s")
    print()
    return model


# ============================================================================
# Step 3 -- Remap state dict to Qwen3 format
# ============================================================================

def step3_remap_state_dict(model):
    """Rename HunYuanDenseV1 keys to Qwen3 format.

    The only difference: query_layernorm -> q_norm, key_layernorm -> k_norm.
    """
    print("=" * 60)
    print("Step 3: Remapping state dict (HunYuanDenseV1 -> Qwen3)")
    print("=" * 60)
    elapsed = _timer()

    sd = model.state_dict()
    new_sd = {}
    renamed_count = 0
    for k, v in sd.items():
        new_key = k.replace("query_layernorm", "q_norm").replace("key_layernorm", "k_norm")
        if new_key != k:
            renamed_count += 1
        new_sd[new_key] = v
    print(f"  Renamed {renamed_count} keys, total {len(new_sd)} keys")
    print(f"  Time: {elapsed():.1f}s")
    print()
    return new_sd


# ============================================================================
# Step 4 -- Create standalone Qwen3ForCausalLM, save checkpoint
# ============================================================================

def step4_create_standalone(state_dict: dict, paths: dict, model_id: str):
    print("=" * 60)
    print("Step 4: Creating standalone Qwen3ForCausalLM")
    print("=" * 60)
    elapsed = _timer()

    from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer

    # Config matching HunYuanDenseV1-1.8B architecture
    config = Qwen3Config(
        hidden_size=2048,
        intermediate_size=6144,
        max_position_embeddings=262144,
        num_attention_heads=16,
        num_hidden_layers=32,
        num_key_value_heads=4,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        rope_scaling=None,
        sliding_window=None,
        tie_word_embeddings=True,
        vocab_size=120818,
        head_dim=128,
        use_cache=True,
    )

    standalone = Qwen3ForCausalLM(config)
    result = standalone.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {result.unexpected_keys}")

    save_dir = paths["standalone_dir"]
    os.makedirs(save_dir, exist_ok=True)
    standalone.save_pretrained(str(save_dir))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(save_dir))

    print(f"  Saved to: {save_dir}")
    print(f"  Time: {elapsed():.1f}s")
    print()
    return str(save_dir)


# ============================================================================
# Step 5 -- Export via optimum-intel with weight compression
# ============================================================================

def step5_export_openvino(standalone_dir: str, paths: dict):
    print("=" * 60)
    print("Step 5: Exporting to OpenVINO IR with weight compression")
    print("=" * 60)
    elapsed = _timer()

    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    export_dir = str(paths["export_dir"])
    os.makedirs(export_dir, exist_ok=True)

    print(f"  Source: {standalone_dir}")
    print(f"  Target: {export_dir}")
    print("  This may take several minutes...")

    ov_model = OVModelForCausalLM.from_pretrained(
        standalone_dir,
        export=True,
        load_in_4bit=True,
    )
    ov_model.save_pretrained(export_dir)

    # Save HF tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(standalone_dir)
    tokenizer.save_pretrained(export_dir)

    # Convert tokenizer to OpenVINO format for LLMPipeline
    from openvino_tokenizers import convert_tokenizer
    import openvino as ov
    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, os.path.join(export_dir, "openvino_tokenizer.xml"))
    ov.save_model(ov_detokenizer, os.path.join(export_dir, "openvino_detokenizer.xml"))

    # List output files
    for f in sorted(os.listdir(export_dir)):
        fpath = os.path.join(export_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    {f}: {size_mb:.1f} MB")

    print(f"  Time: {elapsed():.1f}s")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare HY-MT1.5-1.8B for OpenVINO NPU deployment",
    )
    parser.add_argument(
        "--model-id",
        default="tencent/HY-MT1.5-1.8B",
        help="HuggingFace model ID or local path (default: tencent/HY-MT1.5-1.8B)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use --model-id as local path)",
    )
    args = parser.parse_args()

    # Project root: hymt/scripts/ -> hymt/ -> translatorle/
    project_root = Path(__file__).resolve().parent.parent.parent
    paths = _resolve_paths(project_root)

    print()
    print("=" * 60)
    print("  HY-MT1.5-1.8B Model Preparation for OpenVINO NPU")
    print("=" * 60)
    print(f"  Model ID:     {args.model_id}")
    print(f"  Project root: {project_root}")
    print(f"  Output dir:   {paths['models_dir']}")
    print("=" * 60)
    print()

    os.makedirs(str(paths["models_dir"]), exist_ok=True)
    total_start = time.perf_counter()

    # Step 1: Download (unless skipped or using local path)
    model_path = Path(args.model_id)
    if not args.skip_download and not model_path.is_dir():
        model_id = step1_download(paths)
    else:
        model_id = args.model_id
        if not model_path.is_dir():
            # Check if already downloaded to models/
            model_id = str(paths["download_dir"])
        print(f"  Using model at: {model_id}")
        print()

    # Step 2: Load model
    model = step2_load_model(model_id)

    # Step 3: Remap state dict
    state_dict = step3_remap_state_dict(model)
    del model
    _free_memory()

    # Step 4: Create standalone Qwen3 checkpoint
    standalone_dir = step4_create_standalone(state_dict, paths, model_id)
    del state_dict
    _free_memory()

    # Step 5: Export to OpenVINO
    step5_export_openvino(standalone_dir, paths)

    total_elapsed = time.perf_counter() - total_start
    print("=" * 60)
    print(f"  All done! Total time: {total_elapsed:.1f}s")
    print("=" * 60)
    print()
    print("Output files:")
    for key, path in sorted(paths.items()):
        if key == "models_dir":
            continue
        exists = path.exists() if isinstance(path, Path) else os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {path}")
    print()


if __name__ == "__main__":
    main()
