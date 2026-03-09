"""Talker model: export, IR surgery, and INT4 quantization (steps 1-9)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Prefix mapping (HF checkpoint -> standalone Qwen3ForCausalLM)
# Derived from qwen3_tts_talker.py hf_to_vllm_mapper
# ---------------------------------------------------------------------------
_PREFIX_MAP = [
    ("talker.model.layers.", "model.layers."),
    ("talker.model.norm.", "model.norm."),
    ("talker.model.codec_embedding.", "model.embed_tokens."),
    ("talker.codec_head.", "lm_head."),
]

_AUXILIARY_PREFIXES = [
    "talker.model.text_embedding.",
    "talker.text_projection.",
    "talker.code_predictor.",
]

# ---------------------------------------------------------------------------
# Architecture constants (Talker, from Qwen3-TTS config.json -> talker_config)
# ---------------------------------------------------------------------------
_TALKER_HIDDEN_SIZE = 1024
_TALKER_VOCAB_SIZE = 3072


# ---------------------------------------------------------------------------
# Step 1: Load safetensors
# ---------------------------------------------------------------------------

def _step1_load_safetensors(hf_model_dir: Path) -> dict[str, torch.Tensor]:
    """Load safetensors and filter to talker-only keys."""
    from safetensors.torch import load_file

    safetensors_path = hf_model_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Safetensors not found: {safetensors_path}")

    print(f"  [Step 1/9] Loading safetensors: {safetensors_path}")
    all_weights = load_file(str(safetensors_path))

    talker_weights = {
        k: v for k, v in all_weights.items() if k.startswith("talker.")
    }
    print(
        f"  [Step 1/9] Total keys: {len(all_weights)}, "
        f"Talker keys: {len(talker_weights)}"
    )
    return talker_weights


# ---------------------------------------------------------------------------
# Step 2: Map weights to Qwen3ForCausalLM namespace
# ---------------------------------------------------------------------------

def _step2_map_weights(
    talker_weights: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Map talker weights to standard Qwen3ForCausalLM namespace.

    Returns:
        decoder_sd: Weights mapped to Qwen3ForCausalLM namespace.
        auxiliary_weights: text_embedding, text_projection, code_predictor
                          (kept in original namespace).
    """
    print("  [Step 2/9] Mapping weights to Qwen3ForCausalLM namespace...")

    decoder_sd: dict[str, torch.Tensor] = {}
    auxiliary_weights: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, tensor in talker_weights.items():
        # Check auxiliary first
        is_auxiliary = False
        for aux_prefix in _AUXILIARY_PREFIXES:
            if key.startswith(aux_prefix):
                auxiliary_weights[key] = tensor
                is_auxiliary = True
                break
        if is_auxiliary:
            continue

        # Map backbone keys
        mapped = False
        for old_prefix, new_prefix in _PREFIX_MAP:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                decoder_sd[new_key] = tensor
                mapped = True
                break
        if not mapped:
            unmapped.append(key)

    print(
        f"  [Step 2/9] Mapped decoder keys: {len(decoder_sd)}, "
        f"Auxiliary keys: {len(auxiliary_weights)}"
    )
    if unmapped:
        print(f"  [Step 2/9] WARNING: Unmapped keys ({len(unmapped)}): {unmapped[:5]}")

    return decoder_sd, auxiliary_weights


# ---------------------------------------------------------------------------
# Step 3: Create Qwen3Config
# ---------------------------------------------------------------------------

def _step3_create_config():
    """Create Qwen3Config matching the talker sub-network."""
    from transformers import Qwen3Config

    print("  [Step 3/9] Creating Qwen3Config for the talker backbone...")
    qwen3_config = Qwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        max_position_embeddings=32768,
        num_attention_heads=16,
        num_hidden_layers=28,
        num_key_value_heads=8,
        rms_norm_eps=1e-06,
        rope_scaling={
            "interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
            "type": "default",
        },
        rope_theta=1000000.0,
        sliding_window=None,
        tie_word_embeddings=False,
        vocab_size=3072,
        head_dim=128,
        attention_dropout=0.0,
        attention_bias=False,
        hidden_act="silu",
        initializer_range=0.02,
        use_cache=True,
    )
    print(
        f"  [Step 3/9] Config: hidden={qwen3_config.hidden_size}, "
        f"layers={qwen3_config.num_hidden_layers}, "
        f"heads={qwen3_config.num_attention_heads}/{qwen3_config.num_key_value_heads}, "
        f"intermediate={qwen3_config.intermediate_size}, "
        f"vocab={qwen3_config.vocab_size}"
    )
    return qwen3_config


# ---------------------------------------------------------------------------
# Step 4: Create standalone model, load weights, save
# ---------------------------------------------------------------------------

def _step4_create_standalone(
    decoder_sd: dict[str, torch.Tensor],
    qwen3_config,
    standalone_dir: Path,
    hf_model_dir: Path,
) -> None:
    """Create Qwen3ForCausalLM, load mapped weights, save to standalone_dir."""
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    print(f"  [Step 4/9] Creating standalone Qwen3ForCausalLM -> {standalone_dir}")

    standalone = Qwen3ForCausalLM(qwen3_config)
    standalone.eval()

    missing, unexpected = standalone.load_state_dict(decoder_sd, strict=False)
    if missing:
        print(f"  [Step 4/9] WARNING: Missing keys ({len(missing)})")
    else:
        print("  [Step 4/9] No missing keys -- all decoder weights loaded")
    if unexpected:
        print(f"  [Step 4/9] WARNING: Unexpected keys ({len(unexpected)})")

    # Quick forward pass verification
    with torch.no_grad():
        dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        output = standalone(input_ids=dummy_ids)
        logits_shape = output.logits.shape
        assert logits_shape == (1, 5, _TALKER_VOCAB_SIZE), (
            f"Expected [1, 5, {_TALKER_VOCAB_SIZE}], got {list(logits_shape)}"
        )
    print(f"  [Step 4/9] Forward pass OK, logits shape: {list(logits_shape)}")

    # Save standalone model
    os.makedirs(str(standalone_dir), exist_ok=True)
    standalone.save_pretrained(str(standalone_dir))
    print(f"  [Step 4/9] Model saved to: {standalone_dir}")

    # Copy tokenizer so optimum-intel does not complain
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(hf_model_dir), trust_remote_code=True
        )
        tokenizer.save_pretrained(str(standalone_dir))
        print("  [Step 4/9] Tokenizer saved to standalone dir")
    except Exception:
        print("  [Step 4/9] Could not copy tokenizer (non-fatal)")

    del standalone


# ---------------------------------------------------------------------------
# Step 5: Export via optimum-intel + save numpy artifacts
# ---------------------------------------------------------------------------

def _step5_export_openvino(
    standalone_dir: Path,
    stateful_dir: Path,
    talker_weights: dict[str, torch.Tensor],
) -> None:
    """Export stateful OV model and save auxiliary numpy arrays."""
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    print(f"  [Step 5/9] Exporting with optimum-intel (stateful=True) -> {stateful_dir}")

    ov_model = OVModelForCausalLM.from_pretrained(
        str(standalone_dir),
        export=True,
        stateful=True,
        compile=False,
        trust_remote_code=True,
    )
    os.makedirs(str(stateful_dir), exist_ok=True)
    ov_model.save_pretrained(str(stateful_dir))
    print(f"  [Step 5/9] OV model saved to: {stateful_dir}")

    # Copy tokenizer to stateful dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(standalone_dir))
        tokenizer.save_pretrained(str(stateful_dir))
    except Exception:
        pass

    del ov_model

    # List exported files
    for f in sorted(os.listdir(str(stateful_dir))):
        fpath = stateful_dir / f
        if fpath.is_file():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"  [Step 5/9]   {f}: {size_mb:.1f} MB")

    # -- Save numpy artifacts from original talker weights --

    # Codec embedding [3072, 1024]
    codec_embed_key = "talker.model.codec_embedding.weight"
    if codec_embed_key in talker_weights:
        arr = talker_weights[codec_embed_key].float().numpy()
        np.save(str(stateful_dir / "talker_embed_tokens.npy"), arr)
        print(f"  [Step 5/9] talker_embed_tokens.npy: shape={arr.shape}")
    else:
        print(f"  [Step 5/9] WARNING: Key '{codec_embed_key}' not found")

    # Text embedding [151936, 2048]
    text_embed_key = "talker.model.text_embedding.weight"
    if text_embed_key in talker_weights:
        arr = talker_weights[text_embed_key].float().numpy()
        np.save(str(stateful_dir / "text_embedding.npy"), arr)
        print(f"  [Step 5/9] text_embedding.npy: shape={arr.shape}")
    else:
        print(f"  [Step 5/9] WARNING: Key '{text_embed_key}' not found")

    # Text projection MLP (2-layer: linear_fc1 + linear_fc2, both with bias)
    proj_keys = {
        "linear_fc1.weight": "talker.text_projection.linear_fc1.weight",
        "linear_fc1.bias": "talker.text_projection.linear_fc1.bias",
        "linear_fc2.weight": "talker.text_projection.linear_fc2.weight",
        "linear_fc2.bias": "talker.text_projection.linear_fc2.bias",
    }
    proj_arrays: dict[str, np.ndarray] = {}
    all_found = True
    for short_name, full_key in proj_keys.items():
        if full_key in talker_weights:
            proj_arrays[short_name] = talker_weights[full_key].float().numpy()
        else:
            print(f"  [Step 5/9] WARNING: Key '{full_key}' not found")
            all_found = False

    if all_found:
        np.savez(str(stateful_dir / "text_projection.npz"), **proj_arrays)
        for name, arr in proj_arrays.items():
            print(f"  [Step 5/9] text_projection.npz[{name}]: shape={arr.shape}")
    else:
        print("  [Step 5/9] WARNING: Incomplete text projection -- skipping .npz")


# ---------------------------------------------------------------------------
# Step 6: IR surgery -> dual-output CPU model (talker_stateful_embeds)
# ---------------------------------------------------------------------------

def _step6_ir_surgery_dual(stateful_dir: Path, embeds_dir: Path) -> None:
    """IR surgery: remove input_ids, add inputs_embeds + hidden_states output."""
    from ._ir_surgery import do_ir_surgery

    src_xml = stateful_dir / "openvino_model.xml"
    print(f"  [Step 6/9] IR surgery (dual-output CPU model) -> {embeds_dir}")
    do_ir_surgery(
        src_xml=src_xml,
        dst_dir=embeds_dir,
        hidden_size=_TALKER_HIDDEN_SIZE,
        add_hidden_output=True,
        vocab_size=_TALKER_VOCAB_SIZE,
    )
    print(f"  [Step 6/9] Saved to: {embeds_dir}")


# ---------------------------------------------------------------------------
# Step 7: IR surgery -> single-output NPU model + pseudo-inverse
# ---------------------------------------------------------------------------

def _step7_ir_surgery_npu(
    stateful_dir: Path,
    npu_dir: Path,
    talker_weights: dict[str, torch.Tensor],
    models_dir: Path,
) -> None:
    """IR surgery for NPU (single output) + compute lm_head pseudo-inverse."""
    from ._ir_surgery import do_ir_surgery

    src_xml = stateful_dir / "openvino_model.xml"
    print(f"  [Step 7/9] IR surgery (single-output NPU model) -> {npu_dir}")
    do_ir_surgery(
        src_xml=src_xml,
        dst_dir=npu_dir,
        hidden_size=_TALKER_HIDDEN_SIZE,
        add_hidden_output=False,
        vocab_size=_TALKER_VOCAB_SIZE,
    )
    print(f"  [Step 7/9] Saved to: {npu_dir}")

    # Compute pseudo-inverse of lm_head.weight from ORIGINAL safetensors
    codec_head_key = "talker.codec_head.weight"
    if codec_head_key in talker_weights:
        lm_head_w = talker_weights[codec_head_key].float().numpy()  # [3072, 1024]
        print(
            f"  [Step 7/9] Computing pseudo-inverse of {codec_head_key}: "
            f"shape={lm_head_w.shape}"
        )
        # logits = hidden @ W.T, so hidden = logits @ pinv(W.T)
        # pinv(W.T) = W @ inv(W.T @ W), shape [3072, 1024]
        W = lm_head_w  # [3072, 1024]
        WtW = W.T @ W  # [1024, 1024]
        WtW_inv = np.linalg.inv(WtW)  # [1024, 1024]
        lm_head_pinv = W @ WtW_inv  # [3072, 1024]
        pinv_path = models_dir / "talker_lm_head_pinv.npy"
        np.save(str(pinv_path), lm_head_pinv)
        print(f"  [Step 7/9] talker_lm_head_pinv.npy: shape={lm_head_pinv.shape}")
    else:
        print(
            f"  [Step 7/9] WARNING: Key '{codec_head_key}' not found, "
            "cannot compute pseudo-inverse"
        )


# ---------------------------------------------------------------------------
# Step 8: INT4 quantization (talker_stateful_embeds -> talker_stateful_embeds_int4)
# ---------------------------------------------------------------------------

def _step8_quantize_int4(embeds_dir: Path, int4_dir: Path) -> None:
    """Quantize the dual-output FP16 model to INT4 with NNCF."""
    import nncf
    import openvino as ov

    src_xml = embeds_dir / "openvino_model.xml"
    print(f"  [Step 8/9] INT4 quantization (NNCF INT4_SYM) -> {int4_dir}")
    print(f"  [Step 8/9]   Source: {src_xml}")
    print(f"  [Step 8/9]   NNCF version: {nncf.__version__}")

    core = ov.Core()
    model = core.read_model(str(src_xml))

    # Report FP16 model size
    bin_path = src_xml.with_suffix(".bin")
    if bin_path.exists():
        fp16_mb = bin_path.stat().st_size / (1024 * 1024)
        print(f"  [Step 8/9]   FP16 model size: {fp16_mb:.1f} MB")
    else:
        fp16_mb = 0.0

    print("  [Step 8/9]   Compressing weights (INT4_SYM, group_size=128, ratio=1.0)...")
    compressed = nncf.compress_weights(
        model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        group_size=128,
        ratio=1.0,
    )
    del model

    int4_dir.mkdir(parents=True, exist_ok=True)
    int4_xml = int4_dir / "openvino_model.xml"
    ov.save_model(compressed, str(int4_xml))
    del compressed
    print(f"  [Step 8/9]   Saved compressed model to: {int4_dir}")

    # Report INT4 model size
    int4_bin = int4_xml.with_suffix(".bin")
    if int4_bin.exists():
        int4_mb = int4_bin.stat().st_size / (1024 * 1024)
        print(f"  [Step 8/9]   INT4 model size: {int4_mb:.1f} MB")
        if fp16_mb > 0:
            ratio = int4_mb / fp16_mb
            print(f"  [Step 8/9]   Compression: {1.0 / ratio:.2f}x ({(1.0 - ratio) * 100:.1f}% reduction)")

    # Copy config/tokenizer files from source directory
    copied = []
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "generation_config.json",
    ]:
        src = embeds_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(int4_dir / fname))
            copied.append(fname)
    if copied:
        print(f"  [Step 8/9]   Copied config files: {', '.join(copied)}")


# ---------------------------------------------------------------------------
# Step 9: IR surgery on INT4 -> NPU variant
# ---------------------------------------------------------------------------

def _step9_ir_surgery_npu_int4(int4_dir: Path, npu_int4_dir: Path) -> None:
    """Strip hidden_states output from INT4 embeds model for NPU (single output).

    The INT4 model (from step 8) already has inputs_embeds (IR surgery was done
    in step 6 before quantization). We only need to remove the second output
    (hidden_states) so NPUW_LLM can handle the single-output model.
    """
    import openvino as ov
    import shutil

    src_xml = int4_dir / "openvino_model.xml"
    print(f"  [Step 9/9] Strip hidden_states output (NPU INT4) -> {npu_int4_dir}")

    core = ov.Core()
    model = core.read_model(str(src_xml))

    # The INT4 embeds model has 2 outputs: [logits, hidden_states]
    # NPU NPUW_LLM requires single output, so keep only logits
    if len(model.outputs) > 1:
        print(f"  [Step 9/9] Model has {len(model.outputs)} outputs, stripping to 1")
        logits_result = model.outputs[0].node
        # Build new model with only the logits output
        new_model = ov.Model(
            results=[logits_result],
            parameters=list(model.get_parameters()),
            sinks=list(model.get_sinks()),
            name="model",
        )
        new_model.validate_nodes_and_infer_types()
    else:
        print("  [Step 9/9] Model already has single output, copying as-is")
        new_model = model

    npu_int4_dir.mkdir(parents=True, exist_ok=True)
    dst_xml = npu_int4_dir / "openvino_model.xml"
    ov.save_model(new_model, str(dst_xml), compress_to_fp16=False)

    # Copy config/tokenizer files
    for fname in sorted(int4_dir.iterdir()):
        if fname.suffix in (".json", ".txt"):
            shutil.copy2(str(fname), str(npu_int4_dir / fname.name))

    print(f"  [Step 9/9] Saved to: {npu_int4_dir}")


# ============================================================================
# Public API
# ============================================================================

def export_talker(
    hf_model_dir: Path,
    models_dir: Path,
    skip_existing: bool = False,
) -> None:
    """Run all talker export steps 1-9.

    Args:
        hf_model_dir: Path to the HF model directory containing model.safetensors
        models_dir: Output root directory for exported models
        skip_existing: If True, skip steps whose output directories already exist
    """
    print("=" * 60)
    print("  Talker Export Pipeline (Steps 1-9)")
    print("=" * 60)

    # Output directories
    standalone_dir = models_dir / "talker_standalone"
    stateful_dir = models_dir / "talker_stateful"
    embeds_dir = models_dir / "talker_stateful_embeds"
    npu_dir = models_dir / "talker_npu"
    int4_dir = models_dir / "talker_stateful_embeds_int4"
    npu_int4_dir = models_dir / "talker_npu_int4"

    # Determine which steps can be skipped
    skip_standalone = (
        skip_existing and (standalone_dir / "config.json").exists()
    )
    skip_stateful = (
        skip_existing and (stateful_dir / "openvino_model.xml").exists()
    )
    skip_embeds = (
        skip_existing and (embeds_dir / "openvino_model.xml").exists()
    )
    skip_npu = (
        skip_existing
        and (npu_dir / "openvino_model.xml").exists()
        and (models_dir / "talker_lm_head_pinv.npy").exists()
    )
    skip_int4 = (
        skip_existing and (int4_dir / "openvino_model.xml").exists()
    )
    skip_npu_int4 = (
        skip_existing and (npu_int4_dir / "openvino_model.xml").exists()
    )

    # Fast path: everything already exported
    if all([skip_standalone, skip_stateful, skip_embeds, skip_npu,
            skip_int4, skip_npu_int4]):
        print("  [Steps 1-9] All talker outputs already exist, skipping")
        return

    # Step 1: Load safetensors (needed by steps 2, 5, 7)
    need_safetensors = (
        not skip_standalone or not skip_stateful or not skip_npu
    )
    talker_weights: dict[str, torch.Tensor] | None = None
    if need_safetensors:
        talker_weights = _step1_load_safetensors(hf_model_dir)
    else:
        print("  [Step 1/9] Skipping safetensors load (not needed)")

    # Steps 2-4: Map weights, create config, create standalone model
    if skip_standalone:
        print("  [Steps 2-4/9] Skipping (talker_standalone already exists)")
    else:
        assert talker_weights is not None
        decoder_sd, _aux = _step2_map_weights(talker_weights)
        qwen3_config = _step3_create_config()
        _step4_create_standalone(decoder_sd, qwen3_config, standalone_dir, hf_model_dir)
        del decoder_sd, _aux, qwen3_config

    # Step 5: Export via optimum-intel + numpy artifacts
    if skip_stateful:
        print("  [Step 5/9] Skipping (talker_stateful already exists)")
    else:
        assert talker_weights is not None
        _step5_export_openvino(standalone_dir, stateful_dir, talker_weights)

    # Step 6: IR surgery -> dual-output CPU model
    if skip_embeds:
        print("  [Step 6/9] Skipping (talker_stateful_embeds already exists)")
    else:
        _step6_ir_surgery_dual(stateful_dir, embeds_dir)

    # Step 7: IR surgery -> single-output NPU model + pseudo-inverse
    if skip_npu:
        print("  [Step 7/9] Skipping (talker_npu already exists)")
    else:
        assert talker_weights is not None
        _step7_ir_surgery_npu(stateful_dir, npu_dir, talker_weights, models_dir)

    # Free safetensors weights
    del talker_weights

    # Step 8: INT4 quantization
    if skip_int4:
        print("  [Step 8/9] Skipping (talker_stateful_embeds_int4 already exists)")
    else:
        _step8_quantize_int4(embeds_dir, int4_dir)

    # Step 9: IR surgery on INT4 -> NPU variant
    if skip_npu_int4:
        print("  [Step 9/9] Skipping (talker_npu_int4 already exists)")
    else:
        _step9_ir_surgery_npu_int4(int4_dir, npu_int4_dir)

    print("=" * 60)
    print("  Talker Export Pipeline complete (Steps 1-9)")
    print("=" * 60)
