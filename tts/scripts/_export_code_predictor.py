"""Code Predictor model: export and IR surgery (steps 10-13).

Extracts the 5-layer code predictor transformer from the Qwen3-TTS safetensors,
exports it as a stateful OpenVINO model with KV cache, performs IR surgery
(remove input_ids, add inputs_embeds), and saves auxiliary numpy artifacts
(codec embeddings, lm_heads, proj_in).

Identity lm_head trick:
  - Set vocab_size = 1024 (= hidden_size) so lm_head is square
  - Set lm_head.weight = torch.eye(1024) so "logits" = hidden_states
  - Set tie_word_embeddings = False to keep embed_tokens and lm_head independent
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Architecture constants (Code Predictor, from Qwen3-TTS config)
# ---------------------------------------------------------------------------
_CP_PREFIX = "talker.code_predictor."
_CP_HIDDEN_SIZE = 1024
_CP_NUM_LAYERS = 5
_CP_NUM_HEADS = 16
_CP_NUM_KV_HEADS = 8
_CP_HEAD_DIM = 128
_CP_INTERMEDIATE_SIZE = 3072
_CP_RMS_NORM_EPS = 1e-6
_CP_ROPE_THETA = 1_000_000
_CP_CODE_GROUPS = 16
_NUM_AUX_GROUPS = _CP_CODE_GROUPS - 1  # 15 (groups 1..15)
_IDENTITY_VOCAB_SIZE = _CP_HIDDEN_SIZE  # 1024 -- the identity lm_head trick
_TALKER_HIDDEN_SIZE = 1024


# ---------------------------------------------------------------------------
# Step 10: Load safetensors and extract code predictor weights
# ---------------------------------------------------------------------------

def _step10_load_and_map(
    safetensors_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load safetensors and extract code predictor weights with identity lm_head.

    Returns:
        cp_weights: Raw CP weights (prefix stripped, all keys including auxiliary)
        decoder_sd: Mapped state dict for Qwen3ForCausalLM (backbone + identity lm_head)
    """
    from safetensors.torch import load_file

    print(f"  [Step 10/15] Loading safetensors: {safetensors_path}")
    all_weights = load_file(str(safetensors_path))
    print(f"  [Step 10/15] Total keys in safetensors: {len(all_weights)}")

    # Filter to code predictor keys and strip prefix
    cp_weights: dict[str, torch.Tensor] = {}
    for key, tensor in all_weights.items():
        if key.startswith(_CP_PREFIX):
            cp_weights[key[len(_CP_PREFIX) :]] = tensor

    print(f"  [Step 10/15] Code predictor keys: {len(cp_weights)}")

    # Categorise for diagnostics
    layer_keys = [k for k in cp_weights if k.startswith("model.layers.")]
    norm_keys = [k for k in cp_weights if k.startswith("model.norm.")]
    emb_keys = [k for k in cp_weights if k.startswith("model.codec_embedding.")]
    head_keys = [k for k in cp_weights if k.startswith("lm_head.")]
    proj_keys = [k for k in cp_weights if k.startswith("small_to_mtp_projection.")]
    print(f"  [Step 10/15]   Transformer layers: {len(layer_keys)} tensors")
    print(f"  [Step 10/15]   Final norm:         {len(norm_keys)} tensors")
    print(f"  [Step 10/15]   Codec embeddings:   {len(emb_keys)} tensors")
    print(f"  [Step 10/15]   LM heads:           {len(head_keys)} tensors")
    print(f"  [Step 10/15]   Proj-in:            {len(proj_keys)} tensors")

    # Map transformer backbone to Qwen3ForCausalLM namespace.
    # Skip auxiliary modules (exported separately as numpy in step 13).
    SKIP_PREFIXES = [
        "model.codec_embedding.",
        "lm_head.",
        "small_to_mtp_projection.",
    ]

    decoder_sd: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, tensor in cp_weights.items():
        is_skip = False
        for prefix in SKIP_PREFIXES:
            if key.startswith(prefix):
                skipped.append(key)
                is_skip = True
                break
        if is_skip:
            continue

        # Transformer backbone keys map directly (same key names)
        if key.startswith("model.layers.") or key.startswith("model.norm."):
            decoder_sd[key] = tensor

    # KEY TRICK: identity lm_head so "logits" output = hidden_states
    decoder_sd["lm_head.weight"] = torch.eye(
        _IDENTITY_VOCAB_SIZE, dtype=torch.float32
    )

    print(
        f"  [Step 10/15] Mapped decoder keys: {len(decoder_sd)} "
        f"(identity lm_head included)"
    )
    print(f"  [Step 10/15] Skipped auxiliary keys: {len(skipped)}")

    return cp_weights, decoder_sd


# ---------------------------------------------------------------------------
# Step 11: Create config, standalone model, export stateful OV
# ---------------------------------------------------------------------------

def _step11_config_and_export(
    decoder_sd: dict[str, torch.Tensor],
    models_dir: Path,
    hf_model_dir: Path,
) -> Path:
    """Create Qwen3Config, build standalone model, export via optimum-intel."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    standalone_dir = models_dir / "cp_standalone"
    stateful_dir = models_dir / "cp_stateful"

    # -- 11a: Create Qwen3Config --
    print("  [Step 11/15] Creating Qwen3Config for Code Predictor")
    qwen3_config = Qwen3Config(
        hidden_size=_CP_HIDDEN_SIZE,
        intermediate_size=_CP_INTERMEDIATE_SIZE,
        max_position_embeddings=65536,
        num_attention_heads=_CP_NUM_HEADS,
        num_hidden_layers=_CP_NUM_LAYERS,
        num_key_value_heads=_CP_NUM_KV_HEADS,
        rms_norm_eps=_CP_RMS_NORM_EPS,
        rope_theta=_CP_ROPE_THETA,
        sliding_window=None,
        tie_word_embeddings=False,  # keep embed_tokens and lm_head independent
        vocab_size=_IDENTITY_VOCAB_SIZE,  # 1024 = hidden_size for identity trick
        head_dim=_CP_HEAD_DIM,
        attention_dropout=0.0,
        attention_bias=False,
        hidden_act="silu",
        initializer_range=0.02,
        use_cache=True,
        attn_implementation="eager",
    )
    print(
        f"  [Step 11/15] Config: hidden={qwen3_config.hidden_size}, "
        f"layers={qwen3_config.num_hidden_layers}, "
        f"heads={qwen3_config.num_attention_heads}/{qwen3_config.num_key_value_heads}, "
        f"intermediate={qwen3_config.intermediate_size}, "
        f"vocab={qwen3_config.vocab_size} (identity trick)"
    )

    # -- 11b: Create standalone model and load weights --
    print("  [Step 11/15] Creating Qwen3ForCausalLM and loading weights")
    standalone = Qwen3ForCausalLM(qwen3_config)
    standalone.eval()

    missing, unexpected = standalone.load_state_dict(decoder_sd, strict=False)
    if missing:
        print(
            f"  [Step 11/15] Missing keys ({len(missing)}) "
            "-- expected for embed_tokens"
        )
    if unexpected:
        print(f"  [Step 11/15] WARNING: Unexpected keys ({len(unexpected)})")

    # Verify identity lm_head
    identity_ok = torch.allclose(
        standalone.lm_head.weight.data,
        torch.eye(_IDENTITY_VOCAB_SIZE),
    )
    assert identity_ok, "lm_head.weight should be identity matrix!"
    print(f"  [Step 11/15] lm_head.weight is identity: {identity_ok}")

    # -- 11c: Save standalone HF model --
    print(f"  [Step 11/15] Saving standalone to {standalone_dir}")
    os.makedirs(str(standalone_dir), exist_ok=True)
    standalone.save_pretrained(str(standalone_dir))

    # Copy tokenizer (optimum-intel requires tokenizer files)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(hf_model_dir), trust_remote_code=True
        )
        tokenizer.save_pretrained(str(standalone_dir))
        print("  [Step 11/15] Tokenizer saved to standalone dir")
    except Exception:
        print("  [Step 11/15] Could not copy tokenizer (non-fatal)")

    # Free standalone model before heavy export
    del standalone

    # -- 11d: Export with optimum-intel (stateful) --
    print(
        f"  [Step 11/15] Exporting with optimum-intel "
        f"(stateful=True) to {stateful_dir}"
    )

    from optimum.intel import OVModelForCausalLM

    ov_model = OVModelForCausalLM.from_pretrained(
        str(standalone_dir),
        export=True,
        stateful=True,
        compile=False,
        trust_remote_code=False,
    )
    os.makedirs(str(stateful_dir), exist_ok=True)
    ov_model.save_pretrained(str(stateful_dir))

    # Copy tokenizer to stateful dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(standalone_dir))
        tokenizer.save_pretrained(str(stateful_dir))
    except Exception:
        pass

    # List exported files
    for f in sorted(os.listdir(str(stateful_dir))):
        fpath = stateful_dir / f
        if fpath.is_file():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"  [Step 11/15]   {f}: {size_mb:.1f} MB")

    print(f"  [Step 11/15] Stateful model exported to {stateful_dir}")
    return stateful_dir


# ---------------------------------------------------------------------------
# Step 12: IR surgery on Code Predictor
# ---------------------------------------------------------------------------

def _step12_ir_surgery(models_dir: Path) -> None:
    """Run IR surgery: remove input_ids, add inputs_embeds."""
    from ._ir_surgery import do_ir_surgery

    stateful_dir = models_dir / "cp_stateful"
    final_dir = models_dir / "cp_stateful_embeds"
    src_xml = stateful_dir / "openvino_model.xml"

    print(f"  [Step 12/15] Running IR surgery on Code Predictor")
    print(f"  [Step 12/15]   Source: {src_xml}")
    print(f"  [Step 12/15]   Target: {final_dir}")

    do_ir_surgery(
        src_xml=src_xml,
        dst_dir=final_dir,
        hidden_size=_CP_HIDDEN_SIZE,
        add_hidden_output=False,  # identity trick means logits = hidden_states
        vocab_size=_IDENTITY_VOCAB_SIZE,  # 1024, not 3072
        model_name="cp_stateful_embeds",
    )

    print("  [Step 12/15] IR surgery complete")


# ---------------------------------------------------------------------------
# Step 13: Export auxiliary numpy artifacts
# ---------------------------------------------------------------------------

def _step13_export_numpy(
    cp_weights: dict[str, torch.Tensor],
    output_dir: Path,
) -> None:
    """Export codec embeddings, lm_heads, and proj_in from the ORIGINAL safetensors."""
    print(f"  [Step 13/15] Exporting numpy artifacts to {output_dir}")

    # -- 13a: Codec embeddings (15 tables) --
    embeds_path = output_dir / "code_predictor_embeds.npz"
    emb_arrays: dict[str, np.ndarray] = {}
    for j in range(_NUM_AUX_GROUPS):
        key = f"model.codec_embedding.{j}.weight"
        if key not in cp_weights:
            raise KeyError(f"Missing codec embedding weight: {key}")
        emb_arrays[f"emb_{j}"] = cp_weights[key].float().numpy()
    np.savez(str(embeds_path), **emb_arrays)
    shape0 = emb_arrays["emb_0"].shape
    print(
        f"  [Step 13/15] Saved {_NUM_AUX_GROUPS} codec embedding tables "
        f"({list(shape0)}) -> {embeds_path.name}"
    )

    # -- 13b: LM heads (15 matrices) --
    # These are the ACTUAL lm_head weights from the original checkpoint,
    # NOT the identity matrix used in the exported model.
    heads_path = output_dir / "code_predictor_lm_heads.npz"
    head_arrays: dict[str, np.ndarray] = {}
    for j in range(_NUM_AUX_GROUPS):
        key = f"lm_head.{j}.weight"
        if key not in cp_weights:
            raise KeyError(f"Missing lm_head weight: {key}")
        head_arrays[f"head_{j}"] = cp_weights[key].float().numpy()
    np.savez(str(heads_path), **head_arrays)
    shape0 = head_arrays["head_0"].shape
    print(
        f"  [Step 13/15] Saved {_NUM_AUX_GROUPS} lm_head matrices "
        f"({list(shape0)}) -> {heads_path.name}"
    )

    # -- 13c: Projection in (small_to_mtp_projection) --
    proj_path = output_dir / "code_predictor_proj_in.npz"
    weight_key = "small_to_mtp_projection.weight"
    bias_key = "small_to_mtp_projection.bias"
    proj_arrays: dict[str, np.ndarray] = {}

    if weight_key in cp_weights:
        proj_arrays["weight"] = cp_weights[weight_key].float().numpy()
        if bias_key in cp_weights:
            proj_arrays["bias"] = cp_weights[bias_key].float().numpy()
        proj_arrays["is_identity"] = np.array(0, dtype=np.int32)
        print(f"  [Step 13/15] Proj-in weight: {list(proj_arrays['weight'].shape)}")
        if "bias" in proj_arrays:
            print(f"  [Step 13/15] Proj-in bias: {list(proj_arrays['bias'].shape)}")
    else:
        # talker_hidden_size == cp_hidden_size -> identity, no weights needed
        proj_arrays["is_identity"] = np.array(1, dtype=np.int32)
        print(
            f"  [Step 13/15] Proj-in: identity "
            f"(talker_hidden={_TALKER_HIDDEN_SIZE} == cp_hidden={_CP_HIDDEN_SIZE})"
        )

    np.savez(str(proj_path), **proj_arrays)
    print(f"  [Step 13/15] Saved proj_in -> {proj_path.name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_code_predictor(
    hf_model_dir: Path,
    models_dir: Path,
    skip_existing: bool = False,
) -> None:
    """Run all code predictor export steps 10-13.

    Args:
        hf_model_dir: Path to the HF model directory containing model.safetensors
        models_dir: Output root directory for exported models
        skip_existing: If True, skip steps whose outputs already exist
    """
    final_dir = models_dir / "cp_stateful_embeds"
    stateful_dir = models_dir / "cp_stateful"

    expected_files = [
        final_dir / "openvino_model.xml",
        final_dir / "code_predictor_embeds.npz",
        final_dir / "code_predictor_lm_heads.npz",
        final_dir / "code_predictor_proj_in.npz",
    ]

    # Fast path: everything already exists
    if skip_existing and all(f.exists() for f in expected_files):
        print("  [Steps 10-13] Code Predictor already exported, skipping")
        return

    safetensors_path = hf_model_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Safetensors not found: {safetensors_path}"
        )

    # Step 10: Load safetensors and map weights
    cp_weights, decoder_sd = _step10_load_and_map(safetensors_path)

    # Step 11: Create config, standalone model, export stateful OV
    if skip_existing and (stateful_dir / "openvino_model.xml").exists():
        print(
            "  [Step 11/15] Stateful model already exists at "
            f"{stateful_dir}, skipping export"
        )
    else:
        _step11_config_and_export(decoder_sd, models_dir, hf_model_dir)
    del decoder_sd

    # Step 12: IR surgery
    if skip_existing and (final_dir / "openvino_model.xml").exists():
        print(
            "  [Step 12/15] Surgered model already exists at "
            f"{final_dir}, skipping"
        )
    else:
        _step12_ir_surgery(models_dir)

    # Step 13: Export numpy artifacts (always runs -- fast and idempotent)
    _step13_export_numpy(cp_weights, final_dir)
    del cp_weights

    print("  [Steps 10-13] Code Predictor export complete")
