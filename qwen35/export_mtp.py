"""Export MTP (Multi-Token Prediction) block for Qwen3.5 speculative decoding.

The MTP block enables speculative decoding by predicting the next+1 token:
  - Takes main model's last hidden state + embedding of predicted token
  - Processes through: 2xRMSNorm -> concat -> Linear -> DecoderLayer -> RMSNorm
  - Output hidden can be fed to the shared lm_head (head.xml)

The MTP block does NOT include lm_head or embed_tokens -- those are
shared with the main model (head.xml / embed_tokens.npy).

The MTP attention layer has its OWN KV cache, separate from the main model's
6 attention layers. KV I/O uses the same FixedKVCache + ScatterUpdate-3
pattern as the main attention blocks for GPU/NPU compatibility.

Export (qwen35 venv, needs transformers>=5):
  powershell.exe -Command 'cd C:\\Apps\\translatorle; uv run --project qwen35 python -m qwen35.export_mtp'
"""

import glob
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM

import openvino as ov
from openvino import Dimension, PartialShape, Symbol
from openvino.frontend.pytorch import ConversionExtension, ModuleExtension

from .export_hybrid import (
    FixedKVCache,
    KVCacheScatterUpdate,
    MAX_CACHE_LEN,
)
from .ov_ops import convert_kv_cache_scatter_update

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MTP module wrapper for export
# ---------------------------------------------------------------------------

class MTPBlockWrapper(nn.Module):
    """Wraps the full MTP forward pass for OpenVINO export.

    Forward: 2xRMSNorm -> concat -> Linear -> DecoderLayer -> RMSNorm

    Internally uses FixedKVCache + KVCacheScatterUpdate for the MTP
    attention layer's own KV cache, matching AttnBlockWrapper's pattern.
    """

    def __init__(self, pre_fc_norm_embedding, pre_fc_norm_hidden, fc,
                 decoder_layer, final_norm, rotary_emb):
        super().__init__()
        self.pre_fc_norm_embedding = pre_fc_norm_embedding
        self.pre_fc_norm_hidden = pre_fc_norm_hidden
        self.fc = fc
        self.decoder_layer = decoder_layer
        self.final_norm = final_norm
        self.rotary_emb = rotary_emb
        if FixedKVCache.update_method == "scatter_update_ext":
            self.kv_scatter = KVCacheScatterUpdate()
        else:
            self.kv_scatter = None

    def forward(self, hidden_states, input_embeds, position_ids,
                key_cache, value_cache, cache_position, attention_mask):
        """MTP block forward pass.

        Args:
            hidden_states: [B, S, hidden_size] main model's last hidden state
            input_embeds:  [B, S, hidden_size] embed_tokens(predicted_token)
            position_ids:  [3, B, S] mRoPE position IDs
            key_cache:     [B, H, MAX_CACHE_LEN, D] MTP KV cache (key)
            value_cache:   [B, H, MAX_CACHE_LEN, D] MTP KV cache (value)
            cache_position: [S] int64 write positions in cache
            attention_mask: [B, 1, S, MAX_CACHE_LEN] causal mask

        Returns:
            (output_hidden [B, S, hidden_size], key_cache, value_cache)
        """
        # 1. Normalize and project: concat(RMSNorm(emb), RMSNorm(hid)) -> Linear
        emb = self.pre_fc_norm_embedding(input_embeds)
        hid = self.pre_fc_norm_hidden(hidden_states)
        x = self.fc(torch.cat([emb, hid], dim=-1))

        # 2. Compute RoPE embeddings
        cos, sin = self.rotary_emb(x, position_ids)

        # 3. Run decoder layer with KV cache
        cache = FixedKVCache(key_cache, value_cache, cache_position, self.kv_scatter)
        out = self.decoder_layer(
            x,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
            past_key_values=cache,
            use_cache=True,
        )

        # 4. Final norm (output goes to shared lm_head)
        x = self.final_norm(out[0])

        return x, cache.key_cache[0], cache.value_cache[0]


# ---------------------------------------------------------------------------
# Weight loading from safetensors
# ---------------------------------------------------------------------------

# Map from safetensors key -> MTPBlockWrapper state_dict key
_MTP_WEIGHT_MAP = {
    "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
    "mtp.fc.weight": "fc.weight",
    "mtp.layers.0.input_layernorm.weight": "decoder_layer.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight": "decoder_layer.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "decoder_layer.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "decoder_layer.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "decoder_layer.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "decoder_layer.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "decoder_layer.self_attn.k_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight": "decoder_layer.post_attention_layernorm.weight",
    "mtp.layers.0.mlp.gate_proj.weight": "decoder_layer.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight": "decoder_layer.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight": "decoder_layer.mlp.down_proj.weight",
    "mtp.norm.weight": "final_norm.weight",
}


def _load_mtp_weights(hf_model_id: str, wrapper: MTPBlockWrapper) -> None:
    """Load MTP weights from safetensors into the wrapper.

    Uses huggingface_hub to resolve the model snapshot directory, then
    scans all .safetensors files for keys matching the MTP weight map.
    """
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(hf_model_id)
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    state = {}
    for sf_path in safetensors_files:
        with safe_open(sf_path, framework="pt") as f:
            for key in f.keys():
                if key in _MTP_WEIGHT_MAP:
                    state[_MTP_WEIGHT_MAP[key]] = f.get_tensor(key).to(torch.float32)

    if len(state) != len(_MTP_WEIGHT_MAP):
        loaded = set(state.keys())
        expected = set(_MTP_WEIGHT_MAP.values())
        missing = expected - loaded
        raise RuntimeError(f"Missing MTP weights in safetensors: {missing}")

    # Load into wrapper (strict=False: rotary_emb and kv_scatter have no learned weights)
    missing_keys, unexpected_keys = wrapper.load_state_dict(state, strict=False)
    non_infra = [k for k in missing_keys
                 if not k.startswith("rotary_emb.") and k != "kv_scatter"]
    if non_infra:
        logger.warning("Missing MTP weights after load: %s", non_infra)
    logger.info("Loaded %d MTP weight tensors", len(state))


# ---------------------------------------------------------------------------
# Dynamic shapes and I/O naming
# ---------------------------------------------------------------------------

def _make_dynamic_shapes_mtp(ov_model, in_names, out_names):
    """Set dynamic batch/seq dims on MTP block I/O.

    KV cache seq dim (MAX_CACHE_LEN) stays STATIC for NPU compatibility.
    Same approach as _make_dynamic_shapes_attn in export_hybrid.py.
    """
    batch_sym = Symbol()
    seq_sym = Symbol()

    for i, name in enumerate(in_names):
        ps = ov_model.inputs[i].partial_shape
        if name in ("in_hidden", "in_embeds"):
            b = Dimension(-1); b.set_symbol(batch_sym)
            s = Dimension(-1); s.set_symbol(seq_sym)
            ps[0] = b; ps[1] = s
        elif name == "in_position_ids":
            b = Dimension(-1); b.set_symbol(batch_sym)
            s = Dimension(-1); s.set_symbol(seq_sym)
            ps[1] = b; ps[2] = s
        elif name in ("in_mtp_key_cache", "in_mtp_value_cache"):
            b = Dimension(-1); b.set_symbol(batch_sym)
            ps[0] = b
            # ps[2] = MAX_CACHE_LEN -- STATIC, don't change
        elif name == "in_cache_position":
            s = Dimension(-1); s.set_symbol(seq_sym)
            ps[0] = s
        elif name == "in_attention_mask":
            # Shape: [B, 1, query_seq, MAX_CACHE_LEN] -- cache dim is STATIC
            b = Dimension(-1); b.set_symbol(batch_sym)
            s = Dimension(-1); s.set_symbol(seq_sym)
            ps[0] = b; ps[2] = s
            # ps[3] = MAX_CACHE_LEN -- STATIC, don't change
        ov_model.inputs[i].get_node().set_partial_shape(ps)
        ov_model.inputs[i].get_tensor().set_names({name})

    for i, name in enumerate(out_names):
        ov_model.outputs[i].get_tensor().set_names({name})

    ov_model.validate_nodes_and_infer_types()


# ---------------------------------------------------------------------------
# Export entry point
# ---------------------------------------------------------------------------

def export_mtp_block(
    hf_model_id: str,
    output_dir: Path,
    max_cache_len: int = MAX_CACHE_LEN,
    model=None,
) -> None:
    """Export MTP block as a single OpenVINO IR file (mtp_block.xml/bin).

    The exported IR has explicit KV cache I/O (not stateful). At inference
    time, apply_make_stateful_transformation can convert it to stateful
    (same as the main attention blocks).

    KV I/O names are namespaced with "mtp_" prefix to avoid conflicts with
    the main model's attention KV caches when MakeStateful is applied.

    Args:
        hf_model_id: HuggingFace model ID or local path (e.g. "Qwen/Qwen3.5-0.8B").
        output_dir: Directory for output IR files.
        max_cache_len: Fixed KV cache sequence length (default: MAX_CACHE_LEN).
        model: Optional pre-loaded HF model to avoid loading twice.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model for config and rotary_emb (or reuse provided model)
    if model is None:
        logger.info("Loading model for MTP export: %s", hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id, torch_dtype=torch.float32, trust_remote_code=True,
        )
        model.eval()

    text_cfg = getattr(model.config, "text_config", model.config)

    # 2. Build MTP components
    norm_cls = type(model.model.norm)  # reuse the same RMSNorm class
    eps = text_cfg.rms_norm_eps
    hidden = text_cfg.hidden_size

    pre_fc_norm_embedding = norm_cls(hidden, eps=eps)
    pre_fc_norm_hidden = norm_cls(hidden, eps=eps)
    fc = nn.Linear(hidden * 2, hidden, bias=False)

    # MTP uses a full_attention decoder layer, NOT linear_attention (GDN).
    # Temporarily patch config.layer_types so layer_idx=0 creates self_attn.
    import copy
    mtp_cfg = copy.deepcopy(text_cfg)
    mtp_cfg.layer_types = ["full_attention"]
    decoder_layer = Qwen3_5DecoderLayer(mtp_cfg, layer_idx=0)
    final_norm = norm_cls(hidden, eps=eps)
    rotary_emb = model.model.rotary_emb

    logger.info("MTP DecoderLayer config: num_attention_heads=%d, num_key_value_heads=%d, "
                "head_dim=%d, intermediate_size=%d, hidden_size=%d",
                text_cfg.num_attention_heads, text_cfg.num_key_value_heads,
                text_cfg.head_dim, text_cfg.intermediate_size, text_cfg.hidden_size)

    wrapper = MTPBlockWrapper(
        pre_fc_norm_embedding, pre_fc_norm_hidden, fc,
        decoder_layer, final_norm, rotary_emb,
    )

    # 3. Load MTP weights from safetensors
    _load_mtp_weights(hf_model_id, wrapper)
    wrapper.eval()

    # 4. Prepare dummy inputs for tracing
    num_kv_heads = text_cfg.num_key_value_heads
    head_dim = getattr(text_cfg, "head_dim", hidden // text_cfg.num_attention_heads)

    B, S = 1, 1
    cache_pos = 5
    dummy = {
        "hidden_states": torch.randn(B, S, hidden),
        "input_embeds": torch.randn(B, S, hidden),
        "position_ids": torch.full((3, B, S), cache_pos, dtype=torch.int64),
        "key_cache": torch.randn(B, num_kv_heads, max_cache_len, head_dim) * 0.01,
        "value_cache": torch.randn(B, num_kv_heads, max_cache_len, head_dim) * 0.01,
        "cache_position": torch.tensor([cache_pos], dtype=torch.int64),
        # 4D mask: [B, 1, query_seq, MAX_CACHE_LEN]  0.0=attend, -65504=ignore
        "attention_mask": torch.zeros(B, 1, S, max_cache_len, dtype=torch.float32),
    }

    # 5. Convert to OpenVINO IR
    extensions = []
    if FixedKVCache.update_method == "scatter_update_ext":
        extensions.append(ModuleExtension(KVCacheScatterUpdate, "KVCacheScatterUpdateOp"))
        extensions.append(ConversionExtension("KVCacheScatterUpdateOp", convert_kv_cache_scatter_update))

    logger.info("Tracing MTP block (KV update method: %s) ...", FixedKVCache.update_method)
    with torch.no_grad():
        ov_model = ov.convert_model(
            wrapper, example_input=dummy,
            extension=extensions if extensions else None,
        )

    # 6. Name I/O and set dynamic shapes
    # KV cache names use "mtp_" prefix for MakeStateful namespacing
    in_names = [
        "in_hidden", "in_embeds", "in_position_ids",
        "in_mtp_key_cache", "in_mtp_value_cache",
        "in_cache_position", "in_attention_mask",
    ]
    out_names = ["out_hidden", "out_mtp_key_cache", "out_mtp_value_cache"]
    _make_dynamic_shapes_mtp(ov_model, in_names, out_names)

    # 7. Verify no Loop/TensorIterator ops (MTP is pure attention, should be clean)
    loops = [op for op in ov_model.get_ops()
             if op.get_type_name() in ("Loop", "TensorIterator")]
    total_ops = len(list(ov_model.get_ops()))
    if loops:
        logger.warning("WARNING: Found %d Loop ops in MTP block!", len(loops))
    else:
        logger.info("No Loop/TensorIterator ops (clean)")

    # Report scatter ops
    scatter_types = {}
    for op in ov_model.get_ops():
        name = op.get_type_name()
        if "Scatter" in name:
            scatter_types[name] = scatter_types.get(name, 0) + 1
    if scatter_types:
        logger.info("Scatter ops: %s", dict(scatter_types))

    # 8. Save with FP16 compression
    path = output_dir / "mtp_block.xml"
    ov.save_model(ov_model, str(path), compress_to_fp16=True)
    logger.info("Saved %s (%d ops)", path, total_ops)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Export Qwen3.5 MTP block to OpenVINO IR")
    parser.add_argument(
        "--model-dir", default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", default="models/qwen35/Qwen3.5-0.8B-hybrid",
        help="Output directory for IR files",
    )
    parser.add_argument(
        "--max-cache-len", type=int, default=MAX_CACHE_LEN,
        help=f"Fixed KV cache size (default: {MAX_CACHE_LEN})",
    )
    parser.add_argument(
        "--kv-update-method", default="scatter_update_ext",
        choices=["select", "index_copy", "scatter_elements", "scatter_nd", "scatter_update_ext"],
        help="KV cache update method (default: scatter_update_ext)",
    )
    args = parser.parse_args()

    FixedKVCache.update_method = args.kv_update_method
    export_mtp_block(args.model_dir, args.output_dir, args.max_cache_len)

    # Verify exported IR
    logger.info("=== Verifying exported IR ===")
    ir_path = Path(args.output_dir) / "mtp_block.xml"
    core = ov.Core()
    ir = core.read_model(str(ir_path))

    logger.info("Inputs:")
    for inp in ir.inputs:
        names = inp.get_tensor().get_names()
        logger.info("  %s: %s %s", names, inp.partial_shape, inp.get_element_type())

    logger.info("Outputs:")
    for out in ir.outputs:
        names = out.get_tensor().get_names()
        logger.info("  %s: %s %s", names, out.partial_shape, out.get_element_type())

    loops = [op for op in ir.get_ops() if op.get_type_name() in ("Loop", "TensorIterator")]
    if loops:
        logger.warning("FAIL: Found %d Loop/TensorIterator ops!", len(loops))
    else:
        logger.info("PASS: No Loop/TensorIterator ops")

    logger.info("Total ops: %d", len(list(ir.get_ops())))
