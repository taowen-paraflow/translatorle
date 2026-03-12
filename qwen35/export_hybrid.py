"""Hybrid GPU+NPU subgraph export for Qwen3.5.

Exports 13 independent IR subgraphs for mixed-device inference:
  - 6 GDN blocks (3 layers each) -> GPU  (contain Loop nodes, need FP32)
  - 6 Attention blocks (1 layer each) -> NPU  (standard SDPA, FP16 ok)
  - 1 Head block (norm + lm_head) -> GPU

All state is explicit I/O (no ReadValue/Assign stateful variables).

Export (qwen35 venv, needs transformers>=5):
  powershell.exe -Command 'cd C:\\Apps\\translatorle; uv run --project qwen35 python -m qwen35.export_hybrid'
"""

import logging
import shutil
import types
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM

import openvino as ov
from openvino import Dimension, PartialShape, Symbol
from openvino.frontend.pytorch import ConversionExtension, ModuleExtension

from .export import (
    RecurrentAttentionCell,
    _classify_layers,
    _copy_tokenizer_and_config,
    ov_causal_conv1d,
    patched_recurrent_gated_delta_rule,
    qwen3_5_gated_delta_net_forward,
)
from .ov_ops import convert_recurrent_attention_cell

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal cache for GDN layers (explicit state, no HF DynamicCache)
# ---------------------------------------------------------------------------

class MinimalCacheParams:
    """Lightweight cache object for GDN forward with explicit state lists."""

    def __init__(self, conv_states, recurrent_states, linear_attn_mapping):
        self.conv_states = conv_states
        self.recurrent_states = recurrent_states
        self.linear_attn_mapping = linear_attn_mapping


# ---------------------------------------------------------------------------
# KV cache for attention layers (same as Phase 1 test_single_attn_npu.py)
# ---------------------------------------------------------------------------

class KVCache:
    """Single-layer KV cache that traces cleanly through ov.convert_model."""

    def __init__(self, k, v):
        self.key_cache = [k]
        self.value_cache = [v]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        v = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v
        return k, v

    def get_seq_length(self, layer_idx=0):
        return self.key_cache[layer_idx].shape[2]


# ---------------------------------------------------------------------------
# Wrapper modules for subgraph export
# ---------------------------------------------------------------------------

class GDNBlockWrapper(nn.Module):
    """Wraps 3 GDN layers with explicit conv/recurrent state I/O."""

    def __init__(self, layers: nn.ModuleList, layer_indices: List[int]):
        super().__init__()
        self.layers = layers
        self.layer_indices = layer_indices

    def forward(
        self,
        hidden_states,
        attention_mask,
        conv_state_0, recurrent_state_0,
        conv_state_1, recurrent_state_1,
        conv_state_2, recurrent_state_2,
    ):
        conv_states = [conv_state_0, conv_state_1, conv_state_2]
        recurrent_states = [recurrent_state_0, recurrent_state_1, recurrent_state_2]
        linear_attn_mapping = {idx: i for i, idx in enumerate(self.layer_indices)}
        cache = MinimalCacheParams(conv_states, recurrent_states, linear_attn_mapping)

        for layer in self.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states = layer.linear_attn(
                hidden_states, cache_params=cache, attention_mask=attention_mask,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return (
            hidden_states,
            cache.conv_states[0], cache.recurrent_states[0],
            cache.conv_states[1], cache.recurrent_states[1],
            cache.conv_states[2], cache.recurrent_states[2],
        )


class AttnBlockWrapper(nn.Module):
    """Wraps 1 attention layer with explicit KV cache I/O.

    Calls the full DecoderLayer.forward() which handles input_layernorm,
    self_attn (with attention_mask), residual, post_attn_layernorm, MLP.
    Same approach as the working Phase 1 test_single_attn_npu.py.
    """

    def __init__(self, layer, rotary_emb):
        super().__init__()
        self.layer = layer
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states, position_ids, key_cache, value_cache, attention_mask):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cache = KVCache(key_cache, value_cache)

        # Call full DecoderLayer forward (handles layernorm, attn, MLP, residuals)
        out = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
            past_key_values=cache,
            use_cache=True,
        )

        return out[0], cache.key_cache[0], cache.value_cache[0]


class HeadWrapper(nn.Module):
    """Wraps final_norm + lm_head."""

    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


# ---------------------------------------------------------------------------
# GDN patching (apply to each GDN layer before tracing)
# ---------------------------------------------------------------------------

def _patch_gdn_layers(layers):
    """Patch GDN layers with RecurrentAttentionCell for OV Loop conversion."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    for layer in layers:
        if not (hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet)):
            continue
        gdn = layer.linear_attn
        gdn.forward = types.MethodType(qwen3_5_gated_delta_net_forward, gdn)
        gdn.recurrent_gated_delta_rule = patched_recurrent_gated_delta_rule
        gdn.recurrent_attention_cell = RecurrentAttentionCell()


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def _make_dynamic_shapes_gdn(ov_model, in_names, out_names):
    """Set dynamic batch/seq dims with shared Symbols on GDN block I/O."""
    batch_sym = Symbol()
    seq_sym = Symbol()

    for i, name in enumerate(in_names):
        ps = ov_model.inputs[i].partial_shape
        if name in ("hidden_states", "attention_mask"):
            b_dim = Dimension(-1)
            b_dim.set_symbol(batch_sym)
            s_dim = Dimension(-1)
            s_dim.set_symbol(seq_sym)
            ps[0] = b_dim
            ps[1] = s_dim
        elif name.startswith("conv_state") or name.startswith("recurrent_state"):
            b_dim = Dimension(-1)
            b_dim.set_symbol(batch_sym)
            ps[0] = b_dim
        ov_model.inputs[i].get_node().set_partial_shape(ps)
        ov_model.inputs[i].get_tensor().set_names({name})

    for i, name in enumerate(out_names):
        ov_model.outputs[i].get_tensor().set_names({name})

    ov_model.validate_nodes_and_infer_types()


def _make_dynamic_shapes_attn(ov_model, in_names, out_names):
    """Set dynamic batch/seq/past dims with shared Symbols on Attn block I/O."""
    batch_sym = Symbol()
    seq_sym = Symbol()
    past_sym = Symbol()

    for i, name in enumerate(in_names):
        ps = ov_model.inputs[i].partial_shape
        if name == "in_hidden":
            b = Dimension(-1); b.set_symbol(batch_sym)
            s = Dimension(-1); s.set_symbol(seq_sym)
            ps[0] = b; ps[1] = s
        elif name == "in_position_ids":
            b = Dimension(-1); b.set_symbol(batch_sym)
            s = Dimension(-1); s.set_symbol(seq_sym)
            ps[1] = b; ps[2] = s
        elif name in ("in_key_cache", "in_value_cache"):
            b = Dimension(-1); b.set_symbol(batch_sym)
            p = Dimension(-1); p.set_symbol(past_sym)
            ps[0] = b; ps[2] = p
        elif name == "in_attention_mask":
            # Shape: [B, 1, query_seq, key_seq] where key_seq = past + query
            # key_seq is an independent dynamic dim (can't express sum of symbols)
            b = Dimension(-1); b.set_symbol(batch_sym)
            s = Dimension(-1); s.set_symbol(seq_sym)
            ks = Dimension(-1)  # key_seq: independent dynamic dim
            ps[0] = b; ps[2] = s; ps[3] = ks
        ov_model.inputs[i].get_node().set_partial_shape(ps)
        ov_model.inputs[i].get_tensor().set_names({name})

    for i, name in enumerate(out_names):
        ov_model.outputs[i].get_tensor().set_names({name})

    ov_model.validate_nodes_and_infer_types()


def export_gdn_block(layers, layer_indices, group_idx, text_cfg, output_dir):
    """Export a GDN block (3 layers) to OpenVINO IR."""
    logger.info("Exporting GDN block %d (layers %s) ...", group_idx, layer_indices)

    wrapper = GDNBlockWrapper(nn.ModuleList(layers), layer_indices)
    _patch_gdn_layers(wrapper.layers)
    wrapper.eval()

    # Dimensions
    hidden_size = text_cfg.hidden_size
    conv_dim = (
        text_cfg.linear_num_key_heads * text_cfg.linear_key_head_dim * 2
        + text_cfg.linear_num_value_heads * text_cfg.linear_value_head_dim
    )
    conv_kernel = text_cfg.linear_conv_kernel_dim
    num_v_heads = text_cfg.linear_num_value_heads
    k_head_dim = text_cfg.linear_key_head_dim
    v_head_dim = text_cfg.linear_value_head_dim

    # B=2, S=2 for proper Loop tracing with dynamic shapes
    B, S = 2, 2
    dummy = {
        "hidden_states": torch.zeros(B, S, hidden_size),
        "attention_mask": torch.ones(B, S, dtype=torch.int64),
        "conv_state_0": torch.zeros(B, conv_dim, conv_kernel),
        "recurrent_state_0": torch.zeros(B, num_v_heads, k_head_dim, v_head_dim),
        "conv_state_1": torch.zeros(B, conv_dim, conv_kernel),
        "recurrent_state_1": torch.zeros(B, num_v_heads, k_head_dim, v_head_dim),
        "conv_state_2": torch.zeros(B, conv_dim, conv_kernel),
        "recurrent_state_2": torch.zeros(B, num_v_heads, k_head_dim, v_head_dim),
    }

    extensions = [
        ModuleExtension(RecurrentAttentionCell, "RecurrentAttentionCellOp"),
        ConversionExtension("RecurrentAttentionCellOp", convert_recurrent_attention_cell),
    ]

    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=dummy, extension=extensions)

    # Name I/O with unique prefixes to avoid name collisions
    in_names = [
        "in_hidden", "in_mask",
        "in_conv0", "in_rec0",
        "in_conv1", "in_rec1",
        "in_conv2", "in_rec2",
    ]
    out_names = [
        "out_hidden",
        "out_conv0", "out_rec0",
        "out_conv1", "out_rec1",
        "out_conv2", "out_rec2",
    ]
    _make_dynamic_shapes_gdn(ov_model, in_names, out_names)

    path = Path(output_dir) / f"gdn_block_{group_idx}.xml"
    ov.save_model(ov_model, str(path), compress_to_fp16=True)
    logger.info("  Saved %s (%d ops)", path, len(list(ov_model.get_ops())))


def export_attn_block(layer, rotary_emb, group_idx, text_cfg, output_dir):
    """Export an attention block (1 layer) to OpenVINO IR."""
    attn_layer_idx = layer.self_attn.layer_idx
    logger.info("Exporting Attn block %d (layer %d) ...", group_idx, attn_layer_idx)

    # Override layer_idx to 0 for single-entry KV cache
    orig_idx = layer.self_attn.layer_idx
    layer.self_attn.layer_idx = 0

    wrapper = AttnBlockWrapper(layer, rotary_emb)
    wrapper.eval()

    hidden_size = text_cfg.hidden_size
    num_kv_heads = text_cfg.num_key_value_heads
    head_dim = getattr(text_cfg, "head_dim", hidden_size // text_cfg.num_attention_heads)

    # B=1, S=1, P=2 (decode step)
    B, S, P = 1, 1, 2
    dummy = {
        "hidden_states": torch.randn(B, S, hidden_size),
        "position_ids": torch.full((3, B, S), P, dtype=torch.int64),
        "key_cache": torch.randn(B, num_kv_heads, P, head_dim) * 0.01,
        "value_cache": torch.randn(B, num_kv_heads, P, head_dim) * 0.01,
        # 4D causal mask: [B, 1, query_seq, key_seq] where key_seq = past + query
        # 0.0 = attend, large negative = ignore
        "attention_mask": torch.zeros(B, 1, S, P + S, dtype=torch.float32),
    }

    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=dummy)

    # Use in_/out_ prefixes to avoid OV tensor sharing issues
    in_names = ["in_hidden", "in_position_ids", "in_key_cache", "in_value_cache", "in_attention_mask"]
    out_names = ["out_hidden", "out_key_cache", "out_value_cache"]
    _make_dynamic_shapes_attn(ov_model, in_names, out_names)

    # Verify no Loop nodes
    loops = [op for op in ov_model.get_ops() if op.get_type_name() in ("Loop", "TensorIterator")]
    if loops:
        logger.warning("  WARNING: Found %d Loop ops in attention block %d!", len(loops), group_idx)

    path = Path(output_dir) / f"attn_block_{group_idx}.xml"
    ov.save_model(ov_model, str(path), compress_to_fp16=True)
    logger.info("  Saved %s (%d ops)", path, len(list(ov_model.get_ops())))

    # Restore original layer_idx
    layer.self_attn.layer_idx = orig_idx


def export_head(norm, lm_head, text_cfg, output_dir):
    """Export head (norm + lm_head) to OpenVINO IR."""
    logger.info("Exporting Head block ...")

    wrapper = HeadWrapper(norm, lm_head)
    wrapper.eval()

    hidden_size = text_cfg.hidden_size
    B, S = 1, 1
    dummy = {"hidden_states": torch.randn(B, S, hidden_size)}

    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=dummy)

    # Dynamic batch + seq
    batch_sym = Symbol()
    seq_sym = Symbol()
    ps = ov_model.inputs[0].partial_shape
    b = Dimension(-1); b.set_symbol(batch_sym)
    s = Dimension(-1); s.set_symbol(seq_sym)
    ps[0] = b; ps[1] = s
    ov_model.inputs[0].get_node().set_partial_shape(ps)
    ov_model.inputs[0].get_tensor().set_names({"hidden_states"})
    ov_model.outputs[0].get_tensor().set_names({"logits"})
    ov_model.validate_nodes_and_infer_types()

    path = Path(output_dir) / "head.xml"
    ov.save_model(ov_model, str(path), compress_to_fp16=True)
    logger.info("  Saved %s (%d ops)", path, len(list(ov_model.get_ops())))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def export_hybrid_subgraphs(model_dir: str, output_dir: str):
    """Export 13 independent subgraph IRs for hybrid GPU+NPU inference.

    Args:
        model_dir: HuggingFace model path or local directory.
        output_dir: Directory for output IR files.
    """
    logger.info("Loading PyTorch model from %s ...", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()

    text_cfg = getattr(model.config, "text_config", model.config)
    layer_types = text_cfg.layer_types

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Walk layers, accumulating GDN layers into blocks
    group_idx = 0
    gdn_layers = []
    gdn_indices = []

    for i, layer in enumerate(model.model.layers):
        if layer_types[i] == "linear_attention":
            gdn_layers.append(layer)
            gdn_indices.append(i)
        else:
            # Export accumulated GDN block
            if gdn_layers:
                export_gdn_block(gdn_layers, gdn_indices, group_idx, text_cfg, output_dir)
                gdn_layers = []
                gdn_indices = []

            # Export attention block
            export_attn_block(layer, model.model.rotary_emb, group_idx, text_cfg, output_dir)
            group_idx += 1

    # Trailing GDN layers (shouldn't happen for 0.8B but handle generically)
    if gdn_layers:
        export_gdn_block(gdn_layers, gdn_indices, group_idx, text_cfg, output_dir)

    # Export head
    export_head(model.model.norm, model.lm_head, text_cfg, output_dir)

    # Save embedding table
    embed = model.model.embed_tokens.weight.detach().cpu().numpy()
    embed_path = output_path / "embed_tokens.npy"
    np.save(str(embed_path), embed.astype(np.float16))
    logger.info("Saved embed_tokens.npy: shape=%s, dtype=float16", embed.shape)

    # Copy tokenizer and config
    model_path = Path(model_dir)
    if model_path.exists():
        _copy_tokenizer_and_config(model_path, output_path)

    # Summary
    logger.info("=== Export complete: %s ===", output_path)
    logger.info("  GDN blocks:  %d", group_idx)
    logger.info("  Attn blocks: %d", group_idx)
    logger.info("  Head block:  1")
    logger.info("  Total:       %d subgraphs", group_idx * 2 + 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Export Qwen3.5 hybrid GPU+NPU subgraphs")
    parser.add_argument(
        "--model-dir", default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", default="models/qwen35/Qwen3.5-0.8B-hybrid",
        help="Output directory for IR files",
    )
    args = parser.parse_args()

    export_hybrid_subgraphs(args.model_dir, args.output_dir)
