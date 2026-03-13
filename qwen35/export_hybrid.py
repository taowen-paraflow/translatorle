"""Hybrid GPU+NPU subgraph export for Qwen3.5.

Exports independent IR subgraphs for mixed-device inference:
  - 6 GDN blocks (3 layers each) -> GPU  (contain Loop nodes, need FP32)
  - 6 GDN prefill blocks (3 layers each) -> GPU  (chunkwise parallel, no Loop)
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
from .chunkwise_gdn import ChunkwiseRecurrentAttentionCell
from .ov_ops import convert_recurrent_attention_cell, convert_kv_cache_scatter_update

logger = logging.getLogger(__name__)

# Fixed KV cache size for NPU-compatible static shapes.
# All attn_block IRs use this as the sequence dimension for KV cache.
MAX_CACHE_LEN = 256


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
# Fixed-size KV cache for attention layers (ScatterUpdate instead of Concat)
# ---------------------------------------------------------------------------

class FixedKVCache:
    """Fixed-size KV cache for NPU-compatible static shapes.

    Unlike DynamicCache which uses torch.cat (growing output shape P -> P+S),
    this writes new KV at cache_position into a fixed-size buffer.
    Input and output shapes are always [B, H, MAX_CACHE_LEN, D].

    Supports multiple update methods to test which OV IR op is fastest on NPU:
      - 'select': torch.where -> Select op (element-wise conditional)
      - 'index_copy': torch.index_copy_ -> ScatterUpdate op (axis-based)
      - 'scatter_elements': torch.scatter -> ScatterElementsUpdate op
    """

    # Module-level setting, changed by CLI --kv-update-method
    update_method = "select"

    def __init__(self, k, v, cache_position, scatter_module=None):
        """
        Args:
            k: Key cache [B, H, MAX_CACHE_LEN, D]
            v: Value cache [B, H, MAX_CACHE_LEN, D]
            cache_position: Write position(s) [S] int64
            scatter_module: Optional KVCacheScatterUpdate instance for ext method
        """
        self.key_cache = [k]
        self.value_cache = [v]
        self._cache_position = cache_position
        self._scatter_module = scatter_module

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Write new KV at cache_position, return full buffer."""
        method = FixedKVCache.update_method
        if method == "select":
            return self._update_select(key_states, value_states, layer_idx)
        elif method == "index_copy":
            return self._update_index_copy(key_states, value_states, layer_idx)
        elif method == "scatter_elements":
            return self._update_scatter_elements(key_states, value_states, layer_idx)
        elif method == "scatter_nd":
            return self._update_scatter_nd(key_states, value_states, layer_idx)
        elif method == "scatter_update_ext":
            return self._update_scatter_update_ext(key_states, value_states, layer_idx)
        else:
            raise ValueError(f"Unknown KV update method: {method}")

    def _update_select(self, key_states, value_states, layer_idx):
        """torch.where -> Select op in OV IR."""
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        seq_dim = k.shape[2]
        indices = torch.arange(seq_dim, device=k.device)
        mask = (indices == self._cache_position.view(-1, 1))
        mask = mask.any(dim=0)
        mask = mask.view(1, 1, seq_dim, 1)

        new_k_expanded = key_states.expand(-1, -1, 1, -1).expand_as(k)
        new_v_expanded = value_states.expand(-1, -1, 1, -1).expand_as(v)

        k = torch.where(mask, new_k_expanded, k)
        v = torch.where(mask, new_v_expanded, v)

        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v
        return k, v

    def _update_index_copy(self, key_states, value_states, layer_idx):
        """torch.index_copy_ -> ScatterUpdate op in OV IR (axis-based scatter)."""
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        # index_copy_(dim, index_1d, source): write source slices at index positions along dim
        # k: [B, H, MAX_CACHE_LEN, D], key_states: [B, H, S, D], cache_position: [S]
        k = k.clone()
        v = v.clone()
        k.index_copy_(2, self._cache_position, key_states)
        v.index_copy_(2, self._cache_position, value_states)

        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v
        return k, v

    def _update_scatter_elements(self, key_states, value_states, layer_idx):
        """torch.scatter -> ScatterElementsUpdate op in OV IR."""
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        idx = self._cache_position.view(1, 1, -1, 1).expand_as(key_states)
        k = k.scatter(2, idx, key_states)
        v = v.scatter(2, idx, value_states)

        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v
        return k, v

    def _update_scatter_nd(self, key_states, value_states, layer_idx):
        """torch.index_put with 4D tensor indices -> ScatterNDUpdate op in OV IR."""
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        B, H, S, D = key_states.shape
        # Build full 4D index tensors so aten::index_put maps to ScatterNDUpdate
        batch_idx = torch.arange(B, device=k.device).view(-1, 1, 1, 1).expand(B, H, S, D)
        head_idx = torch.arange(H, device=k.device).view(1, -1, 1, 1).expand(B, H, S, D)
        seq_idx = self._cache_position.view(1, 1, -1, 1).expand(B, H, S, D)
        dim_idx = torch.arange(D, device=k.device).view(1, 1, 1, -1).expand(B, H, S, D)

        k = k.index_put((batch_idx, head_idx, seq_idx, dim_idx), key_states)
        v = v.index_put((batch_idx, head_idx, seq_idx, dim_idx), value_states)

        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v
        return k, v

    def _update_scatter_update_ext(self, key_states, value_states, layer_idx):
        """KVCacheScatterUpdate module -> ScatterUpdate-3 via ConversionExtension."""
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        k = self._scatter_module(k, key_states, self._cache_position)
        v = self._scatter_module(v, value_states, self._cache_position)

        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v
        return k, v

    def get_seq_length(self, layer_idx=0):
        return self.key_cache[layer_idx].shape[2]


class KVCacheScatterUpdate(nn.Module):
    """Custom module for KV cache scatter update.

    ModuleExtension intercepts this module during ov.convert_model and replaces
    it with the OV op built by the ConversionExtension callback.

    The forward() runs during tracing for shape inference only.
    """
    def forward(self, cache, new_kv, cache_position):
        # Fallback computation for tracing (shape inference).
        # ConversionExtension replaces this with ScatterUpdate-3 in the IR.
        idx = cache_position.view(1, 1, -1, 1).expand_as(new_kv)
        return cache.scatter(2, idx, new_kv)


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
    """Wraps 1 attention layer with fixed-size KV cache I/O.

    Uses FixedKVCache (scatter-update) so input and output KV shapes are
    always [B, H, MAX_CACHE_LEN, D]. This enables MakeStateful on NPU.

    Calls the full DecoderLayer.forward() which handles input_layernorm,
    self_attn (with attention_mask), residual, post_attn_layernorm, MLP.
    """

    def __init__(self, layer, rotary_emb):
        super().__init__()
        self.layer = layer
        self.rotary_emb = rotary_emb
        if FixedKVCache.update_method == "scatter_update_ext":
            self.kv_scatter = KVCacheScatterUpdate()
        else:
            self.kv_scatter = None

    def forward(self, hidden_states, position_ids, key_cache, value_cache, cache_position, attention_mask):
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cache = FixedKVCache(key_cache, value_cache, cache_position, self.kv_scatter)

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


def _patch_gdn_layers_chunkwise(layers):
    """Patch GDN layers with ChunkwiseRecurrentAttentionCell for parallel prefill.

    Uses WY representation instead of sequential Loop. All ops (MatMul, tril,
    cumsum, exp) are standard OV opset — no Loop node, no ModuleExtension needed.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    for layer in layers:
        if not (hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet)):
            continue
        gdn = layer.linear_attn
        gdn.forward = types.MethodType(qwen3_5_gated_delta_net_forward, gdn)
        gdn.recurrent_gated_delta_rule = patched_recurrent_gated_delta_rule
        gdn.recurrent_attention_cell = ChunkwiseRecurrentAttentionCell()


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
    """Set dynamic batch/seq dims on Attn block I/O.

    KV cache dim (MAX_CACHE_LEN) stays STATIC — this is the key change
    that enables NPU stateful mode (constant state shape).
    """
    batch_sym = Symbol()
    seq_sym = Symbol()

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
            ps[0] = b
            # ps[2] = MAX_CACHE_LEN — STATIC, don't change
        elif name == "in_cache_position":
            s = Dimension(-1); s.set_symbol(seq_sym)
            ps[0] = s
        elif name == "in_attention_mask":
            # Shape: [B, 1, query_seq, MAX_CACHE_LEN] — cache dim is STATIC
            b = Dimension(-1); b.set_symbol(batch_sym)
            s = Dimension(-1); s.set_symbol(seq_sym)
            ps[0] = b; ps[2] = s
            # ps[3] = MAX_CACHE_LEN — STATIC, don't change
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


def export_gdn_prefill_block(layers, layer_indices, group_idx, text_cfg, output_dir):
    """Export a GDN prefill block (3 layers) using chunkwise parallel algorithm.

    Unlike the Loop-based decode block, this uses parallel MatMul operations
    (WY representation). No Loop node — all operations are standard opset ops.
    The IR has explicit state I/O (not stateful).
    No ModuleExtension needed — OV traces ChunkwiseRecurrentAttentionCell directly.
    """
    logger.info("Exporting GDN prefill block %d (layers %s, chunkwise) ...", group_idx, layer_indices)

    wrapper = GDNBlockWrapper(nn.ModuleList(layers), layer_indices)
    _patch_gdn_layers_chunkwise(wrapper.layers)
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

    # B=2, S=2 for tracing (needs S>=2 for cumsum/matmul)
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

    # No extensions needed — ChunkwiseRecurrentAttentionCell traces directly
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=dummy)

    # Verify no Loop nodes
    loops = [op for op in ov_model.get_ops() if op.get_type_name() in ("Loop", "TensorIterator")]
    matmuls = sum(1 for op in ov_model.get_ops() if op.get_type_name() == "MatMul")
    if loops:
        logger.warning("  WARNING: Found %d Loop ops in chunkwise prefill block %d!", len(loops), group_idx)
    else:
        logger.info("  No Loop nodes (chunkwise), %d MatMul ops", matmuls)

    # Name I/O (same names as decode block for compatibility)
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

    path = Path(output_dir) / f"gdn_prefill_block_{group_idx}.xml"
    # Keep FP32: Neumann series (7 matrix squarings) needs FP32 precision.
    ov.save_model(ov_model, str(path), compress_to_fp16=False)
    logger.info("  Saved %s (%d ops)", path, len(list(ov_model.get_ops())))


def export_attn_block(layer, rotary_emb, group_idx, text_cfg, output_dir):
    """Export an attention block (1 layer) to OpenVINO IR.

    Uses FixedKVCache so the IR contains a scatter/select op instead of Concat.
    KV cache I/O shape is always [B, H, MAX_CACHE_LEN, D].
    The specific OV op depends on FixedKVCache.update_method.
    """
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

    # B=1, S=1, cache_pos=5 (arbitrary decode position for tracing)
    B, S = 1, 1
    cache_pos = 5
    dummy = {
        "hidden_states": torch.randn(B, S, hidden_size),
        "position_ids": torch.full((3, B, S), cache_pos, dtype=torch.int64),
        "key_cache": torch.randn(B, num_kv_heads, MAX_CACHE_LEN, head_dim) * 0.01,
        "value_cache": torch.randn(B, num_kv_heads, MAX_CACHE_LEN, head_dim) * 0.01,
        "cache_position": torch.tensor([cache_pos], dtype=torch.int64),
        # 4D mask: [B, 1, query_seq, MAX_CACHE_LEN]  0.0=attend, -65504=ignore
        "attention_mask": torch.zeros(B, 1, S, MAX_CACHE_LEN, dtype=torch.float32),
    }

    # Build extensions list for ConversionExtension-based KV update methods
    extensions = []
    if FixedKVCache.update_method == "scatter_update_ext":
        extensions.append(ModuleExtension(KVCacheScatterUpdate, "KVCacheScatterUpdateOp"))
        extensions.append(ConversionExtension("KVCacheScatterUpdateOp", convert_kv_cache_scatter_update))

    with torch.no_grad():
        ov_model = ov.convert_model(
            wrapper, example_input=dummy,
            extension=extensions if extensions else None,
        )

    in_names = ["in_hidden", "in_position_ids", "in_key_cache", "in_value_cache",
                "in_cache_position", "in_attention_mask"]
    out_names = ["out_hidden", "out_key_cache", "out_value_cache"]
    _make_dynamic_shapes_attn(ov_model, in_names, out_names)

    # Report KV update ops in the IR
    selects = [op for op in ov_model.get_ops() if op.get_type_name() == "Select"]
    scatter_types = {}
    for op in ov_model.get_ops():
        name = op.get_type_name()
        if "Scatter" in name:
            scatter_types[name] = scatter_types.get(name, 0) + 1
    concats = [op for op in ov_model.get_ops() if op.get_type_name() == "Concat"]
    logger.info("  KV update method: %s", FixedKVCache.update_method)
    logger.info("  Select ops: %d, Scatter ops: %s, Concat ops: %d",
                len(selects), dict(scatter_types) if scatter_types else 0, len(concats))

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
            # Export accumulated GDN block (decode Loop + prefill chunkwise)
            if gdn_layers:
                export_gdn_block(gdn_layers, gdn_indices, group_idx, text_cfg, output_dir)
                export_gdn_prefill_block(gdn_layers, gdn_indices, group_idx, text_cfg, output_dir)
                gdn_layers = []
                gdn_indices = []

            # Export attention block
            export_attn_block(layer, model.model.rotary_emb, group_idx, text_cfg, output_dir)
            group_idx += 1

    # Trailing GDN layers (shouldn't happen for 0.8B but handle generically)
    if gdn_layers:
        export_gdn_block(gdn_layers, gdn_indices, group_idx, text_cfg, output_dir)
        export_gdn_prefill_block(gdn_layers, gdn_indices, group_idx, text_cfg, output_dir)

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
    logger.info("  GDN blocks (decode Loop):  %d", group_idx)
    logger.info("  GDN blocks (prefill):      %d (chunkwise, no Loop)", group_idx)
    logger.info("  Attn blocks: %d", group_idx)
    logger.info("  Head block:  1")
    logger.info("  Total:       %d subgraphs", group_idx * 3 + 1)


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
    parser.add_argument(
        "--kv-update-method", default="select",
        choices=["select", "index_copy", "scatter_elements", "scatter_nd", "scatter_update_ext"],
        help="KV cache update method: select (torch.where), index_copy (torch.index_copy_), scatter_elements (torch.scatter), scatter_nd (torch.index_put), scatter_update_ext (ConversionExtension -> ScatterUpdate-3)",
    )
    args = parser.parse_args()

    FixedKVCache.update_method = args.kv_update_method
    logger.info("KV update method: %s", args.kv_update_method)
    export_hybrid_subgraphs(args.model_dir, args.output_dir)
