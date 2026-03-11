"""Standalone OpenVINO export pipeline for Qwen3.5 hybrid models.

Adapted from optimum-intel PR #1634 (Qwen3.5 GDN support).  Uses
ModuleExtension + ConversionExtension to convert the GDN recurrence into
an OpenVINO Loop node, enabling dynamic sequence length and batch prefill.

Previous version used TorchScriptPythonDecoder with seq_len=1, requiring
token-by-token prefill at inference time.  This version exports a model
that supports any seq_len (prefill the whole prompt in a single infer call).

This module has NO dependency on optimum / optimum-intel / optimum-onnx.
It requires only: torch, transformers, openvino, numpy, and our .stateful module.
"""

import logging
import shutil
import types
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM

import openvino as ov
from openvino import Dimension, PartialShape, Symbol
from openvino.frontend.pytorch import ConversionExtension, ModuleExtension
from openvino.utils.types import get_element_type

from .ov_ops import convert_recurrent_attention_cell
from .stateful import patch_stateful_hybrid_ssm, patch_stateful_kv_only

logger = logging.getLogger(__name__)

InputInfo = namedtuple("InputInfo", ["name", "shape", "type", "example"])

# Files to copy from the source model directory to the OV output directory
_TOKENIZER_AND_CONFIG_GLOBS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "added_tokens.json",
    "chat_template.json",
]

# Hybrid export: intermediates captured during forward tracing
_hybrid_intermediates: list = []


# ---------------------------------------------------------------------------
# Helper utilities (unchanged from original)
# ---------------------------------------------------------------------------

def _flattenize_inputs(inputs: List[Any]) -> List[Any]:
    """Recursively flatten lists/tuples, skipping None values."""
    flat = []
    for item in inputs:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            flat.extend(_flattenize_inputs(item))
        else:
            flat.append(item)
    return flat


def _remove_none(
    dummy_inputs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[str, List[str]]]]:
    """Remove None values from dummy-inputs dict."""

    def _strip_none(item):
        return type(item)([i for i in item if i is not None])

    upd_dummy: Dict[str, Any] = {}
    dict_dummy: List[Tuple[str, List[str]]] = []
    for k, v in dummy_inputs.items():
        if v is None:
            continue
        if isinstance(v, dict):
            dict_dummy.append((k, list(v.keys())))
            upd_dummy[k] = _strip_none(tuple(v.values()))
            continue
        if isinstance(v, (tuple, list)):
            upd_dummy[k] = _strip_none(v)
            continue
        upd_dummy[k] = v
    return upd_dummy, dict_dummy


def _get_input_info(
    ordered_input_names: List[str],
    inputs_spec: Dict[str, Dict[int, str]],
    flatten_inputs: List[torch.Tensor],
) -> List[InputInfo]:
    """Build InputInfo list with OV PartialShapes and dynamic Symbols."""
    name_to_symbol: Dict[str, Symbol] = {}
    input_info: List[InputInfo] = []

    for i, name in enumerate(ordered_input_names):
        example = flatten_inputs[i]
        if example.dtype == torch.bfloat16:
            ov_type = ov.Type.bf16
        else:
            ov_type = get_element_type(example.cpu().numpy().dtype)
        shape = PartialShape(example.shape)
        if name in inputs_spec:
            named_dims = inputs_spec[name]
            for idx, dim_name in named_dims.items():
                if dim_name in name_to_symbol:
                    symbol = name_to_symbol[dim_name]
                else:
                    symbol = Symbol()
                    name_to_symbol[dim_name] = symbol
                dim = Dimension(-1)
                dim.set_symbol(symbol)
                shape[idx] = dim
        input_info.append(InputInfo(name=name, shape=shape, type=ov_type, example=example))
    return input_info


def _clear_class_registry():
    """Remove TorchScript cached modules to avoid stale state."""
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------

def _classify_layers(text_cfg):
    """Return (linear_layer_indices, attention_layer_indices) from text_cfg."""
    layer_types = text_cfg.layer_types
    num_hidden = text_cfg.num_hidden_layers
    linear = [i for i in range(num_hidden) if layer_types[i] == "linear_attention"]
    attention = [i for i in range(num_hidden) if layer_types[i] == "full_attention"]
    return linear, attention


# ---------------------------------------------------------------------------
# GDN patching functions (from optimum-intel PR #1634)
# ---------------------------------------------------------------------------

def ov_causal_conv1d(conv_state, input_embeds, weight, bias):
    """CausalConv1D with cache, works for any seq_len.

    Prepends cached state to input, runs conv1d, and updates cache.
    This replaces both the old causal_conv1d_fn (prefill) and
    causal_conv1d_update (decode) with a single unified function.
    """
    _, hidden_size, seq_len = input_embeds.shape
    _, w_in_channels, _ = weight.shape
    state_len = conv_state.shape[-1]
    groups = hidden_size // w_in_channels

    input_embeds_new = torch.cat([conv_state, input_embeds], dim=-1).to(weight.dtype)
    conv_out = F.conv1d(input_embeds_new, weight, bias, padding=0, groups=groups)
    conv_out = conv_out[:, :, -seq_len:]

    new_conv_state = input_embeds_new[:, :, -state_len:]

    return conv_out, new_conv_state


class RecurrentAttentionCell(torch.nn.Module):
    """GDN recurrence as a traceable Module.

    This module implements the gated delta rule recurrence loop.
    During OV conversion, it gets replaced by ModuleExtension with
    ``RecurrentAttentionCellOp``, which ConversionExtension then converts
    to an OpenVINO Loop node (see ov_ops.py).

    At PyTorch level, this runs the actual for-loop (used during tracing
    to verify correctness).
    """

    def forward(
        self,
        query,                 # (B, H, T, D_k)
        key,                   # (B, H, T, D_k)
        value,                 # (B, H, T, D_v)
        g,                     # (B, H, T)
        beta,                  # (B, H, T)
        last_recurrent_state,  # (B, H, D_k, D_v)
    ):
        _, _, sequence_length, _ = key.shape
        core_attn_out = torch.zeros_like(value)

        for i in range(sequence_length):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

        # Single-output workaround for ModuleExtension (expects one output).
        output_cell = torch.cat([core_attn_out.flatten(), last_recurrent_state.flatten()], dim=0)
        return output_cell


def patched_recurrent_gated_delta_rule(
    self, query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    """Recurrent form of the gated delta rule, calling RecurrentAttentionCell.

    Replaces the original torch_recurrent_gated_delta_rule which uses a
    plain for-loop that gets expanded during trace.  This version delegates
    to self.recurrent_attention_cell (a RecurrentAttentionCell module) which
    gets intercepted by ModuleExtension and converted to an OV Loop.
    """
    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    output_cell = self.recurrent_attention_cell(
        query, key, value, g, beta, last_recurrent_state,
    )

    num_elems = value.numel()
    core_attn_out = output_cell[:num_elems].reshape(value.shape)
    last_recurrent_state = output_cell[num_elems:].reshape(last_recurrent_state.shape)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def patched_recurrent_gated_delta_rule_single_step(
    self, query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    """Single-step GDN recurrence for NPU export (no for-loop, no Loop node).

    Identical to patched_recurrent_gated_delta_rule but processes exactly
    one timestep inline, avoiding the RecurrentAttentionCell for-loop that
    becomes an OV Loop node.  Used when exporting for NPU (seq_len=1).
    """
    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    # --- Single-step computation (no for-loop) ---
    q_t = query[:, :, 0]       # [B, H, D_k]
    k_t = key[:, :, 0]         # [B, H, D_k]
    v_t = value[:, :, 0]       # [B, H, D_v]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
    beta_t = beta[:, :, 0].unsqueeze(-1)                 # [B, H, 1]

    last_recurrent_state = last_recurrent_state * g_t
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    core_attn_out = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    # Reshape to [B, H, 1, D_v] to match the expected output shape
    core_attn_out = core_attn_out.unsqueeze(2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def patched_recurrent_gated_delta_rule_hybrid_step(
    self, query, key, value, g, beta, initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    """Single-step GDN recurrence that also captures intermediates for CPU FP32 shadow state."""
    def l2norm(x, dim=-1, eps=1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    # --- Single-step computation ---
    q_t = query[:, :, 0]
    k_t = key[:, :, 0]
    v_t = value[:, :, 0]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
    beta_t = beta[:, :, 0].unsqueeze(-1)                  # [B, H, 1]

    last_recurrent_state = last_recurrent_state * g_t
    kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    core_attn_out = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.unsqueeze(2)

    # ---- HYBRID: capture intermediates for CPU FP32 state update ----
    # g_t: [B, H, 1, 1] (decay factor, already exp'd)
    # k_t: [B, H, D_k] (key after conv)
    # v_t: [B, H, D_v] (value after conv)
    # beta_t: [B, H, 1] (gate)
    _hybrid_intermediates.append((g_t, k_t, v_t, beta_t))

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def qwen3_5_gated_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    """Patched forward for Qwen3_5GatedDeltaNet layers.

    Uses ov_causal_conv1d for the conv1d cache and
    patched_recurrent_gated_delta_rule (which calls RecurrentAttentionCell)
    for the GDN recurrence.
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    # During torch.jit.trace with [1,1,D], tracer may squeeze to [1,D] (2D).
    # Robustly handle both 2D and 3D inputs.
    if hidden_states.ndim == 2:
        hidden_states = hidden_states.unsqueeze(0)
    batch_size, seq_len, _ = hidden_states.shape

    layer_idx = None
    recurrent_state = None
    if cache_params is not None:
        layer_idx = cache_params.linear_attn_mapping[self.layer_idx]
        conv_state = cache_params.conv_states[layer_idx]
        recurrent_state = cache_params.recurrent_states[layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if cache_params is not None:
        new_mixed_qkv, new_conv_state = ov_causal_conv1d(
            conv_state, mixed_qkv, self.conv1d.weight, self.conv1d.bias,
        )
        mixed_qkv = F.silu(new_mixed_qkv)
        cache_params.conv_states[layer_idx] = new_conv_state
    else:
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )
    query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
    key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
    value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
        self,
        query, key, value,
        g=g, beta=beta,
        initial_state=recurrent_state,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
    )

    if cache_params is not None:
        cache_params.recurrent_states[layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


# ---------------------------------------------------------------------------
# Dummy input generation (updated for dynamic seq_len)
# ---------------------------------------------------------------------------

def _generate_dummy_inputs(
    text_cfg,
    batch_size: int = 2,
    seq_len: int = 2,
    past_seq_len: int = 2,
) -> Dict[str, Any]:
    """Create dummy inputs for tracing with dynamic seq_len support.

    Returns a dict with keys: inputs_embeds, attention_mask, cache_params.
    cache_params is a list of tensors in grouped order:
      18 linear layers x (conv_state, recurrent_state) = 36 tensors
      6  attention layers x (key_cache, value_cache)   = 12 tensors
    """
    linear_indices, attention_indices = _classify_layers(text_cfg)
    num_linear = len(linear_indices)
    num_attention = len(attention_indices)

    linear_num_key_heads = text_cfg.linear_num_key_heads
    linear_key_head_dim = text_cfg.linear_key_head_dim
    linear_value_head_dim = text_cfg.linear_value_head_dim
    linear_num_value_heads = text_cfg.linear_num_value_heads
    linear_conv_kernel_dim = text_cfg.linear_conv_kernel_dim
    conv_dim = (
        linear_num_key_heads * linear_key_head_dim * 2
        + linear_num_value_heads * linear_value_head_dim
    )

    num_key_value_heads = text_cfg.num_key_value_heads
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)

    # Input embeddings and attention mask
    hidden_size = text_cfg.hidden_size
    inputs_embeds = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)

    # mRoPE position IDs: shape [3, batch_size, seq_len]
    # 3 dims = (temporal, height, width); for text-only all 3 are identical
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
    position_ids = position_ids.expand(3, batch_size, seq_len).contiguous()

    cache_params: List[torch.Tensor] = []

    # Linear attention layers: conv_states + recurrent_states
    for _ in range(num_linear):
        conv_state = torch.zeros(batch_size, conv_dim, linear_conv_kernel_dim, dtype=torch.float32)
        cache_params.append(conv_state)

        recurrent_state = torch.zeros(
            batch_size, linear_num_key_heads, linear_key_head_dim, linear_value_head_dim,
            dtype=torch.float32,
        )
        cache_params.append(recurrent_state)

    # Full attention layers: key_cache + value_cache with past_seq_len
    for _ in range(num_attention):
        k = torch.zeros(batch_size, num_key_value_heads, past_seq_len, head_dim, dtype=torch.float32)
        cache_params.append(k)
        v = torch.zeros(batch_size, num_key_value_heads, past_seq_len, head_dim, dtype=torch.float32)
        cache_params.append(v)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_params": cache_params,
    }


# ---------------------------------------------------------------------------
# Input / output naming specs (updated: added attention_mask)
# ---------------------------------------------------------------------------

def _build_input_output_specs(
    text_cfg,
) -> Tuple[OrderedDict, OrderedDict]:
    """Return (inputs_spec, outputs_spec) as OrderedDicts."""
    layer_types = text_cfg.layer_types
    num_hidden = text_cfg.num_hidden_layers

    # -- Inputs --
    inputs = OrderedDict()
    inputs["inputs_embeds"] = {0: "batch_size", 1: "sequence_length"}
    inputs["attention_mask"] = {0: "batch_size", 1: "sequence_length"}
    inputs["position_ids"] = {1: "batch_size", 2: "sequence_length"}

    linear_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "linear_attention":
            inputs[f"cache_params.past.conv.{linear_idx}"] = {0: "batch_size"}
            inputs[f"cache_params.past.recurrent.{linear_idx}"] = {0: "batch_size"}
            linear_idx += 1

    attn_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "full_attention":
            inputs[f"cache_params.past.key.{attn_idx}"] = {
                0: "batch_size",
                2: "past_sequence_length",
            }
            inputs[f"cache_params.past.value.{attn_idx}"] = {
                0: "batch_size",
                2: "past_sequence_length",
            }
            attn_idx += 1

    # -- Outputs --
    outputs = OrderedDict()
    outputs["logits"] = {0: "batch_size", 1: "sequence_length"}

    linear_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "linear_attention":
            outputs[f"cache_params.present.conv.{linear_idx}"] = {0: "batch_size"}
            outputs[f"cache_params.present.recurrent.{linear_idx}"] = {0: "batch_size"}
            linear_idx += 1

    attn_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "full_attention":
            outputs[f"cache_params.present.key.{attn_idx}"] = {
                0: "batch_size",
                2: "past_sequence_length + sequence_length",
            }
            outputs[f"cache_params.present.value.{attn_idx}"] = {
                0: "batch_size",
                2: "past_sequence_length + sequence_length",
            }
            attn_idx += 1

    return inputs, outputs


def _build_npu_input_output_specs(text_cfg):
    """Return (inputs_spec, outputs_spec) for NPU static-cache export.

    ALL shapes are fully static.  KV cache inputs are pre-allocated to
    MAX_CACHE_LEN.  KV outputs are size 1 (only the new key/value for
    the current token).  The attention_mask is a 4D float mask of shape
    [1, 1, 1, MAX_CACHE_LEN+1] computed externally.
    """
    layer_types = text_cfg.layer_types
    num_hidden = text_cfg.num_hidden_layers

    inputs = OrderedDict()
    inputs["inputs_embeds"] = {}
    inputs["attention_mask"] = {}   # 4D: [1, 1, 1, MAX_LEN+1]
    inputs["position_ids"] = {}

    linear_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "linear_attention":
            inputs[f"cache_params.past.conv.{linear_idx}"] = {}
            inputs[f"cache_params.past.recurrent.{linear_idx}"] = {}
            linear_idx += 1

    attn_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "full_attention":
            inputs[f"cache_params.past.key.{attn_idx}"] = {}   # static [1, H, MAX_LEN, D]
            inputs[f"cache_params.past.value.{attn_idx}"] = {}
            attn_idx += 1

    outputs = OrderedDict()
    outputs["logits"] = {}

    linear_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "linear_attention":
            outputs[f"cache_params.present.conv.{linear_idx}"] = {}
            outputs[f"cache_params.present.recurrent.{linear_idx}"] = {}
            linear_idx += 1

    attn_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "full_attention":
            outputs[f"cache_params.present.key.{attn_idx}"] = {}   # [1, H, 1, D] (new only)
            outputs[f"cache_params.present.value.{attn_idx}"] = {}
            attn_idx += 1

    return inputs, outputs


def _build_npuw_input_output_specs(text_cfg):
    """Return (inputs_spec, outputs_spec) for NPUW_LLM-compatible export.

    Uses standard HF naming for KV cache (past_key_values.*.key / present.*.key)
    so NPUW_LLM can recognize and manage them.  GDN states keep the
    cache_params.past/present naming (NPUW_LLM ignores them).

    KV cache dims are dynamic (NPUW_LLM does reshape_to_static internally).
    GDN state dims are static (batch_size is the only dynamic dim).
    """
    layer_types = text_cfg.layer_types
    num_hidden = text_cfg.num_hidden_layers

    # -- Inputs --
    inputs = OrderedDict()
    inputs["inputs_embeds"] = {0: "batch_size", 1: "sequence_length"}
    inputs["attention_mask"] = {0: "batch_size", 1: "attention_mask_sequence_length"}
    inputs["position_ids"] = {1: "batch_size", 2: "sequence_length"}

    # GDN states first (matching _generate_dummy_inputs order)
    linear_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "linear_attention":
            inputs[f"cache_params.past.conv.{linear_idx}"] = {0: "batch_size"}
            inputs[f"cache_params.past.recurrent.{linear_idx}"] = {0: "batch_size"}
            linear_idx += 1

    # KV cache with standard HF naming
    attn_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "full_attention":
            inputs[f"past_key_values.{attn_idx}.key"] = {
                0: "batch_size",
                2: "past_sequence_length",
            }
            inputs[f"past_key_values.{attn_idx}.value"] = {
                0: "batch_size",
                2: "past_sequence_length",
            }
            attn_idx += 1

    # -- Outputs --
    outputs = OrderedDict()
    outputs["logits"] = {0: "batch_size", 1: "sequence_length"}

    # GDN states present
    linear_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "linear_attention":
            outputs[f"cache_params.present.conv.{linear_idx}"] = {0: "batch_size"}
            outputs[f"cache_params.present.recurrent.{linear_idx}"] = {0: "batch_size"}
            linear_idx += 1

    # KV present with standard HF naming
    attn_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "full_attention":
            outputs[f"present.{attn_idx}.key"] = {
                0: "batch_size",
                2: "past_sequence_length + sequence_length",
            }
            outputs[f"present.{attn_idx}.value"] = {
                0: "batch_size",
                2: "past_sequence_length + sequence_length",
            }
            attn_idx += 1

    return inputs, outputs


def _build_hybrid_input_output_specs(text_cfg):
    """Return (inputs_spec, outputs_spec) for hybrid NPU+CPU export.

    Same as NPU specs but with 72 extra outputs: per GDN layer,
    4 intermediates (g_t, k_t, v_t, beta_t) for CPU FP32 state update.
    """
    inputs, outputs = _build_npu_input_output_specs(text_cfg)

    # Add GDN intermediate outputs (after all present states)
    layer_types = text_cfg.layer_types
    num_hidden = text_cfg.num_hidden_layers
    linear_idx = 0
    for i in range(num_hidden):
        if layer_types[i] == "linear_attention":
            outputs[f"gdn_intermediate.{linear_idx}.g_t"] = {}
            outputs[f"gdn_intermediate.{linear_idx}.k_t"] = {}
            outputs[f"gdn_intermediate.{linear_idx}.v_t"] = {}
            outputs[f"gdn_intermediate.{linear_idx}.beta_t"] = {}
            linear_idx += 1

    return inputs, outputs


# ---------------------------------------------------------------------------
# Model patching (rewritten: uses RecurrentAttentionCell + ModuleExtension)
# ---------------------------------------------------------------------------

@contextmanager
def _patch_model_for_export(model, text_cfg):
    """Context manager that patches the model for OpenVINO conversion.

    Key changes vs the old version:
    * GDN layers use RecurrentAttentionCell (converted to OV Loop via
      ModuleExtension/ConversionExtension) instead of torch fallbacks.
    * ov_causal_conv1d handles conv cache for any seq_len.
    * attention_mask is passed through (not forced to None).
    * The model supports dynamic seq_len (not fixed to 1).
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DynamicCache,
        Qwen3_5GatedDeltaNet,
    )

    linear_layer_indices, attention_layer_indices = _classify_layers(text_cfg)
    num_linear = len(linear_layer_indices)
    num_attention = len(attention_layer_indices)

    # -- HybridCacheWrap: compact cache lists with abs->compact index mapping --
    class Qwen3_5HybridCacheWrap(Qwen3_5DynamicCache):
        def __init__(self, config, conv_states, recurrent_states, key_cache, value_cache):
            super().__init__(config=config)
            # Replace parent's full-size lists with compact lists
            self.conv_states = conv_states
            self.recurrent_states = recurrent_states
            self.key_cache = key_cache
            self.value_cache = value_cache

            # Build abs_layer_idx -> compact_idx mappings
            self.full_attn_mapping = {}
            self.linear_attn_mapping = {}
            full_idx = 0
            linear_idx = 0
            for i in range(len(config.layer_types)):
                if config.layer_types[i] == "full_attention":
                    self.full_attn_mapping[i] = full_idx
                    full_idx += 1
                elif config.layer_types[i] == "linear_attention":
                    self.linear_attn_mapping[i] = linear_idx
                    linear_idx += 1

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            layer_idx = self.full_attn_mapping[layer_idx]
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat(
                    [self.key_cache[layer_idx], key_states], dim=2,
                )
                self.value_cache[layer_idx] = torch.cat(
                    [self.value_cache[layer_idx], value_states], dim=2,
                )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx=0):
            layer_idx = (
                self.transformer_layers[0]
                if layer_idx not in self.transformer_layers
                else layer_idx
            )
            compact_idx = self.full_attn_mapping[layer_idx]
            if len(self.key_cache) <= compact_idx or self.key_cache[compact_idx] is None:
                return 0
            return self.key_cache[compact_idx].shape[-2]

        @property
        def has_previous_state(self):
            layer_idx = self.linear_attn_mapping[self.last_linear_layer]
            return self.conv_states[layer_idx] is not None

    # -- Save original forward --
    orig_forward = model.forward

    # -- Build patched_forward --
    def patched_forward(
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        cache_params=None,
    ):
        use_cache = False
        wrapped = None
        if cache_params is not None:
            use_cache = True
            conv_states = []
            recurrent_states = []
            for idx in range(num_linear):
                conv_states.append(cache_params[2 * idx])
                recurrent_states.append(cache_params[2 * idx + 1])

            key_cache = []
            value_cache = []
            offset = 2 * num_linear
            for idx in range(num_attention):
                key_cache.append(cache_params[offset + 2 * idx])
                value_cache.append(cache_params[offset + 2 * idx + 1])

            wrapped = Qwen3_5HybridCacheWrap(
                text_cfg, conv_states, recurrent_states, key_cache, value_cache,
            )

        causal_lm_output = orig_forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=wrapped,
            use_cache=use_cache,
        )

        result = {"logits": causal_lm_output.logits}
        if use_cache:
            pkv = causal_lm_output.past_key_values
            present = []
            for compact_idx in range(num_linear):
                present.append(pkv.conv_states[compact_idx])
                present.append(pkv.recurrent_states[compact_idx])
            for compact_idx in range(num_attention):
                present.append(pkv.key_cache[compact_idx])
                present.append(pkv.value_cache[compact_idx])
            result["present_key_values"] = present
        return result

    # -- Wrapper with explicit signature for ov.convert_model --
    # ov.convert_model uses inspect.signature to map example_input dict
    # keys to forward parameters.  *args/**kwargs doesn't work because
    # process_dict_inputs can't resolve the mapping.
    def explicit_forward(inputs_embeds=None, attention_mask=None, position_ids=None, cache_params=None):
        outputs = patched_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
        )
        return tuple(
            value if not isinstance(value, list) else tuple(value)
            for value in outputs.values()
        )

    # -- Apply patches --
    model.forward = explicit_forward

    # Patch GDN layers with RecurrentAttentionCell + ov_causal_conv1d
    gdn_originals = []
    for layer in model.model.layers:
        if not (hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet)):
            continue
        gdn = layer.linear_attn
        gdn_originals.append((
            gdn,
            gdn.forward,
        ))
        gdn.forward = types.MethodType(qwen3_5_gated_delta_net_forward, gdn)
        gdn.recurrent_gated_delta_rule = patched_recurrent_gated_delta_rule
        gdn.recurrent_attention_cell = RecurrentAttentionCell()

    try:
        yield
    finally:
        model.forward = orig_forward
        for gdn, orig_fwd in gdn_originals:
            gdn.forward = orig_fwd
            if hasattr(gdn, "recurrent_attention_cell"):
                del gdn.recurrent_attention_cell
            if hasattr(gdn, "recurrent_gated_delta_rule"):
                del gdn.recurrent_gated_delta_rule


@contextmanager
def _patch_model_for_npu_export(model, text_cfg, output_full_kv=False, capture_intermediates=False):
    """Context manager that patches the model for NPU OpenVINO conversion.

    NPU static-cache approach:
    * Single-step GDN recurrence (no Loop node)
    * KV cache: pre-allocated to MAX_LEN, concatenation produces MAX_LEN+1
    * Output only NEW key/value (size 1), Python manages the buffer
    * 4D attention_mask passed through directly (no internal mask generation)
    * All shapes are fully static — no NPUW_LLM needed

    If output_full_kv=True, KV outputs contain the full concatenated cache
    (for NPUW_LLM which extracts the new token via redirect_new_kv_to_output).

    If capture_intermediates=True, uses the hybrid GDN step function that
    appends (g_t, k_t, v_t, beta_t) to _hybrid_intermediates during tracing,
    and adds them to the model output as gdn_intermediates.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DynamicCache,
        Qwen3_5GatedDeltaNet,
    )

    linear_layer_indices, attention_layer_indices = _classify_layers(text_cfg)
    num_linear = len(linear_layer_indices)
    num_attention = len(attention_layer_indices)

    # -- HybridCacheWrap with concatenation (produces static shapes) --
    class Qwen3_5HybridCacheWrap(Qwen3_5DynamicCache):
        def __init__(self, config, conv_states, recurrent_states, key_cache, value_cache):
            super().__init__(config=config)
            self.conv_states = conv_states
            self.recurrent_states = recurrent_states
            self.key_cache = key_cache
            self.value_cache = value_cache

            self.full_attn_mapping = {}
            self.linear_attn_mapping = {}
            full_idx = 0
            linear_idx = 0
            for i in range(len(config.layer_types)):
                if config.layer_types[i] == "full_attention":
                    self.full_attn_mapping[i] = full_idx
                    full_idx += 1
                elif config.layer_types[i] == "linear_attention":
                    self.linear_attn_mapping[i] = linear_idx
                    linear_idx += 1

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            layer_idx = self.full_attn_mapping[layer_idx]
            # Concatenate: [1, H, MAX_LEN, D] + [1, H, 1, D] → [1, H, MAX_LEN+1, D]
            # This is always the same shape since past is always MAX_LEN.
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2,
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2,
            )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx=0):
            compact_idx = self.full_attn_mapping.get(layer_idx)
            if compact_idx is None:
                compact_idx = list(self.full_attn_mapping.values())[0]
            if len(self.key_cache) <= compact_idx or self.key_cache[compact_idx] is None:
                return 0
            return self.key_cache[compact_idx].shape[-2]

        @property
        def has_previous_state(self):
            layer_idx = self.linear_attn_mapping[self.last_linear_layer]
            return self.conv_states[layer_idx] is not None

    # -- Save originals --
    orig_forward = model.forward
    # NOTE: In transformers 5.x, create_causal_mask() (module-level function)
    # automatically passes through 4D attention masks.  No need to patch it.
    # Our 4D float mask [1, 1, 1, MAX_LEN+1] is returned as-is.

    # -- Build patched_forward --
    # capture_intermediates is a closure variable from the outer scope.
    _do_capture = capture_intermediates

    def patched_forward(
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        cache_params=None,
    ):
        # Clear stale intermediates at the start of each forward call
        if _do_capture:
            _hybrid_intermediates.clear()

        use_cache = False
        wrapped = None
        if cache_params is not None:
            use_cache = True
            conv_states = []
            recurrent_states = []
            for idx in range(num_linear):
                conv_states.append(cache_params[2 * idx])
                recurrent_states.append(cache_params[2 * idx + 1])

            key_cache = []
            value_cache = []
            offset = 2 * num_linear
            for idx in range(num_attention):
                key_cache.append(cache_params[offset + 2 * idx])
                value_cache.append(cache_params[offset + 2 * idx + 1])

            wrapped = Qwen3_5HybridCacheWrap(
                text_cfg, conv_states, recurrent_states, key_cache, value_cache,
            )

        causal_lm_output = orig_forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=wrapped,
            use_cache=use_cache,
        )

        result = {"logits": causal_lm_output.logits}
        if use_cache:
            pkv = causal_lm_output.past_key_values
            present = []
            for compact_idx in range(num_linear):
                present.append(pkv.conv_states[compact_idx])
                present.append(pkv.recurrent_states[compact_idx])
            for compact_idx in range(num_attention):
                if output_full_kv:
                    # Full concatenated KV for NPUW_LLM
                    # (it will extract new token via redirect_new_kv_to_output)
                    present.append(pkv.key_cache[compact_idx])
                    present.append(pkv.value_cache[compact_idx])
                else:
                    # Output ONLY the new key/value (last position after concat)
                    # full shape: [1, H, MAX_LEN+1, D] -> new: [1, H, 1, D]
                    present.append(pkv.key_cache[compact_idx][:, :, -1:, :])
                    present.append(pkv.value_cache[compact_idx][:, :, -1:, :])
            result["present_key_values"] = present

        # Collect GDN intermediates for hybrid CPU FP32 state update
        if _do_capture and _hybrid_intermediates:
            gdn_inters = []
            for g_t, k_t, v_t, beta_t in _hybrid_intermediates:
                gdn_inters.extend([g_t, k_t, v_t, beta_t])
            result["gdn_intermediates"] = gdn_inters
            _hybrid_intermediates.clear()

        return result

    def explicit_forward(inputs_embeds=None, attention_mask=None, position_ids=None, cache_params=None):
        outputs = patched_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
        )
        return tuple(
            value if not isinstance(value, list) else tuple(value)
            for value in outputs.values()
        )

    # -- Apply patches --
    model.forward = explicit_forward

    # Patch GDN layers with single-step recurrence (no RecurrentAttentionCell)
    gdn_originals = []
    gdn_recurrent_fn = (
        patched_recurrent_gated_delta_rule_hybrid_step
        if capture_intermediates
        else patched_recurrent_gated_delta_rule_single_step
    )
    for layer in model.model.layers:
        if not (hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet)):
            continue
        gdn = layer.linear_attn
        gdn_originals.append((
            gdn,
            gdn.forward,
        ))
        gdn.forward = types.MethodType(qwen3_5_gated_delta_net_forward, gdn)
        gdn.recurrent_gated_delta_rule = gdn_recurrent_fn

    try:
        yield
    finally:
        model.forward = orig_forward
        for gdn, orig_fwd in gdn_originals:
            gdn.forward = orig_fwd
            if hasattr(gdn, "recurrent_gated_delta_rule"):
                del gdn.recurrent_gated_delta_rule


# ---------------------------------------------------------------------------
# Copy tokenizer / config files
# ---------------------------------------------------------------------------

def _copy_tokenizer_and_config(model_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer artefacts, config.json, etc."""
    for pattern in _TOKENIZER_AND_CONFIG_GLOBS:
        for src in model_dir.glob(pattern):
            dst = output_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.info("Copied %s", src.name)


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_model(
    model_dir: str,
    output_dir: str,
    compress_to_fp16: bool = True,
) -> None:
    """Export a Qwen3.5 PyTorch model to a stateful OpenVINO IR.

    Uses ModuleExtension + ConversionExtension to convert GDN recurrence
    into OpenVINO Loop nodes, enabling dynamic sequence length.
    """
    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load PyTorch model
    logger.info("Loading PyTorch model from %s ...", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.config.torchscript = False
    model.config.return_dict = True

    # 2. Resolve text config
    text_cfg = (
        model.config.text_config
        if hasattr(model.config, "text_config")
        else model.config
    )

    # 3. Generate dummy inputs (seq_len=2 for proper Loop tracing)
    logger.info("Generating dummy inputs ...")
    dummy_inputs = _generate_dummy_inputs(text_cfg, batch_size=2, seq_len=2, past_seq_len=2)

    # 4-7. Patch model, build input info, convert
    with torch.no_grad(), _patch_model_for_export(model, text_cfg):
        # 4a. Remove None values and flatten
        dummy_inputs, dict_inputs = _remove_none(dummy_inputs)

        # 4b. Build input / output naming specs
        inputs_spec, outputs_spec = _build_input_output_specs(text_cfg)

        # 4c. Flatten example tensors and build input_info
        ordered_input_names = list(inputs_spec.keys())
        flatten_inputs = _flattenize_inputs(list(dummy_inputs.values()))

        if len(flatten_inputs) != len(ordered_input_names):
            raise RuntimeError(
                f"Mismatch: {len(flatten_inputs)} flattened tensors vs "
                f"{len(ordered_input_names)} input names in spec"
            )

        input_info = _get_input_info(ordered_input_names, inputs_spec, flatten_inputs)

        # 6. Build extensions for RecurrentAttentionCell -> OV Loop
        extensions = [
            ModuleExtension(RecurrentAttentionCell, "RecurrentAttentionCellOp"),
            ConversionExtension("RecurrentAttentionCellOp", convert_recurrent_attention_cell),
        ]

        # 7. Convert to OpenVINO model (direct PyTorch path, NOT TorchScriptPythonDecoder)
        logger.info("Converting to OpenVINO IR with ModuleExtension ...")
        ov_model = ov.convert_model(
            model,
            example_input=dummy_inputs,
            input=[(item.shape, item.type) for item in input_info],
            extension=extensions,
        )

    ov_model.validate_nodes_and_infer_types()

    # 8. Name outputs
    output_names = list(outputs_spec.keys())
    for idx, out_tensor in enumerate(ov_model.outputs):
        if idx < len(output_names):
            out_tensor.get_tensor().set_names({output_names[idx]})

    # 9. Name inputs
    input_names = [item.name for item in input_info]
    for idx, inp_tensor in enumerate(ov_model.inputs):
        if idx < len(input_names):
            inp_tensor.get_tensor().set_names({input_names[idx]})

    # 10. Convert to stateful model
    logger.info("Applying stateful transformation ...")
    patch_stateful_hybrid_ssm(ov_model)

    # 11. Save
    xml_path = out_path / "openvino_model.xml"
    logger.info("Saving model to %s (compress_to_fp16=%s) ...", xml_path, compress_to_fp16)
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)

    # 11b. Extract embedding table for external lookup
    embed_table = model.model.embed_tokens.weight.detach().cpu().numpy()
    embed_path = out_path / "embed_tokens.npy"
    np.save(str(embed_path), embed_table.astype(np.float16))
    logger.info("Saved embed_tokens.npy: shape=%s, dtype=float16", embed_table.shape)

    # 12. Copy tokenizer / config files
    _copy_tokenizer_and_config(model_path, out_path)

    # Cleanup
    _clear_class_registry()
    del ov_model
    del model

    logger.info("Export complete: %s", out_path)


def export_model_npu(
    model_dir: str,
    output_dir: str,
    compress_to_fp16: bool = True,
    max_cache_len: int = 256,
) -> None:
    """Export a Qwen3.5 model to a fully-static OpenVINO IR for NPU.

    Unlike export_model(), this version:
    * Traces with seq_len=1 (no dynamic sequence length for GDN layers)
    * Does NOT use ModuleExtension/ConversionExtension (no OV Loop node)
    * GDN recurrence is inlined as single-step computation
    * KV cache is pre-allocated to max_cache_len (static shapes)
    * Attention mask is 4D float, computed externally
    * No stateful transformation — all states are explicit I/O
    * No NPUW_LLM needed — NPU compiles all-static model directly
    """
    from .config import NPU_MAX_CACHE_LEN
    if max_cache_len <= 0:
        max_cache_len = NPU_MAX_CACHE_LEN

    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load PyTorch model
    logger.info("Loading PyTorch model from %s ...", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.config.torchscript = False
    model.config.return_dict = True

    # 2. Resolve text config
    text_cfg = (
        model.config.text_config
        if hasattr(model.config, "text_config")
        else model.config
    )

    # 3. Generate dummy inputs with pre-allocated KV cache
    logger.info("Generating dummy inputs (NPU static cache: max_cache_len=%d) ...", max_cache_len)
    dummy_inputs = _generate_dummy_inputs(text_cfg, batch_size=1, seq_len=1, past_seq_len=max_cache_len)

    # Replace 2D attention_mask [1, 1] with 4D float mask [1, 1, 1, MAX_LEN+1]
    # The +1 accounts for the current token after KV concatenation.
    dummy_inputs["attention_mask"] = torch.zeros(1, 1, 1, max_cache_len + 1, dtype=torch.float32)

    # 4-7. Patch model (NPU mode: single-step GDN + 4D mask passthrough)
    with torch.no_grad(), _patch_model_for_npu_export(model, text_cfg):
        dummy_inputs, dict_inputs = _remove_none(dummy_inputs)

        inputs_spec, outputs_spec = _build_npu_input_output_specs(text_cfg)

        ordered_input_names = list(inputs_spec.keys())
        flatten_inputs = _flattenize_inputs(list(dummy_inputs.values()))

        if len(flatten_inputs) != len(ordered_input_names):
            raise RuntimeError(
                f"Mismatch: {len(flatten_inputs)} flattened tensors vs "
                f"{len(ordered_input_names)} input names in spec"
            )

        input_info = _get_input_info(ordered_input_names, inputs_spec, flatten_inputs)

        logger.info("Converting to OpenVINO IR (NPU static cache: no Loop) ...")
        ov_model = ov.convert_model(
            model,
            example_input=dummy_inputs,
            input=[(item.shape, item.type) for item in input_info],
        )

    ov_model.validate_nodes_and_infer_types()

    # 8. Name outputs
    output_names = list(outputs_spec.keys())
    for idx, out_tensor in enumerate(ov_model.outputs):
        if idx < len(output_names):
            out_tensor.get_tensor().set_names({output_names[idx]})

    # 9. Name inputs
    input_names = [item.name for item in input_info]
    for idx, inp_tensor in enumerate(ov_model.inputs):
        if idx < len(input_names):
            inp_tensor.get_tensor().set_names({input_names[idx]})

    # 10. NO stateful transformation — all states are explicit I/O.
    # Python manages KV cache buffers and GDN states at inference time.
    logger.info("Skipping stateful transformation (all-static NPU model)")

    # 11. Save
    xml_path = out_path / "openvino_model.xml"
    logger.info("Saving NPU model to %s (compress_to_fp16=%s) ...", xml_path, compress_to_fp16)
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)

    # 11b. Extract embedding table
    embed_table = model.model.embed_tokens.weight.detach().cpu().numpy()
    embed_path = out_path / "embed_tokens.npy"
    np.save(str(embed_path), embed_table.astype(np.float16))
    logger.info("Saved embed_tokens.npy: shape=%s, dtype=float16", embed_table.shape)

    # 12. Copy tokenizer / config files
    _copy_tokenizer_and_config(model_path, out_path)

    # Cleanup
    _clear_class_registry()
    del ov_model
    del model

    logger.info("NPU export complete (max_cache_len=%d): %s", max_cache_len, out_path)


def export_model_npuw(
    model_dir: str,
    output_dir: str,
    compress_to_fp16: bool = True,
) -> None:
    """Export Qwen3.5 for NPUW_LLM: dynamic KV shapes + Loop-free GDN.

    Produces a model compatible with NPUW_LLM KV cache management:
    * Dynamic KV cache shapes (NPUW_LLM does reshape_to_static)
    * Standard HF naming (past_key_values.*.key / present.*.key)
    * Full KV concat output (NPUW_LLM redirects to new-only)
    * 2D attention mask (standard LLM interface)
    * Loop-free GDN (single-step, seq_len=1 only)
    * NPUW_LLM_MAX_PROMPT_LEN=1 required (GDN can't handle seq_len>1)

    Use with: NPUW_LLM=YES, NPUW_LLM_MAX_PROMPT_LEN=1, NPUW_FOLD=NO
    """
    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load PyTorch model
    logger.info("Loading PyTorch model from %s ...", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.eval()
    model.config.torchscript = False
    model.config.return_dict = True

    # 2. Resolve text config
    text_cfg = (
        model.config.text_config
        if hasattr(model.config, "text_config")
        else model.config
    )

    # 3. Generate dummy inputs with small past (shapes will be dynamic)
    logger.info("Generating dummy inputs (NPUW_LLM dynamic shapes) ...")
    dummy_inputs = _generate_dummy_inputs(text_cfg, batch_size=1, seq_len=1, past_seq_len=5)
    # Override attention_mask to 2D covering past + current
    dummy_inputs["attention_mask"] = torch.ones(1, 5 + 1, dtype=torch.int64)

    # 4-7. Patch model (NPU mode + full KV concat output)
    with torch.no_grad(), _patch_model_for_npu_export(model, text_cfg, output_full_kv=True):
        dummy_inputs, dict_inputs = _remove_none(dummy_inputs)
        inputs_spec, outputs_spec = _build_npuw_input_output_specs(text_cfg)

        ordered_input_names = list(inputs_spec.keys())
        flatten_inputs = _flattenize_inputs(list(dummy_inputs.values()))

        if len(flatten_inputs) != len(ordered_input_names):
            raise RuntimeError(
                f"Mismatch: {len(flatten_inputs)} flattened tensors vs "
                f"{len(ordered_input_names)} input names in spec"
            )

        input_info = _get_input_info(ordered_input_names, inputs_spec, flatten_inputs)

        logger.info("Converting to OpenVINO IR (NPUW_LLM compatible: dynamic KV, Loop-free GDN) ...")
        ov_model = ov.convert_model(
            model,
            example_input=dummy_inputs,
            input=[(item.shape, item.type) for item in input_info],
        )

    ov_model.validate_nodes_and_infer_types()

    # 8. Name outputs
    output_names = list(outputs_spec.keys())
    for idx, out_tensor in enumerate(ov_model.outputs):
        if idx < len(output_names):
            out_tensor.get_tensor().set_names({output_names[idx]})

    # 9. Name inputs
    input_names = [item.name for item in input_info]
    for idx, inp_tensor in enumerate(ov_model.inputs):
        if idx < len(input_names):
            inp_tensor.get_tensor().set_names({input_names[idx]})

    # 10. Add beam_idx parameter (required by NPUW_LLM's StatefulToStateless)
    #     It's a dummy input not connected to any computation, but NPUW_LLM
    #     needs it to exist for KV cache reordering during beam search.
    beam_idx_param = ov.opset13.parameter(ov.PartialShape([-1]), dtype=ov.Type.i64)
    beam_idx_param.get_output_tensor(0).set_names({"beam_idx"})
    beam_idx_param.set_friendly_name("beam_idx")
    ov_model.add_parameters([beam_idx_param])
    logger.info("Added beam_idx parameter for NPUW_LLM compatibility")

    # 11. No stateful transformation -- NPUW_LLM manages state internally
    logger.info("Skipping stateful transformation (NPUW_LLM will manage KV cache)")

    # 11. Save
    xml_path = out_path / "openvino_model.xml"
    logger.info("Saving NPUW_LLM model to %s (compress_to_fp16=%s) ...", xml_path, compress_to_fp16)
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)

    # 11b. Extract embedding table
    embed_table = model.model.embed_tokens.weight.detach().cpu().numpy()
    embed_path = out_path / "embed_tokens.npy"
    np.save(str(embed_path), embed_table.astype(np.float16))
    logger.info("Saved embed_tokens.npy: shape=%s, dtype=float16", embed_table.shape)

    # 12. Copy tokenizer / config files
    _copy_tokenizer_and_config(model_path, out_path)

    # Cleanup
    _clear_class_registry()
    del ov_model
    del model

    logger.info("NPUW_LLM export complete: %s", out_path)


def export_model_hybrid(
    model_dir: str,
    output_dir: str,
    compress_to_fp16: bool = True,
    max_cache_len: int = 256,
) -> None:
    """Export Qwen3.5 for hybrid NPU+CPU inference.

    Same as export_model_npu but adds 72 intermediate outputs
    (g_t, k_t, v_t, beta_t per GDN layer) for CPU FP32 state update.
    The NPU model still does full state update + readout in FP16.
    CPU maintains a shadow FP32 state using these intermediates.
    """
    from .config import NPU_MAX_CACHE_LEN
    if max_cache_len <= 0:
        max_cache_len = NPU_MAX_CACHE_LEN

    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PyTorch model from %s ...", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.eval()
    model.config.torchscript = False
    model.config.return_dict = True

    text_cfg = (
        model.config.text_config
        if hasattr(model.config, "text_config")
        else model.config
    )

    logger.info("Generating dummy inputs (hybrid: max_cache_len=%d) ...", max_cache_len)
    dummy_inputs = _generate_dummy_inputs(text_cfg, batch_size=1, seq_len=1, past_seq_len=max_cache_len)
    dummy_inputs["attention_mask"] = torch.zeros(1, 1, 1, max_cache_len + 1, dtype=torch.float32)

    # Clear any stale intermediates
    _hybrid_intermediates.clear()

    with torch.no_grad(), _patch_model_for_npu_export(model, text_cfg, capture_intermediates=True):
        dummy_inputs, dict_inputs = _remove_none(dummy_inputs)
        inputs_spec, outputs_spec = _build_hybrid_input_output_specs(text_cfg)

        ordered_input_names = list(inputs_spec.keys())
        flatten_inputs = _flattenize_inputs(list(dummy_inputs.values()))

        if len(flatten_inputs) != len(ordered_input_names):
            raise RuntimeError(
                f"Mismatch: {len(flatten_inputs)} flattened tensors vs "
                f"{len(ordered_input_names)} input names in spec"
            )

        input_info = _get_input_info(ordered_input_names, inputs_spec, flatten_inputs)

        logger.info("Converting to OpenVINO IR (hybrid: NPU + CPU FP32 state) ...")
        ov_model = ov.convert_model(
            model,
            example_input=dummy_inputs,
            input=[(item.shape, item.type) for item in input_info],
        )

    ov_model.validate_nodes_and_infer_types()

    # Name outputs
    output_names = list(outputs_spec.keys())
    for idx, out_tensor in enumerate(ov_model.outputs):
        if idx < len(output_names):
            out_tensor.get_tensor().set_names({output_names[idx]})

    # Name inputs
    input_names = [item.name for item in input_info]
    for idx, inp_tensor in enumerate(ov_model.inputs):
        if idx < len(input_names):
            inp_tensor.get_tensor().set_names({input_names[idx]})

    logger.info("Skipping stateful transformation (hybrid all-static model)")

    xml_path = out_path / "openvino_model.xml"
    logger.info("Saving hybrid model to %s (compress_to_fp16=%s) ...", xml_path, compress_to_fp16)
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)

    embed_table = model.model.embed_tokens.weight.detach().cpu().numpy()
    embed_path = out_path / "embed_tokens.npy"
    np.save(str(embed_path), embed_table.astype(np.float16))
    logger.info("Saved embed_tokens.npy: shape=%s, dtype=float16", embed_table.shape)

    _copy_tokenizer_and_config(model_path, out_path)

    _clear_class_registry()
    del ov_model
    del model

    logger.info("Hybrid export complete (max_cache_len=%d): %s", max_cache_len, out_path)


def export_model_multisubgraph(
    model_dir: str,
    output_dir: str,
    compress_to_fp16: bool = True,
    max_cache_len: int = 256,
) -> None:
    """Export Qwen3.5 as 6 subgraph IRs for multi-subgraph NPU inference.

    Splits 24 layers into 6 groups of 4 layers each. Between subgraphs,
    hidden_states return to CPU as FP32, limiting FP16 error accumulation
    to 4 layers instead of 24.

    Each subgraph covers [GDN, GDN, GDN, FullAttn] and has:
    - Input: hidden_states [1,1,1024] (FP32, auto-truncated to FP16 by NPU)
    - Input: 3 conv states + 3 recurrent states (GDN layers)
    - Input: 1 key cache + 1 value cache (FullAttn layer)
    - Input: attention_mask [1,1,1,max_cache_len+1] (4D float)
    - Input: position_ids [3,1,1]
    - Output: hidden_states [1,1,1024]
    - Output: updated conv/recurrent/key/value states
    - Output: 3x4 GDN intermediates (g_t, k_t, v_t, beta_t per GDN layer)
    - Output (subgraph 5 only): logits [1,1,vocab_size]

    Uses a lightweight SubgraphModule(nn.Module) wrapper per subgraph so that
    ov.convert_model traces only the subgraph layers, not the full CausalLM.
    Cache tensors are passed as individual positional args (*cache_params),
    not as a list, to avoid torch.jit.trace issues with list-of-tensors.

    Saves: subgraph_0.xml through subgraph_5.xml + embed_tokens.npy + tokenizer files.
    """
    from .config import NPU_MAX_CACHE_LEN, MULTISUB_LAYERS_PER_SUBGRAPH, MULTISUB_NUM_SUBGRAPHS
    if max_cache_len <= 0:
        max_cache_len = NPU_MAX_CACHE_LEN

    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PyTorch model from %s ...", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.eval()
    model.config.torchscript = False
    model.config.return_dict = True

    text_cfg = (
        model.config.text_config
        if hasattr(model.config, "text_config")
        else model.config
    )

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DynamicCache,
        Qwen3_5GatedDeltaNet,
    )

    linear_layer_indices, attention_layer_indices = _classify_layers(text_cfg)

    # Architecture constants
    num_layers = text_cfg.num_hidden_layers  # 24
    layers_per_sub = MULTISUB_LAYERS_PER_SUBGRAPH  # 4
    num_subs = MULTISUB_NUM_SUBGRAPHS  # 6
    hidden_size = text_cfg.hidden_size  # 1024

    # GDN dimensions
    linear_num_key_heads = text_cfg.linear_num_key_heads  # 16
    linear_key_head_dim = text_cfg.linear_key_head_dim  # 128
    linear_value_head_dim = text_cfg.linear_value_head_dim  # 128
    linear_num_value_heads = text_cfg.linear_num_value_heads  # 16
    linear_conv_kernel_dim = text_cfg.linear_conv_kernel_dim  # 4
    conv_dim = (
        linear_num_key_heads * linear_key_head_dim * 2
        + linear_num_value_heads * linear_value_head_dim
    )

    # KV dimensions
    num_key_value_heads = text_cfg.num_key_value_heads  # 4
    head_dim = getattr(text_cfg, "head_dim", hidden_size // text_cfg.num_attention_heads)  # 256

    layer_types = text_cfg.layer_types

    # ------------------------------------------------------------------
    # SubgraphCacheWrap: maps absolute layer indices to compact indices
    # within a single subgraph so that the decoder layers can look up
    # their conv/recurrent/key/value states correctly.
    # ------------------------------------------------------------------
    class SubgraphCacheWrap(Qwen3_5DynamicCache):
        def __init__(self, config, conv_states, recurrent_states,
                     key_cache, value_cache, _linear_mapping, _attn_mapping):
            super().__init__(config=config)
            self.conv_states = conv_states
            self.recurrent_states = recurrent_states
            self.key_cache = key_cache
            self.value_cache = value_cache
            self.full_attn_mapping = _attn_mapping
            self.linear_attn_mapping = _linear_mapping

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            layer_idx = self.full_attn_mapping[layer_idx]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2,
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2,
            )
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx=0):
            compact_idx = list(self.full_attn_mapping.values())[0]
            if len(self.key_cache) <= compact_idx or self.key_cache[compact_idx] is None:
                return 0
            return self.key_cache[compact_idx].shape[-2]

        @property
        def has_previous_state(self):
            return self.conv_states[0] is not None

    # ------------------------------------------------------------------
    # SubgraphModule: a lightweight nn.Module that wraps only the layers
    # belonging to one subgraph.  ov.convert_model traces this wrapper
    # instead of the full CausalLM, avoiding tracer confusion.
    #
    # forward() signature: (hidden_states, attention_mask, position_ids,
    #                        *cache_params)
    # Each cache tensor is a separate positional arg so that
    # ov.convert_model / torch.jit.trace sees individual tensors, not a
    # list.
    # ------------------------------------------------------------------
    class SubgraphModule(torch.nn.Module):
        def __init__(self, layers_list, rotary_emb, final_norm, lm_head,
                     sub_text_cfg, n_linear, n_attn,
                     linear_idx_map, attn_idx_map, is_final):
            super().__init__()
            self.sub_layers = torch.nn.ModuleList(layers_list)
            self.rotary_emb = rotary_emb
            self.final_norm = final_norm if is_final else None
            self.lm_head = lm_head if is_final else None
            self._cfg = sub_text_cfg
            self._n_linear = n_linear
            self._n_attn = n_attn
            self._linear_idx_map = linear_idx_map   # {abs_layer_idx: compact_idx}
            self._attn_idx_map = attn_idx_map       # {abs_layer_idx: compact_idx}
            self._is_final = is_final

        def forward(self, hidden_states, attention_mask, position_ids, *cache_params):
            # cache_params is a *args tuple of individual tensors, NOT a list.
            _hybrid_intermediates.clear()

            # --- Unpack cache_params into lists for SubgraphCacheWrap ---
            conv_states = []
            recurrent_states = []
            for i in range(self._n_linear):
                conv_states.append(cache_params[2 * i])
                recurrent_states.append(cache_params[2 * i + 1])
            key_cache = []
            value_cache = []
            offset = 2 * self._n_linear
            for i in range(self._n_attn):
                key_cache.append(cache_params[offset + 2 * i])
                value_cache.append(cache_params[offset + 2 * i + 1])

            cache = SubgraphCacheWrap(
                self._cfg, conv_states, recurrent_states,
                key_cache, value_cache,
                self._linear_idx_map, self._attn_idx_map,
            )

            # --- Compute rotary position embeddings (shared by all layers) ---
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # --- Run through the subgraph layers ---
            h = hidden_states
            for layer in self.sub_layers:
                layer_outputs = layer(
                    h,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    cache_position=None,
                )
                h = layer_outputs[0]

            # --- Collect outputs ---
            results = []

            if self._is_final:
                h_normed = self.final_norm(h)
                logits = self.lm_head(h_normed)
                results.append(logits)

            # Always output hidden_states (for chaining to next subgraph)
            results.append(h)

            # Updated cache states
            for i in range(self._n_linear):
                results.append(cache.conv_states[i])
                results.append(cache.recurrent_states[i])
            for i in range(self._n_attn):
                # Output only the NEW key/value entry (last position after cat).
                # At inference time, the Python side manages the full cache
                # buffer and handles the append.
                results.append(cache.key_cache[i][:, :, -1:, :])
                results.append(cache.value_cache[i][:, :, -1:, :])

            # GDN intermediates for CPU FP32 shadow state update
            for g_t, k_t, v_t, beta_t in _hybrid_intermediates:
                results.extend([g_t, k_t, v_t, beta_t])

            return tuple(results)

    # ------------------------------------------------------------------
    # Patch GDN layers for single-step + intermediates capture
    # ------------------------------------------------------------------
    gdn_originals = []
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet):
            gdn = layer.linear_attn
            gdn_originals.append((gdn, gdn.forward))
            gdn.forward = types.MethodType(qwen3_5_gated_delta_net_forward, gdn)
            gdn.recurrent_gated_delta_rule = patched_recurrent_gated_delta_rule_hybrid_step

    try:
        for sub_idx in range(num_subs):
            start_layer = sub_idx * layers_per_sub
            end_layer = (sub_idx + 1) * layers_per_sub
            is_last = (sub_idx == num_subs - 1)

            logger.info("Exporting subgraph %d (layers %d-%d) ...", sub_idx, start_layer, end_layer - 1)

            # Count GDN and attention layers in this subgraph
            sub_linear_indices = []
            sub_attn_indices = []
            for i in range(start_layer, end_layer):
                if layer_types[i] == "linear_attention":
                    sub_linear_indices.append(i)
                else:
                    sub_attn_indices.append(i)
            num_sub_linear = len(sub_linear_indices)   # typically 3
            num_sub_attn = len(sub_attn_indices)       # typically 1

            # Build absolute-layer-index -> compact-index mappings
            linear_mapping = {}
            compact_linear = 0
            for i in range(start_layer, end_layer):
                if layer_types[i] == "linear_attention":
                    linear_mapping[i] = compact_linear
                    compact_linear += 1

            attn_mapping = {}
            compact_attn = 0
            for i in range(start_layer, end_layer):
                if layer_types[i] == "full_attention":
                    attn_mapping[i] = compact_attn
                    compact_attn += 1

            # --- Create SubgraphModule wrapper ---
            wrapper = SubgraphModule(
                layers_list=[model.model.layers[i] for i in range(start_layer, end_layer)],
                rotary_emb=model.model.rotary_emb,
                final_norm=model.model.norm,
                lm_head=model.lm_head,
                sub_text_cfg=text_cfg,
                n_linear=num_sub_linear,
                n_attn=num_sub_attn,
                linear_idx_map=linear_mapping,
                attn_idx_map=attn_mapping,
                is_final=is_last,
            )
            wrapper.eval()

            # --- Generate dummy inputs (all static shapes) ---
            hidden_states_dummy = torch.zeros(1, 1, hidden_size, dtype=torch.float32)
            attention_mask_dummy = torch.zeros(1, 1, 1, max_cache_len + 1, dtype=torch.float32)
            position_ids_dummy = torch.zeros(3, 1, 1, dtype=torch.int64)

            # Cache params: interleaved (conv, recurrent) per GDN, then (key, value) per attn
            sub_cache_tensors = []
            for _ in range(num_sub_linear):
                sub_cache_tensors.append(
                    torch.zeros(1, conv_dim, linear_conv_kernel_dim, dtype=torch.float32))
                sub_cache_tensors.append(
                    torch.zeros(1, linear_num_key_heads, linear_key_head_dim,
                                linear_value_head_dim, dtype=torch.float32))
            for _ in range(num_sub_attn):
                sub_cache_tensors.append(
                    torch.zeros(1, num_key_value_heads, max_cache_len, head_dim, dtype=torch.float32))
                sub_cache_tensors.append(
                    torch.zeros(1, num_key_value_heads, max_cache_len, head_dim, dtype=torch.float32))

            # Build example_input as a FLAT tuple of positional args.
            # This is critical: ov.convert_model will call
            #   wrapper.forward(*example_input)
            # so each cache tensor becomes an individual positional arg
            # matching the *cache_params varargs.
            example_input = (
                hidden_states_dummy,
                attention_mask_dummy,
                position_ids_dummy,
            ) + tuple(sub_cache_tensors)

            # Clear intermediates before tracing
            _hybrid_intermediates.clear()

            # --- Build input / output name specs ---
            inputs_spec = OrderedDict()
            inputs_spec["hidden_states"] = {}
            inputs_spec["attention_mask"] = {}
            inputs_spec["position_ids"] = {}
            for local_idx in range(num_sub_linear):
                inputs_spec[f"cache_params.past.conv.{local_idx}"] = {}
                inputs_spec[f"cache_params.past.recurrent.{local_idx}"] = {}
            for local_idx in range(num_sub_attn):
                inputs_spec[f"cache_params.past.key.{local_idx}"] = {}
                inputs_spec[f"cache_params.past.value.{local_idx}"] = {}

            outputs_spec = OrderedDict()
            if is_last:
                outputs_spec["logits"] = {}
            outputs_spec["hidden_states"] = {}
            for local_idx in range(num_sub_linear):
                outputs_spec[f"cache_params.present.conv.{local_idx}"] = {}
                outputs_spec[f"cache_params.present.recurrent.{local_idx}"] = {}
            for local_idx in range(num_sub_attn):
                outputs_spec[f"cache_params.present.key.{local_idx}"] = {}
                outputs_spec[f"cache_params.present.value.{local_idx}"] = {}
            for local_idx in range(num_sub_linear):
                outputs_spec[f"gdn_intermediate.{local_idx}.g_t"] = {}
                outputs_spec[f"gdn_intermediate.{local_idx}.k_t"] = {}
                outputs_spec[f"gdn_intermediate.{local_idx}.v_t"] = {}
                outputs_spec[f"gdn_intermediate.{local_idx}.beta_t"] = {}

            # --- Convert to OpenVINO IR ---
            with torch.no_grad():
                logger.info("Converting subgraph %d to OpenVINO IR ...", sub_idx)
                ov_model = ov.convert_model(
                    wrapper,
                    example_input=example_input,
                )

            ov_model.validate_nodes_and_infer_types()

            # Name inputs
            input_names = list(inputs_spec.keys())
            for idx, inp_tensor in enumerate(ov_model.inputs):
                if idx < len(input_names):
                    inp_tensor.get_tensor().set_names({input_names[idx]})

            # Name outputs
            output_names = list(outputs_spec.keys())
            for idx, out_tensor in enumerate(ov_model.outputs):
                if idx < len(output_names):
                    out_tensor.get_tensor().set_names({output_names[idx]})

            # Save
            xml_path = out_path / f"subgraph_{sub_idx}.xml"
            logger.info("Saving subgraph %d to %s ...", sub_idx, xml_path)
            ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)

            del ov_model
            _hybrid_intermediates.clear()

    finally:
        # Restore GDN layers
        for gdn, orig_fwd in gdn_originals:
            gdn.forward = orig_fwd
            if hasattr(gdn, "recurrent_gated_delta_rule"):
                del gdn.recurrent_gated_delta_rule

    # Save embedding table
    embed_table = model.model.embed_tokens.weight.detach().cpu().numpy()
    embed_path = out_path / "embed_tokens.npy"
    np.save(str(embed_path), embed_table.astype(np.float16))
    logger.info("Saved embed_tokens.npy: shape=%s, dtype=float16", embed_table.shape)

    # Copy tokenizer / config files
    _copy_tokenizer_and_config(model_path, out_path)

    _clear_class_registry()
    del model

    logger.info("Multi-subgraph export complete (%d subgraphs): %s", num_subs, out_path)


def export_model_loop_ir(
    model_dir: str,
    output_dir: str,
    compress_to_fp16: bool = True,
) -> None:
    """Export a non-stateful Qwen3.5 IR with Loop nodes for LowLatency2 pipeline.

    Same as export_model() but WITHOUT applying patch_stateful_hybrid_ssm.
    The output IR has 18 TensorIterator (Loop) nodes and all 48 cache tensors
    as explicit Parameters/Results.  This is the input for the LL2 pipeline
    which will: reshape → ConstantFolding → LowLatency2 → MakeStateful.

    Output naming uses cache_params.past/present.* convention (same as export_model).
    """
    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load PyTorch model
    logger.info("Loading PyTorch model from %s ...", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.config.torchscript = False
    model.config.return_dict = True

    # 2. Resolve text config
    text_cfg = (
        model.config.text_config
        if hasattr(model.config, "text_config")
        else model.config
    )

    # 3. Generate dummy inputs (seq_len=2 for proper Loop tracing)
    logger.info("Generating dummy inputs ...")
    dummy_inputs = _generate_dummy_inputs(text_cfg, batch_size=2, seq_len=2, past_seq_len=2)

    # 4-7. Patch model, build input info, convert
    with torch.no_grad(), _patch_model_for_export(model, text_cfg):
        dummy_inputs, dict_inputs = _remove_none(dummy_inputs)
        inputs_spec, outputs_spec = _build_input_output_specs(text_cfg)

        ordered_input_names = list(inputs_spec.keys())
        flatten_inputs = _flattenize_inputs(list(dummy_inputs.values()))

        if len(flatten_inputs) != len(ordered_input_names):
            raise RuntimeError(
                f"Mismatch: {len(flatten_inputs)} flattened tensors vs "
                f"{len(ordered_input_names)} input names in spec"
            )

        input_info = _get_input_info(ordered_input_names, inputs_spec, flatten_inputs)

        extensions = [
            ModuleExtension(RecurrentAttentionCell, "RecurrentAttentionCellOp"),
            ConversionExtension("RecurrentAttentionCellOp", convert_recurrent_attention_cell),
        ]

        logger.info("Converting to OpenVINO IR with Loop nodes (non-stateful) ...")
        ov_model = ov.convert_model(
            model,
            example_input=dummy_inputs,
            input=[(item.shape, item.type) for item in input_info],
            extension=extensions,
        )

    ov_model.validate_nodes_and_infer_types()

    # 8. Name outputs
    output_names = list(outputs_spec.keys())
    for idx, out_tensor in enumerate(ov_model.outputs):
        if idx < len(output_names):
            out_tensor.get_tensor().set_names({output_names[idx]})

    # 9. Name inputs
    input_names = [item.name for item in input_info]
    for idx, inp_tensor in enumerate(ov_model.inputs):
        if idx < len(input_names):
            inp_tensor.get_tensor().set_names({input_names[idx]})

    # 10. NO stateful transformation — IR saved with explicit cache I/O + Loop nodes.
    loop_count = sum(
        1 for op in ov_model.get_ordered_ops()
        if op.get_type_name() in ("Loop", "TensorIterator")
    )
    logger.info("Non-stateful IR: %d Loop nodes, %d inputs, %d outputs",
                loop_count, len(ov_model.inputs), len(ov_model.outputs))

    # 11. Save
    xml_path = out_path / "openvino_model.xml"
    logger.info("Saving non-stateful Loop IR to %s (compress_to_fp16=%s) ...", xml_path, compress_to_fp16)
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)

    # 11b. Extract embedding table
    embed_table = model.model.embed_tokens.weight.detach().cpu().numpy()
    embed_path = out_path / "embed_tokens.npy"
    np.save(str(embed_path), embed_table.astype(np.float16))
    logger.info("Saved embed_tokens.npy: shape=%s, dtype=float16", embed_table.shape)

    # 12. Copy tokenizer / config files
    _copy_tokenizer_and_config(model_path, out_path)

    # Cleanup
    _clear_class_registry()
    del ov_model
    del model

    logger.info("Non-stateful Loop IR export complete: %s", out_path)
