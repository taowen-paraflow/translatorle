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
from .stateful import patch_stateful_hybrid_ssm

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
