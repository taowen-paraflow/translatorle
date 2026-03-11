"""NPU v2 export: 6 subgraph IRs with host-side rotary precomputation.

Key difference from export_model_multisubgraph in export.py: rotary embedding
cos/sin are explicit model inputs (precomputed on host), not computed inside
the subgraph.  This avoids tracing failures from the rotary_emb module inside
SubgraphModule, which can cause issues with torch.jit.trace / ov.convert_model
when the rotary_emb module uses mRoPE position indexing internally.

Architecture: Qwen3.5-0.8B has 24 layers = [GDN, GDN, GDN, FullAttn] x 6.
We split into 6 subgraphs of 4 layers each.  Between subgraphs, hidden_states
return to CPU as FP32, limiting NPU FP16 error accumulation to 4 layers.

Each subgraph IR has:
  Inputs:  hidden_states, cos, sin, attention_mask, cache_params.*
  Outputs: [logits if final], hidden_states, cache_params.present.*, gdn_intermediates

The host precomputes cos/sin from saved inv_freq + mRoPE section config and
feeds them to every subgraph.  This also enables FP32 rotary computation on
the host, avoiding FP16 rotary precision loss on NPU.

Output files:
  subgraph_0.xml .. subgraph_5.xml  - OpenVINO IR per subgraph
  embed_tokens.npy                  - embedding table (FP16)
  rotary_inv_freq.npy               - rotary inverse frequencies (FP32)
  rotary_config.json                - mRoPE parameters
  config.json, tokenizer.json, ...  - tokenizer and model config
"""

import json
import logging
import types
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM

import openvino as ov

from .export import (
    _copy_tokenizer_and_config,
    _clear_class_registry,
    _classify_layers,
    _hybrid_intermediates,
    ov_causal_conv1d,
    qwen3_5_gated_delta_net_forward,
    patched_recurrent_gated_delta_rule_hybrid_step,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_npu_v2(
    model_dir: str,
    output_dir: str,
    compress_to_fp16: bool = True,
    max_cache_len: int = 256,
) -> None:
    """Export Qwen3.5 as 6 subgraph IRs with host-side rotary precomputation.

    Each subgraph takes precomputed cos/sin as explicit inputs instead of
    computing rotary embeddings internally.  This avoids tracing failures
    from the rotary_emb module and lets the host compute mRoPE in FP32.

    Args:
        model_dir: Path to the HuggingFace Qwen3.5 model checkpoint.
        output_dir: Output directory for the exported subgraph IRs.
        compress_to_fp16: Whether to compress IR weights to FP16.
        max_cache_len: Maximum KV cache length for static allocation.
    """
    from .config import (
        NPU_MAX_CACHE_LEN,
        MULTISUB_LAYERS_PER_SUBGRAPH,
        MULTISUB_NUM_SUBGRAPHS,
    )

    if max_cache_len <= 0:
        max_cache_len = NPU_MAX_CACHE_LEN

    model_path = Path(model_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load PyTorch model
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Import transformers 5.x GDN classes (must be inside function)
    # ------------------------------------------------------------------
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DynamicCache,
        Qwen3_5GatedDeltaNet,
    )

    # ------------------------------------------------------------------
    # 3. Classify layers and compute architecture dimensions
    # ------------------------------------------------------------------
    linear_layer_indices, attention_layer_indices = _classify_layers(text_cfg)

    num_layers = text_cfg.num_hidden_layers              # 24
    layers_per_sub = MULTISUB_LAYERS_PER_SUBGRAPH        # 4
    num_subs = MULTISUB_NUM_SUBGRAPHS                    # 6
    hidden_size = text_cfg.hidden_size                   # 1024

    # GDN dimensions
    linear_num_key_heads = text_cfg.linear_num_key_heads          # 16
    linear_key_head_dim = text_cfg.linear_key_head_dim            # 128
    linear_value_head_dim = text_cfg.linear_value_head_dim        # 128
    linear_num_value_heads = text_cfg.linear_num_value_heads      # 16
    linear_conv_kernel_dim = text_cfg.linear_conv_kernel_dim      # 4
    conv_dim = (
        linear_num_key_heads * linear_key_head_dim * 2
        + linear_num_value_heads * linear_value_head_dim
    )  # 6144

    # Full attention KV dimensions
    num_key_value_heads = text_cfg.num_key_value_heads  # 4
    head_dim = getattr(
        text_cfg, "head_dim",
        hidden_size // text_cfg.num_attention_heads,
    )  # 256

    layer_types = text_cfg.layer_types

    # ------------------------------------------------------------------
    # 4. Probe rotary embedding to determine cos/sin output shape
    # ------------------------------------------------------------------
    with torch.no_grad():
        dummy_h = torch.zeros(1, 1, hidden_size)
        dummy_pos = torch.zeros(3, 1, 1, dtype=torch.int64)
        cos_probe, sin_probe = model.model.rotary_emb(dummy_h, dummy_pos)
        rotary_dim = cos_probe.shape[-1]

    logger.info(
        "Rotary embedding probe: cos shape=%s, rotary_dim=%d",
        list(cos_probe.shape), rotary_dim,
    )

    # ------------------------------------------------------------------
    # 5. SubgraphCacheWrapV2: maps absolute layer indices to compact
    #    subgraph indices for the cache lookup
    # ------------------------------------------------------------------
    class SubgraphCacheWrapV2(Qwen3_5DynamicCache):
        """Thin cache adapter mapping absolute layer indices to compact
        subgraph-local indices."""

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
            # Concatenate: [1, H, MAX_LEN, D] + [1, H, 1, D]
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
    # 6. SubgraphModuleV2: wraps a group of 4 layers WITHOUT rotary_emb.
    #    cos/sin are explicit forward() inputs.
    # ------------------------------------------------------------------
    class SubgraphModuleV2(torch.nn.Module):
        """Subgraph wrapper that takes precomputed cos/sin instead of
        computing rotary embeddings internally.

        forward() signature:
            (hidden_states, cos, sin, attention_mask, *cache_params) -> tuple

        No rotary_emb module is included.  The host precomputes cos/sin
        using the saved inv_freq and mRoPE config.
        """

        def __init__(self, layers_list, final_norm, lm_head,
                     sub_text_cfg, n_linear, n_attn,
                     linear_idx_map, attn_idx_map, is_final):
            super().__init__()
            self.sub_layers = torch.nn.ModuleList(layers_list)
            self.final_norm = final_norm if is_final else None
            self.lm_head = lm_head if is_final else None
            self._cfg = sub_text_cfg
            self._n_linear = n_linear
            self._n_attn = n_attn
            self._linear_idx_map = linear_idx_map   # {abs_layer_idx: compact_idx}
            self._attn_idx_map = attn_idx_map       # {abs_layer_idx: compact_idx}
            self._is_final = is_final

        def forward(self, hidden_states, cos, sin, attention_mask, *cache_params):
            # Clear stale intermediates from previous subgraph
            _hybrid_intermediates.clear()

            # --- Unpack cache_params into lists ---
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

            cache = SubgraphCacheWrapV2(
                self._cfg, conv_states, recurrent_states,
                key_cache, value_cache,
                self._linear_idx_map, self._attn_idx_map,
            )

            # Position embeddings from explicit inputs (NOT computed internally).
            # GDN layers ignore position_embeddings; FullAttn layers use them
            # for rotary on Q and K.
            position_embeddings = (cos, sin)

            # --- Run through the subgraph layers ---
            h = hidden_states
            for layer in self.sub_layers:
                layer_out = layer(
                    h,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=None,   # not needed when position_embeddings provided
                    past_key_values=cache,
                    cache_position=None,
                )
                # Qwen3_5DecoderLayer.forward returns a single tensor, not a tuple
                h = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]

            # --- Collect outputs ---
            results = []

            # Final subgraph produces logits
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
                # Output ONLY the new KV entry (last position after concat).
                # Python-side inference manages the full cache buffer.
                results.append(cache.key_cache[i][:, :, -1:, :])
                results.append(cache.value_cache[i][:, :, -1:, :])

            # GDN intermediates for CPU FP32 shadow state update
            for g_t, k_t, v_t, beta_t in _hybrid_intermediates:
                results.extend([g_t, k_t, v_t, beta_t])

            return tuple(results)

    # ------------------------------------------------------------------
    # 7. Patch GDN layers for single-step + intermediates capture
    # ------------------------------------------------------------------
    gdn_originals = []
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet):
            gdn = layer.linear_attn
            gdn_originals.append((gdn, gdn.forward))
            gdn.forward = types.MethodType(qwen3_5_gated_delta_net_forward, gdn)
            gdn.recurrent_gated_delta_rule = patched_recurrent_gated_delta_rule_hybrid_step

    try:
        # --------------------------------------------------------------
        # 8. Export each of the 6 subgraphs
        # --------------------------------------------------------------
        for sub_idx in range(num_subs):
            start_layer = sub_idx * layers_per_sub
            end_layer = (sub_idx + 1) * layers_per_sub
            is_last = (sub_idx == num_subs - 1)

            logger.info(
                "Exporting subgraph %d (layers %d-%d) ...",
                sub_idx, start_layer, end_layer - 1,
            )

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
            compact = 0
            for i in range(start_layer, end_layer):
                if layer_types[i] == "linear_attention":
                    linear_mapping[i] = compact
                    compact += 1

            attn_mapping = {}
            compact = 0
            for i in range(start_layer, end_layer):
                if layer_types[i] == "full_attention":
                    attn_mapping[i] = compact
                    compact += 1

            # --- Create SubgraphModuleV2 wrapper (NO rotary_emb) ---
            wrapper = SubgraphModuleV2(
                layers_list=[model.model.layers[i] for i in range(start_layer, end_layer)],
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
            cos_dummy = torch.zeros(1, 1, rotary_dim, dtype=torch.float32)
            sin_dummy = torch.zeros(1, 1, rotary_dim, dtype=torch.float32)
            attention_mask_dummy = torch.zeros(
                1, 1, 1, max_cache_len + 1, dtype=torch.float32,
            )

            # Cache params: interleaved (conv, recurrent) per GDN,
            # then (key, value) per attention layer.
            sub_cache_tensors = []
            for _ in range(num_sub_linear):
                sub_cache_tensors.append(
                    torch.zeros(1, conv_dim, linear_conv_kernel_dim,
                                dtype=torch.float32))
                sub_cache_tensors.append(
                    torch.zeros(1, linear_num_key_heads, linear_key_head_dim,
                                linear_value_head_dim, dtype=torch.float32))
            for _ in range(num_sub_attn):
                sub_cache_tensors.append(
                    torch.zeros(1, num_key_value_heads, max_cache_len, head_dim,
                                dtype=torch.float32))
                sub_cache_tensors.append(
                    torch.zeros(1, num_key_value_heads, max_cache_len, head_dim,
                                dtype=torch.float32))

            # Build example_input as a FLAT tuple of positional args.
            # ov.convert_model calls wrapper.forward(*example_input), so
            # each cache tensor becomes an individual positional arg
            # matching the *cache_params varargs.
            example_input = (
                hidden_states_dummy,
                cos_dummy,
                sin_dummy,
                attention_mask_dummy,
            ) + tuple(sub_cache_tensors)

            # Clear intermediates before tracing
            _hybrid_intermediates.clear()

            # --- Build input / output name specs ---
            inputs_spec = OrderedDict()
            inputs_spec["hidden_states"] = {}
            inputs_spec["cos"] = {}
            inputs_spec["sin"] = {}
            inputs_spec["attention_mask"] = {}
            for local_idx in range(num_sub_linear):
                inputs_spec[f"cache_params.past.conv.{local_idx}"] = {}
                inputs_spec[f"cache_params.past.recurrent.{local_idx}"] = {}
            for local_idx in range(num_sub_attn):
                inputs_spec[f"cache_params.past.key.{local_idx}"] = {}
                inputs_spec[f"cache_params.past.value.{local_idx}"] = {}

            outputs_spec = OrderedDict()
            if is_last:
                outputs_spec["logits"] = {}
            # Use "output_hidden_states" to avoid name collision with input "hidden_states"
            outputs_spec["output_hidden_states"] = {}
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

            # --- Convert to OpenVINO IR with explicit static shapes ---
            # Build input shape specs to force all-static shapes (NPU requirement)
            input_specs = []
            for tensor in example_input:
                shape = ov.PartialShape(list(tensor.shape))
                etype = ov.Type.f32
                input_specs.append((shape, etype))

            with torch.no_grad():
                logger.info("Converting subgraph %d to OpenVINO IR ...", sub_idx)
                ov_model = ov.convert_model(
                    wrapper,
                    example_input=example_input,
                    input=input_specs,
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

            # Save subgraph IR
            xml_path = out_path / f"subgraph_{sub_idx}.xml"
            logger.info("Saving subgraph %d to %s ...", sub_idx, xml_path)
            ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)

            del ov_model
            _hybrid_intermediates.clear()

    finally:
        # Restore GDN layers to original state
        for gdn, orig_fwd in gdn_originals:
            gdn.forward = orig_fwd
            if hasattr(gdn, "recurrent_gated_delta_rule"):
                del gdn.recurrent_gated_delta_rule

    # ------------------------------------------------------------------
    # 9. Save embedding table (FP16)
    # ------------------------------------------------------------------
    embed_table = model.model.embed_tokens.weight.detach().cpu().numpy()
    embed_path = out_path / "embed_tokens.npy"
    np.save(str(embed_path), embed_table.astype(np.float16))
    logger.info("Saved embed_tokens.npy: shape=%s, dtype=float16", embed_table.shape)

    # ------------------------------------------------------------------
    # 10. Save rotary parameters for host-side precomputation
    # ------------------------------------------------------------------
    rotary_emb = model.model.rotary_emb

    # Extract inv_freq (registered buffer in the rotary_emb module).
    # If not available as a buffer, recompute from rope_theta.
    if hasattr(rotary_emb, "inv_freq") and rotary_emb.inv_freq is not None:
        inv_freq = rotary_emb.inv_freq.detach().cpu().numpy()
    else:
        rope_theta = float(getattr(text_cfg, "rope_theta", 1_000_000))
        # Standard RoPE: inv_freq = 1 / (theta ^ (2i / dim))
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
        )
        inv_freq = inv_freq.astype(np.float32)
        logger.warning(
            "rotary_emb.inv_freq not found; recomputed from rope_theta=%s", rope_theta,
        )

    inv_freq_path = out_path / "rotary_inv_freq.npy"
    np.save(str(inv_freq_path), inv_freq)
    logger.info(
        "Saved rotary_inv_freq.npy: shape=%s, dtype=%s", inv_freq.shape, inv_freq.dtype,
    )

    # Save mRoPE config as JSON for the host-side rotary computation.
    mrope_section = getattr(text_cfg, "mrope_section", [11, 11, 10])
    rope_theta = float(getattr(text_cfg, "rope_theta", 1_000_000))
    rope_scaling = getattr(text_cfg, "rope_scaling", None)

    # Get the actual attention_scaling from the model's rotary embedding
    attention_scaling = float(getattr(rotary_emb, "attention_scaling", 1.0))

    rotary_config = {
        "mrope_section": mrope_section,
        "rope_theta": rope_theta,
        "rotary_dim": rotary_dim,
        "head_dim": head_dim,
        "attention_scaling": attention_scaling,
        "max_cache_len": max_cache_len,
        "num_subgraphs": num_subs,
        "layers_per_subgraph": layers_per_sub,
    }
    if rope_scaling is not None:
        rotary_config["rope_scaling"] = rope_scaling

    rotary_config_path = out_path / "rotary_config.json"
    with open(rotary_config_path, "w") as f:
        json.dump(rotary_config, f, indent=2)
    logger.info("Saved rotary_config.json: %s", rotary_config)

    # ------------------------------------------------------------------
    # 11. Copy tokenizer / config files
    # ------------------------------------------------------------------
    _copy_tokenizer_and_config(model_path, out_path)

    # ------------------------------------------------------------------
    # 12. Cleanup
    # ------------------------------------------------------------------
    _clear_class_registry()
    del model

    logger.info(
        "NPU v2 export complete (%d subgraphs, rotary_dim=%d, max_cache_len=%d): %s",
        num_subs, rotary_dim, max_cache_len, out_path,
    )
