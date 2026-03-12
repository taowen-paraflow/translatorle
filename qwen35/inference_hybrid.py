"""Hybrid GPU+NPU inference for Qwen3.5.

Orchestrates 13 subgraph IRs exported by export_hybrid.py:
  - 6 GDN blocks on GPU (contain Loop nodes, need FP32)
  - 6 Attention blocks on NPU or GPU (standard SDPA, FP16 ok)
  - 1 Head block on GPU

Run (root venv):
  powershell.exe -Command 'cd C:\\Apps\\translatorle; uv run python -m qwen35.inference_hybrid --prompt "Hello" --device HYBRID'
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Union

import numpy as np
import openvino as ov
from openvino._offline_transformations import apply_make_stateful_transformation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main hybrid model class
# ---------------------------------------------------------------------------

class Qwen35HybridModel:
    """Hybrid GPU+NPU inference for Qwen3.5 using 13 subgraph IRs."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        gdn_device: str = "GPU",
        attn_device: str = "NPU",
        head_device: str = "GPU",
        attn_past_seq: int = 256,
        attn_stateful: bool = True,
        prefill_chunk_size: int = 16,
    ):
        """Load and compile all subgraphs.

        Args:
            model_dir: Directory containing the hybrid IR files.
            gdn_device: Device for GDN blocks (default: GPU).
            attn_device: Device for attention blocks (default: NPU).
            head_device: Device for the head block (default: GPU).
            embed_device: Device for the embedding block (default: GPU).
            attn_past_seq: Static past_seq length for NPU attention blocks.
                NPU requires static shapes; KV caches are padded to this size.
        """
        model_dir = Path(model_dir)
        core = ov.Core()

        # Enable model caching — NPU compilation is ~10s/block (60s total).
        # After first run, cached blobs make subsequent starts <1s.
        cache_dir = str(model_dir / "cache")
        core.set_property({"CACHE_DIR": cache_dir})

        self._attn_past_seq = attn_past_seq
        self._gdn_device = gdn_device
        self._attn_device = attn_device

        # Load config to get architecture params
        import json
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", cfg)

        self._hidden_size = text_cfg["hidden_size"]
        self._num_kv_heads = text_cfg["num_key_value_heads"]
        self._head_dim = text_cfg.get("head_dim", self._hidden_size // text_cfg["num_attention_heads"])
        self._num_v_heads = text_cfg["linear_num_value_heads"]
        self._k_head_dim = text_cfg["linear_key_head_dim"]
        self._v_head_dim = text_cfg["linear_value_head_dim"]
        self._conv_dim = (
            text_cfg["linear_num_key_heads"] * text_cfg["linear_key_head_dim"] * 2
            + text_cfg["linear_num_value_heads"] * text_cfg["linear_value_head_dim"]
        )
        self._conv_kernel = text_cfg["linear_conv_kernel_dim"]

        # Detect number of blocks
        num_blocks = 0
        while (model_dir / f"gdn_block_{num_blocks}.xml").exists():
            num_blocks += 1
        self._num_blocks = num_blocks
        logger.info("Found %d GDN+Attn block pairs", num_blocks)

        # --- Compile GDN blocks (GPU, stateful — state persists in GPU memory) ---
        logger.info("Compiling %d GDN blocks on %s (stateful) ...", num_blocks, gdn_device)
        self._gdn_models: List[ov.CompiledModel] = []
        self._gdn_requests: List = []
        t0 = time.time()

        # Map state inputs to outputs for stateful transformation
        state_in_names = ["in_conv0", "in_rec0", "in_conv1", "in_rec1", "in_conv2", "in_rec2"]
        state_out_names = ["out_conv0", "out_rec0", "out_conv1", "out_rec1", "out_conv2", "out_rec2"]
        state_map = dict(zip(state_in_names, state_out_names))

        for i in range(num_blocks):
            ir = core.read_model(str(model_dir / f"gdn_block_{i}.xml"))
            # Convert explicit state I/O to OpenVINO stateful (ReadValue/Assign).
            # After this, state lives on-device; only hidden+mask are passed each call.
            apply_make_stateful_transformation(ir, state_map)
            if gdn_device in ("GPU", "NPU"):
                ir = self._add_f32_output_conversion(ir)
            if i == 0:
                logger.info("  GDN block inputs (stateful): %s",
                            [(inp.get_any_name(), str(inp.partial_shape), str(inp.element_type)) for inp in ir.inputs])
                logger.info("  GDN block sinks: %d", len(ir.get_sinks()))
            compiled = core.compile_model(ir, gdn_device)
            self._gdn_models.append(compiled)
            self._gdn_requests.append(compiled.create_infer_request())
        logger.info("  GDN compilation: %.1fs", time.time() - t0)

        # Initialize GDN stateful variables with correct shapes
        # (dynamic dims default to shape=0 after stateful transform)
        self._init_gdn_states()

        # --- Compile Attention blocks ---
        # Fixed-size KV cache (ScatterUpdate) enables stateful on ALL devices.
        # NPU: try stateful first; if it fails at runtime, can fall back to explicit.
        self._attn_stateful = attn_stateful
        mode_str = "stateful" if self._attn_stateful else "explicit I/O"
        logger.info("Compiling %d Attn blocks on %s (%s, fixed KV=%d) ...",
                     num_blocks, attn_device, mode_str, attn_past_seq)
        self._attn_models: List[ov.CompiledModel] = []
        self._attn_requests: List = []
        t0 = time.time()

        attn_state_map = {"in_key_cache": "out_key_cache", "in_value_cache": "out_value_cache"}

        for i in range(num_blocks):
            ir = core.read_model(str(model_dir / f"attn_block_{i}.xml"))
            if attn_device == "NPU":
                ir = self._reshape_attn_static(ir, attn_past_seq)
            if self._attn_stateful:
                apply_make_stateful_transformation(ir, attn_state_map)
            if attn_device in ("GPU", "NPU"):
                ir = self._add_f32_output_conversion(ir)
            if i == 0:
                logger.info("  Attn block inputs: %s",
                            [(inp.get_any_name(), str(inp.partial_shape), str(inp.element_type)) for inp in ir.inputs])
                if self._attn_stateful:
                    logger.info("  Attn block sinks: %d", len(ir.get_sinks()))
            npu_config = {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"} if attn_device == "NPU" else {}
            compiled = core.compile_model(ir, attn_device, npu_config)
            self._attn_models.append(compiled)
            self._attn_requests.append(compiled.create_infer_request())
        logger.info("  Attn compilation: %.1fs", time.time() - t0)

        if self._attn_stateful:
            self._init_attn_states()
        else:
            self._init_kv_caches()

        # --- Compile prefill Attention blocks (NPU, S=prefill_chunk_size) ---
        self._prefill_chunk_size = prefill_chunk_size
        self._attn_prefill_requests: List = []
        if (prefill_chunk_size > 1 and attn_device == "NPU"
                and not self._attn_stateful):
            logger.info("Compiling %d prefill Attn blocks on NPU (S=%d) ...",
                        num_blocks, prefill_chunk_size)
            t0 = time.time()
            for i in range(num_blocks):
                ir = core.read_model(str(model_dir / f"attn_block_{i}.xml"))
                ir = self._reshape_attn_static(ir, attn_past_seq, seq_len=prefill_chunk_size)
                ir = self._add_f32_output_conversion(ir)
                npu_config = {"NPU_COMPILER_TYPE": "PREFER_PLUGIN"}
                compiled = core.compile_model(ir, "NPU", npu_config)
                self._attn_prefill_requests.append(compiled.create_infer_request())
            logger.info("  Prefill Attn compilation: %.1fs", time.time() - t0)

        # --- Compile Head block (GPU) ---
        logger.info("Compiling Head on %s ...", head_device)
        t0 = time.time()
        head_ir = core.read_model(str(model_dir / "head.xml"))
        if head_device in ("GPU", "NPU"):
            head_ir = self._add_f32_output_conversion(head_ir)
        self._head_model = core.compile_model(head_ir, head_device)
        self._head_request = self._head_model.create_infer_request()
        logger.info("  Head compilation: %.1fs", time.time() - t0)

        # --- Embedding table (load as float32 to avoid indexing issues) ---
        embed_path = model_dir / "embed_tokens.npy"
        self._embed_table = np.load(str(embed_path)).astype(np.float32)
        logger.info("Loaded embed_tokens: shape=%s dtype=%s", self._embed_table.shape, self._embed_table.dtype)

        # --- Tokenizer ---
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

        self._past_length = 0

        # Profiling instrumentation (disable for clean benchmarks)
        self._profiling = False
        self._reset_profile_stats()

    def _reset_profile_stats(self):
        """Reset accumulated profiling counters."""
        self._profile_stats = {
            'embed': [],
            'numpy_prep': [],
            'gdn_blocks': [[] for _ in range(self._num_blocks)],
            'attn_blocks': [[] for _ in range(self._num_blocks)],
            'head': [],
            'total': [],
        }

    def _print_profile_summary(self):
        """Print per-token profiling summary table."""
        stats = self._profile_stats
        n = len(stats['total'])
        if n == 0:
            return

        def avg_ms(lst):
            return sum(lst) / len(lst) * 1000 if lst else 0.0

        embed_avg = avg_ms(stats['embed'])
        numpy_avg = avg_ms(stats['numpy_prep'])
        head_avg = avg_ms(stats['head'])
        total_avg = avg_ms(stats['total'])

        gdn_avgs = [avg_ms(stats['gdn_blocks'][i]) for i in range(self._num_blocks)]
        attn_avgs = [avg_ms(stats['attn_blocks'][i]) for i in range(self._num_blocks)]
        gdn_total_avg = sum(gdn_avgs)
        attn_total_avg = sum(attn_avgs)

        overhead = total_avg - gdn_total_avg - attn_total_avg - head_avg - embed_avg

        logger.info("--- Profile (%d forward calls) ---", n)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Embed lookup", embed_avg, embed_avg / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Numpy prep", numpy_avg, numpy_avg / total_avg * 100 if total_avg else 0)
        for i in range(self._num_blocks):
            logger.info("  %-22s %8.3f ms  (%5.1f%%)", f"GDN block {i}", gdn_avgs[i], gdn_avgs[i] / total_avg * 100 if total_avg else 0)
            logger.info("  %-22s %8.3f ms  (%5.1f%%)", f"Attn block {i}", attn_avgs[i], attn_avgs[i] / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Head", head_avg, head_avg / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms", "---", 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "GDN total", gdn_total_avg, gdn_total_avg / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Attn total", attn_total_avg, attn_total_avg / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "OV inference total", gdn_total_avg + attn_total_avg + head_avg, (gdn_total_avg + attn_total_avg + head_avg) / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms  (%5.1f%%)", "Python overhead", overhead, overhead / total_avg * 100 if total_avg else 0)
        logger.info("  %-22s %8.3f ms", "TOTAL per token", total_avg)

    def _init_gdn_states(self):
        """Initialize GDN stateful variables with correct shapes.

        After apply_make_stateful_transformation, dynamic dims default to
        shape=0. Must manually set each state tensor to the correct shape.
        Cannot use s.reset() — it restores shape=0.
        """
        conv_shape = (1, self._conv_dim, self._conv_kernel)
        rec_shape = (1, self._num_v_heads, self._k_head_dim, self._v_head_dim)
        for req in self._gdn_requests:
            for s in req.query_state():
                if "conv" in s.name:
                    s.state = ov.Tensor(np.zeros(conv_shape, dtype=np.float32))
                elif "rec" in s.name:
                    s.state = ov.Tensor(np.zeros(rec_shape, dtype=np.float32))

    def _init_attn_states(self):
        """Initialize attention KV cache stateful variables.

        Fixed-size KV cache: shape is always [B, H, MAX_CACHE_LEN, D].
        All positions start as zeros; the attention mask ensures only
        written positions are attended to.
        """
        kv_shape = (1, self._num_kv_heads, self._attn_past_seq, self._head_dim)
        for req in self._attn_requests:
            for s in req.query_state():
                s.state = ov.Tensor(np.zeros(kv_shape, dtype=np.float32))

    @staticmethod
    def _add_f32_output_conversion(ir):
        """Add FP16->FP32 output conversion (GPU/NPU don't auto-promote)."""
        from openvino.preprocess import PrePostProcessor
        ppp = PrePostProcessor(ir)
        for i in range(len(ir.outputs)):
            ppp.output(i).tensor().set_element_type(ov.Type.f32)
        return ppp.build()

    def _reshape_attn_static(self, ir, past_seq: int, seq_len: int = 1):
        """Reshape attention IR to fully static shapes for NPU compilation.

        Uses index-based access to avoid name collision issues.
        With fixed-size KV cache, cache dim is already past_seq (MAX_CACHE_LEN).
        """
        B, S = 1, seq_len
        # Index-based: 0=hidden, 1=position_ids, 2=key_cache, 3=value_cache,
        #              4=cache_position, 5=attention_mask
        shapes = {}
        for i, inp in enumerate(ir.inputs):
            name = inp.get_any_name()
            if i == 0:  # hidden: [B, S, hidden_size]
                shapes[name] = ov.PartialShape([B, S, self._hidden_size])
            elif i == 1:  # position_ids: [3, B, S]
                shapes[name] = ov.PartialShape([3, B, S])
            elif i in (2, 3):  # key/value cache: [B, num_kv_heads, past_seq, head_dim]
                shapes[name] = ov.PartialShape([B, self._num_kv_heads, past_seq, self._head_dim])
            elif i == 4:  # cache_position: [S]
                shapes[name] = ov.PartialShape([S])
            elif i == 5:  # attention_mask: [B, 1, S, past_seq]
                shapes[name] = ov.PartialShape([B, 1, S, past_seq])
        ir.reshape(shapes)
        return ir

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def _run_gdn_block(self, block_idx: int, hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Run one GDN block (3 layers) on GPU.

        State (conv/recurrent) persists in GPU memory via OpenVINO stateful
        variables — only hidden + mask are transferred each call.
        """
        req = self._gdn_requests[block_idx]
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(attention_mask)))
        req.infer()
        return req.get_output_tensor(0).data.copy()

    def _run_attn_block(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray,
                        use_prefill: bool = False) -> np.ndarray:
        """Run one Attention block (1 layer) with fixed-size KV cache."""
        if self._attn_stateful:
            return self._run_attn_block_stateful(block_idx, hidden, position_ids)
        return self._run_attn_block_explicit(block_idx, hidden, position_ids, use_prefill=use_prefill)

    def _run_attn_block_stateful(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """Run attention with stateful fixed-size KV cache.

        KV cache persists on device via ReadValue/Assign (constant shape
        [B, H, MAX_CACHE_LEN, D]). Only hidden, position_ids, cache_position,
        and attention_mask are transferred each call.
        """
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]

        cache_position = np.arange(
            self._past_length, self._past_length + seq_len, dtype=np.int64)
        attention_mask = self._build_attn_mask(batch_size, seq_len)

        # After stateful transform: inputs are [hidden, position_ids, cache_position, attention_mask]
        req = self._attn_requests[block_idx]
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(position_ids)))
        req.set_input_tensor(2, ov.Tensor(np.ascontiguousarray(cache_position)))
        req.set_input_tensor(3, ov.Tensor(np.ascontiguousarray(attention_mask)))
        req.infer()

        return req.get_output_tensor(0).data.copy().reshape(batch_size, seq_len, -1)

    def _run_attn_block_explicit(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray,
                                  use_prefill: bool = False) -> np.ndarray:
        """Run attention with explicit fixed-size KV cache I/O.

        KV cache is transferred each step but no padding/compaction needed
        (fixed shape [B, H, MAX_CACHE_LEN, D] in and out).
        """
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]

        cache_position = np.arange(
            self._past_length, self._past_length + seq_len, dtype=np.int64)
        attention_mask = self._build_attn_mask(batch_size, seq_len)

        # Explicit I/O: inputs are [hidden, position_ids, key_cache, value_cache, cache_position, attention_mask]
        # Use prefill infer request (S=chunk_size) when available, else decode request (S=1)
        if use_prefill and self._attn_prefill_requests:
            req = self._attn_prefill_requests[block_idx]
        else:
            req = self._attn_requests[block_idx]
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(position_ids)))
        req.set_input_tensor(2, ov.Tensor(np.ascontiguousarray(self._kv_caches[block_idx][0])))
        req.set_input_tensor(3, ov.Tensor(np.ascontiguousarray(self._kv_caches[block_idx][1])))
        req.set_input_tensor(4, ov.Tensor(np.ascontiguousarray(cache_position)))
        req.set_input_tensor(5, ov.Tensor(np.ascontiguousarray(attention_mask)))
        req.infer()

        hidden_out = req.get_output_tensor(0).data.copy().reshape(batch_size, seq_len, -1)
        # Update stored KV caches (output has same shape as input)
        self._kv_caches[block_idx][0] = req.get_output_tensor(1).data.copy()
        self._kv_caches[block_idx][1] = req.get_output_tensor(2).data.copy()

        return hidden_out

    def _build_attn_mask(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Build 4D causal attention mask for fixed-size KV cache.

        Layout of key positions in the fixed buffer [0..MAX_CACHE_LEN-1]:
          [0 .. past_length-1]                  -> previously written tokens (attend)
          [past_length .. past_length+seq_len-1] -> current token(s) being written (attend)
          [past_length+seq_len .. MAX_CACHE_LEN-1] -> empty/future (masked)

        For seq_len > 1 (chunked prefill), applies causal masking so each
        query can only attend to positions up to its own position.

        Returns:
            attention_mask: shape [B, 1, seq_len, MAX_CACHE_LEN], dtype float32.
            0.0 = attend, -65504.0 = ignore.
        """
        MASK_VALUE = np.float32(-65504.0)  # min fp16, guarantees ~0 softmax weight
        max_len = self._attn_past_seq

        # For each query q in [0..seq_len-1]:
        #   attend to positions 0..past_length+q (inclusive)
        #   mask positions past_length+q+1..MAX_CACHE_LEN-1
        col = np.arange(max_len)
        query_ends = np.arange(self._past_length + 1, self._past_length + seq_len + 1)
        attend = col[None, :] < query_ends[:, None]  # [seq_len, max_len]
        mask = np.where(attend, np.float32(0.0), MASK_VALUE)[None, None, :, :]  # [1, 1, seq_len, max_len]
        if batch_size > 1:
            mask = np.broadcast_to(mask, (batch_size, 1, seq_len, max_len)).copy()
        return mask

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Run one forward pass through all subgraphs.

        Args:
            token_ids: Shape [1, seq_len] int64.

        Returns:
            logits: Shape [1, seq_len, vocab_size] float32.
        """
        profiling = self._profiling
        if profiling:
            t_total_start = time.perf_counter()

        batch_size, seq_len = token_ids.shape

        # Embed tokens — already float32, index gives [B, seq_len, hidden]
        if profiling:
            t0 = time.perf_counter()
        embeds = self._embed_table[token_ids]
        if profiling:
            self._profile_stats['embed'].append(time.perf_counter() - t0)

        # Numpy prep: attention mask + position IDs
        if profiling:
            t0 = time.perf_counter()
        # Attention mask for GDN blocks
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        # Position IDs for attention blocks (mRoPE: [3, B, seq_len])
        positions = np.arange(self._past_length, self._past_length + seq_len, dtype=np.int64)
        position_ids = np.tile(positions[np.newaxis, np.newaxis, :], (3, batch_size, 1))
        if profiling:
            self._profile_stats['numpy_prep'].append(time.perf_counter() - t0)

        hidden = embeds

        # Auto-detect prefill mode: use prefill requests when seq_len matches chunk size
        use_prefill = (seq_len == self._prefill_chunk_size
                       and len(self._attn_prefill_requests) > 0)

        for i in range(self._num_blocks):
            if profiling:
                t0 = time.perf_counter()
            hidden = self._run_gdn_block(i, hidden, attention_mask)
            if profiling:
                self._profile_stats['gdn_blocks'][i].append(time.perf_counter() - t0)

            if profiling:
                t0 = time.perf_counter()
            hidden = self._run_attn_block(i, hidden, position_ids, use_prefill=use_prefill)
            if profiling:
                self._profile_stats['attn_blocks'][i].append(time.perf_counter() - t0)

        # Head
        if profiling:
            t0 = time.perf_counter()
        self._head_request.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        self._head_request.infer()
        logits = self._head_request.get_output_tensor(0).data.copy()
        if profiling:
            self._profile_stats['head'].append(time.perf_counter() - t0)

        self._past_length += seq_len

        if profiling:
            self._profile_stats['total'].append(time.perf_counter() - t_total_start)

        return logits

    # -----------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------

    def reset(self):
        """Reset all states for a new generation."""
        if self._attn_stateful:
            self._init_attn_states()  # Reset attention KV cache stateful variables
        else:
            self._init_kv_caches()    # Reset explicit KV cache buffers
        self._init_gdn_states()       # Reset GDN stateful variables (can't use s.reset())
        self._past_length = 0

    def _init_kv_caches(self):
        """Initialize explicit KV cache buffers (non-stateful mode)."""
        kv_shape = (1, self._num_kv_heads, self._attn_past_seq, self._head_dim)
        self._kv_caches = [
            [np.zeros(kv_shape, dtype=np.float32), np.zeros(kv_shape, dtype=np.float32)]
            for _ in range(self._num_blocks)
        ]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Generated text (excluding prompt).
        """
        self.reset()

        # Tokenize
        token_list = self._tokenizer.encode(prompt)
        input_ids = np.array([token_list], dtype=np.int64)  # [1, seq_len]

        # Prefill — chunked (S=chunk_size) + remainder token-by-token (S=1)
        t0 = time.time()
        prompt_len = input_ids.shape[1]
        chunk_size = self._prefill_chunk_size
        pos = 0
        # Full chunks (S=chunk_size)
        while pos + chunk_size <= prompt_len:
            logits = self.forward(input_ids[:, pos:pos + chunk_size])
            pos += chunk_size
        # Remainder: token-by-token (S=1) — cannot pad because GDN Loop would process padding
        while pos < prompt_len:
            logits = self.forward(input_ids[:, pos:pos + 1])
            pos += 1
        prefill_time = time.time() - t0
        logger.info(
            "Prefill: %d tokens in %.1fms (%.1f tok/s, chunk=%d)",
            prompt_len, prefill_time * 1000, prompt_len / prefill_time, chunk_size,
        )

        # Greedy decode
        next_id = int(np.argmax(logits[0, -1, :]))
        generated = [next_id]

        eos_id = self._tokenizer.eos_token_id
        # Also stop on <|im_end|> (151645) and <|endoftext|> (151643)
        stop_ids = {eos_id, 151645, 151643} if eos_id else {151645, 151643}

        t_decode_start = time.time()
        for step in range(max_new_tokens - 1):
            if next_id in stop_ids:
                break

            token_input = np.array([[next_id]], dtype=np.int64)
            logits = self.forward(token_input)
            next_id = int(np.argmax(logits[0, -1, :]))
            generated.append(next_id)

        decode_time = time.time() - t_decode_start
        num_decoded = len(generated)

        # Print profiling summary before decode stats
        if self._profiling and self._profile_stats['total']:
            self._print_profile_summary()
            self._reset_profile_stats()

        if decode_time > 0:
            logger.info(
                "Decode: %d tokens in %.1fms (%.1f tok/s)",
                num_decoded, decode_time * 1000, num_decoded / decode_time,
            )

        # Remove stop token from output
        if generated and generated[-1] in stop_ids:
            generated = generated[:-1]

        return self._tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Qwen3.5 hybrid GPU+NPU inference")
    parser.add_argument("--model-dir", default="models/qwen35/Qwen3.5-0.8B-hybrid")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--device", default="HYBRID",
        choices=["HYBRID", "GPU_ONLY", "CPU_ONLY"],
        help="HYBRID=GDN on GPU + Attn on NPU, GPU_ONLY=all GPU, CPU_ONLY=all CPU",
    )
    parser.add_argument("--attn-past-seq", type=int, default=256, help="Static KV cache size for NPU")
    parser.add_argument("--no-attn-stateful", action="store_true", help="Use explicit KV I/O instead of stateful")
    parser.add_argument("--prefill-chunk-size", type=int, default=16, help="Prefill chunk size (1=token-by-token)")
    args = parser.parse_args()

    device_map = {
        "HYBRID": ("GPU", "NPU", "GPU"),
        "GPU_ONLY": ("GPU", "GPU", "GPU"),
        "CPU_ONLY": ("CPU", "CPU", "CPU"),
    }
    gdn_dev, attn_dev, head_dev = device_map[args.device]

    model = Qwen35HybridModel(
        model_dir=args.model_dir,
        gdn_device=gdn_dev,
        attn_device=attn_dev,
        head_device=head_dev,
        attn_past_seq=args.attn_past_seq,
        attn_stateful=not args.no_attn_stateful,
        prefill_chunk_size=args.prefill_chunk_size,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Device: {args.device} (GDN={gdn_dev}, Attn={attn_dev}, Head={head_dev})")
    print("-" * 60)

    output = model.generate(args.prompt, max_new_tokens=args.max_tokens)
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
