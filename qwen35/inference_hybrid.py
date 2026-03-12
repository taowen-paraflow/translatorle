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
# Hybrid state (explicit, not OV stateful)
# ---------------------------------------------------------------------------

class HybridState:
    """Manages attention KV cache state for hybrid inference.

    GDN conv/recurrent states are managed internally by OpenVINO stateful
    variables (ReadValue/Assign) — they persist in GPU memory between infer()
    calls, eliminating host<->GPU state transfer overhead.
    """

    def __init__(self, num_blocks: int, num_kv_heads: int, head_dim: int):
        self.num_blocks = num_blocks
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # Attention KV caches: 1 layer per block
        # Start with past_seq=1 (zeros) because OV can't handle 0-dim tensors
        self.key_caches: List[np.ndarray] = [
            np.zeros((1, num_kv_heads, 1, head_dim), dtype=np.float32)
            for _ in range(num_blocks)
        ]
        self.value_caches: List[np.ndarray] = [
            np.zeros((1, num_kv_heads, 1, head_dim), dtype=np.float32)
            for _ in range(num_blocks)
        ]

    def reset(self):
        """Reset attention KV caches to zeros."""
        for i in range(len(self.key_caches)):
            self.key_caches[i] = np.zeros((1, self._num_kv_heads, 1, self._head_dim), dtype=np.float32)
            self.value_caches[i] = np.zeros((1, self._num_kv_heads, 1, self._head_dim), dtype=np.float32)


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
    ):
        """Load and compile all 13 subgraphs.

        Args:
            model_dir: Directory containing the hybrid IR files.
            gdn_device: Device for GDN blocks (default: GPU).
            attn_device: Device for attention blocks (default: NPU).
            head_device: Device for the head block (default: GPU).
            attn_past_seq: Static past_seq length for NPU attention blocks.
                NPU requires static shapes; KV caches are padded to this size.
        """
        model_dir = Path(model_dir)
        core = ov.Core()

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
        # GPU/CPU: stateful KV cache (stays on device, no host transfer)
        # NPU: explicit I/O (static shapes + padding, KV transferred each step)
        self._attn_stateful = (attn_device != "NPU")
        mode_str = "stateful" if self._attn_stateful else "explicit I/O"
        logger.info("Compiling %d Attn blocks on %s (%s) ...", num_blocks, attn_device, mode_str)
        self._attn_models: List[ov.CompiledModel] = []
        self._attn_requests: List = []
        t0 = time.time()

        attn_state_map = {"in_key_cache": "out_key_cache", "in_value_cache": "out_value_cache"}

        for i in range(num_blocks):
            ir = core.read_model(str(model_dir / f"attn_block_{i}.xml"))
            if self._attn_stateful:
                apply_make_stateful_transformation(ir, attn_state_map)
            elif attn_device == "NPU":
                ir = self._reshape_attn_static(ir, attn_past_seq)
            if attn_device in ("GPU", "NPU"):
                ir = self._add_f32_output_conversion(ir)
            if i == 0:
                logger.info("  Attn block inputs: %s",
                            [(inp.get_any_name(), str(inp.partial_shape), str(inp.element_type)) for inp in ir.inputs])
                if self._attn_stateful:
                    logger.info("  Attn block sinks: %d", len(ir.get_sinks()))
            compiled = core.compile_model(ir, attn_device)
            self._attn_models.append(compiled)
            self._attn_requests.append(compiled.create_infer_request())
        logger.info("  Attn compilation: %.1fs", time.time() - t0)

        if self._attn_stateful:
            self._init_attn_states()

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

        # --- State (only KV caches; GDN states are OV-internal) ---
        self._state = HybridState(
            num_blocks=num_blocks,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
        )
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

        KV cache starts empty (past_seq=0). The IR's internal Concat grows it
        each step: Concat([B,H,0,D], [B,H,1,D]) -> [B,H,1,D] on first call.
        No dummy entry needed — all positions are real tokens.
        """
        kv_shape = (1, self._num_kv_heads, 0, self._head_dim)
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

    def _reshape_attn_static(self, ir, past_seq: int):
        """Reshape attention IR to fully static shapes for NPU compilation.

        Uses index-based access to avoid name collision issues
        (input 2/3 may share names with output 1/2).
        """
        B, S = 1, 1
        # Index-based: 0=hidden, 1=position_ids, 2=key_cache, 3=value_cache, 4=attention_mask
        shapes = {}
        for i, inp in enumerate(ir.inputs):
            name = inp.get_any_name()
            ps = inp.partial_shape
            if i == 0:  # hidden: [B, S, hidden_size]
                shapes[name] = ov.PartialShape([B, S, self._hidden_size])
            elif i == 1:  # position_ids: [3, B, S]
                shapes[name] = ov.PartialShape([3, B, S])
            elif i in (2, 3):  # key/value cache: [B, num_kv_heads, past_seq, head_dim]
                shapes[name] = ov.PartialShape([B, self._num_kv_heads, past_seq, self._head_dim])
            elif i == 4:  # attention_mask: [B, 1, S, past_seq + S]
                shapes[name] = ov.PartialShape([B, 1, S, past_seq + S])
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

    def _run_attn_block(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """Run one Attention block (1 layer)."""
        if self._attn_stateful:
            return self._run_attn_block_stateful(block_idx, hidden, position_ids)
        return self._run_attn_block_explicit(block_idx, hidden, position_ids)

    def _run_attn_block_stateful(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """Run attention with stateful KV cache (GPU/CPU).

        KV cache persists on device via ReadValue/Assign. Only hidden,
        position_ids, and attention_mask are transferred each call.
        """
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]

        # All-zeros mask = attend to everything (no padding, no dummy entry)
        key_seq = self._past_length + seq_len
        attention_mask = np.zeros((batch_size, 1, seq_len, key_seq), dtype=np.float32)

        # After stateful transform: inputs are [hidden, position_ids, attention_mask]
        req = self._attn_requests[block_idx]
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(position_ids)))
        req.set_input_tensor(2, ov.Tensor(np.ascontiguousarray(attention_mask)))
        req.infer()

        # Output may be 2D [batch*seq, hidden_size] — reshape to 3D
        return req.get_output_tensor(0).data.copy().reshape(batch_size, seq_len, -1)

    def _run_attn_block_explicit(self, block_idx: int, hidden: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """Run attention with explicit KV cache I/O (NPU).

        KV cache is padded to static size, transferred each step, then compacted.
        """
        state = self._state
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]
        key_cache = state.key_caches[block_idx]
        value_cache = state.value_caches[block_idx]
        actual_cache_len = key_cache.shape[2]  # includes initial dummy at pos 0

        # NPU needs static past_seq — pad KV cache
        key_cache = self._pad_kv_cache(key_cache, self._attn_past_seq, is_key=True)
        value_cache = self._pad_kv_cache(value_cache, self._attn_past_seq, is_key=False)

        # Build 4D attention mask: 0.0 = attend, -65504.0 = ignore
        attention_mask = self._build_attn_mask(
            batch_size, seq_len, actual_cache_len, key_cache.shape[2],
        )

        inputs = [hidden, position_ids, key_cache, value_cache, attention_mask]
        req = self._attn_requests[block_idx]
        for i, arr in enumerate(inputs):
            req.set_input_tensor(i, ov.Tensor(np.ascontiguousarray(arr)))
        req.infer()

        hidden = req.get_output_tensor(0).data.copy().reshape(batch_size, seq_len, -1)
        new_key = req.get_output_tensor(1).data.copy()
        new_value = req.get_output_tensor(2).data.copy()

        # Compact: remove padding zeros, keep real entries + new token
        actual_past = state.key_caches[block_idx].shape[2]
        P = self._attn_past_seq
        new_key = np.concatenate([
            new_key[:, :, :actual_past, :],
            new_key[:, :, P:P+seq_len, :],
        ], axis=2)
        new_value = np.concatenate([
            new_value[:, :, :actual_past, :],
            new_value[:, :, P:P+seq_len, :],
        ], axis=2)

        state.key_caches[block_idx] = new_key
        state.value_caches[block_idx] = new_value

        return hidden

    def _build_attn_mask(
        self,
        batch_size: int,
        seq_len: int,
        actual_cache_len: int,
        padded_cache_len: int,
    ) -> np.ndarray:
        """Build 4D attention mask for one attention block.

        After the attention layer concats input KV cache with the new token's KV,
        the full key_seq = padded_cache_len + seq_len.

        Layout of key positions:
          [0]                          -> initial dummy (always masked)
          [1 .. actual_cache_len-1]    -> real past tokens (attend)
          [actual_cache_len .. padded_cache_len-1] -> padding (masked, NPU only)
          [padded_cache_len .. padded_cache_len+seq_len-1] -> new token(s) (attend)

        Returns:
            attention_mask: shape [B, 1, seq_len, key_seq], dtype float32.
            0.0 = attend, -65504.0 = ignore.
        """
        MASK_VALUE = np.float32(-65504.0)  # min fp16, guarantees ~0 softmax weight
        key_seq = padded_cache_len + seq_len

        mask = np.full((batch_size, 1, seq_len, key_seq), MASK_VALUE, dtype=np.float32)
        # Unmask real past tokens (positions 1..actual_cache_len-1)
        if actual_cache_len > 1:
            mask[:, :, :, 1:actual_cache_len] = 0.0
        # Unmask new token positions (positions padded_cache_len..padded_cache_len+seq_len-1)
        mask[:, :, :, padded_cache_len:padded_cache_len + seq_len] = 0.0
        return mask

    def _pad_kv_cache(self, cache: np.ndarray, target_len: int, is_key: bool = False) -> np.ndarray:
        """Pad KV cache to target_len along dim 2.

        For KEY cache: pad with large negative values so Q@K^T produces
        very negative scores -> softmax gives ~0 weight (effectively masked).
        For VALUE cache: pad with zeros (weight is ~0 anyway).
        """
        current_len = cache.shape[2]
        if current_len >= target_len:
            return cache[:, :, :target_len, :]
        pad_len = target_len - current_len
        if is_key:
            # Large negative K → very negative attention scores → masked by softmax
            pad = np.full(
                (cache.shape[0], cache.shape[1], pad_len, cache.shape[3]),
                -1e4, dtype=cache.dtype,
            )
        else:
            pad = np.zeros(
                (cache.shape[0], cache.shape[1], pad_len, cache.shape[3]),
                dtype=cache.dtype,
            )
        return np.concatenate([cache, pad], axis=2)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Run one forward pass through all 13 subgraphs.

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

        for i in range(self._num_blocks):
            if profiling:
                t0 = time.perf_counter()
            hidden = self._run_gdn_block(i, hidden, attention_mask)
            if profiling:
                self._profile_stats['gdn_blocks'][i].append(time.perf_counter() - t0)

            if profiling:
                t0 = time.perf_counter()
            hidden = self._run_attn_block(i, hidden, position_ids)
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
        if not self._attn_stateful:
            self._state.reset()       # Reset attention KV caches (NPU explicit mode)
        else:
            self._init_attn_states()  # Reset attention stateful variables
        self._init_gdn_states()       # Reset GDN stateful variables (can't use s.reset())
        self._past_length = 0

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

        # Prefill — token-by-token (NPU attention blocks require S=1)
        t0 = time.time()
        for i in range(input_ids.shape[1]):
            token_input = input_ids[:, i:i+1]  # [1, 1]
            logits = self.forward(token_input)
        prefill_time = time.time() - t0
        prompt_len = input_ids.shape[1]
        logger.info(
            "Prefill: %d tokens in %.1fms (%.1f tok/s)",
            prompt_len, prefill_time * 1000, prompt_len / prefill_time,
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
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Device: {args.device} (GDN={gdn_dev}, Attn={attn_dev}, Head={head_dev})")
    print("-" * 60)

    output = model.generate(args.prompt, max_new_tokens=args.max_tokens)
    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
