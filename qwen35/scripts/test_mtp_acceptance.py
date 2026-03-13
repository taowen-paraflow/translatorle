#!/usr/bin/env python3
"""Test MTP (Multi-Token Prediction) acceptance rate for Qwen3.5 speculative decoding.

Measures how often the MTP block correctly predicts the main model's next token.
This determines whether MTP speculative decoding is worth implementing in C++.

Algorithm:
  For each decode step after prefill:
    1. Run main model forward -> hidden_states + logits -> main_token
    2. Run MTP block with (hidden_states, embed(main_token)) -> mtp_hidden
    3. Compute mtp_logits = mtp_hidden @ embed_tokens.T (shared lm_head)
    4. draft_token = argmax(mtp_logits)
    5. Next main model step produces actual_token
    6. Compare: draft_token == actual_token?

The MTP block's output already has mtp.norm applied, so we skip the head
block's norm and directly multiply by the embedding matrix (tied weights).

Run (root venv):
  powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\\Apps\\translatorle; C:\\Users\\taowen\\.local\\bin\\uv.exe run python -m qwen35.scripts.test_mtp_acceptance'
  powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\\Apps\\translatorle; C:\\Users\\taowen\\.local\\bin\\uv.exe run python -m qwen35.scripts.test_mtp_acceptance --prompt "Hello"'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.inference_hybrid import Qwen35HybridModel

logger = logging.getLogger(__name__)

# Reuse prompts from test_quality.py
TEST_PROMPTS = [
    ("factual_en", "The capital of France is"),
    ("factual_cn", "中国的首都是"),
    ("reasoning", "If I have 3 apples and buy 5 more, then give away 2, how many do I have?"),
    ("code", "Write a Python function that checks if a number is prime:"),
    (
        "long_context",
        "Explain the difference between machine learning and deep learning in detail."
        " Cover the key concepts, advantages, and use cases of each approach.",
    ),
    ("translation", "Translate to English: 今天天气很好，我们去公园散步吧。"),
]


# ---------------------------------------------------------------------------
# MTP block runner
# ---------------------------------------------------------------------------

class MTPRunner:
    """Loads and runs the MTP block to produce draft token predictions.

    The MTP block takes main model hidden_states + embedding of predicted token,
    and outputs a hidden state that can be projected to vocab space via the
    shared embedding matrix (tied lm_head weights).

    The MTP attention layer has its own KV cache that persists across decode
    steps, accumulating context from all previous MTP calls within a prompt.
    Call reset() before each new prompt to clear the KV cache.
    """

    def __init__(self, model_dir: str | Path, device: str = "GPU",
                 max_cache_len: int = 256):
        model_dir = Path(model_dir)
        core = ov.Core()

        # Load config for dimensions
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", cfg)

        self._hidden_size = text_cfg["hidden_size"]
        self._num_kv_heads = text_cfg["num_key_value_heads"]
        self._head_dim = text_cfg.get(
            "head_dim",
            self._hidden_size // text_cfg["num_attention_heads"],
        )
        self._max_cache_len = max_cache_len

        # Load and compile MTP block
        mtp_path = model_dir / "mtp_block.xml"
        if not mtp_path.exists():
            raise FileNotFoundError(f"MTP block not found: {mtp_path}")

        mtp_ir = core.read_model(str(mtp_path))
        if device in ("GPU", "NPU"):
            mtp_ir = _add_f32_output(mtp_ir)
        self._compiled = core.compile_model(mtp_ir, device)
        self._request = self._compiled.create_infer_request()

        logger.info("MTP block compiled on %s", device)
        for inp in mtp_ir.inputs:
            logger.info("  Input : %-25s %s %s",
                        inp.get_any_name(), inp.partial_shape, inp.element_type)
        for out in mtp_ir.outputs:
            logger.info("  Output: %-25s %s %s",
                        out.get_any_name(), out.partial_shape, out.element_type)

        # Persistent KV cache for the MTP attention layer.
        # Accumulates across decode steps within a prompt; call reset() to clear.
        kv_shape = (1, self._num_kv_heads, self._max_cache_len, self._head_dim)
        self._kv_key = np.zeros(kv_shape, dtype=np.float32)
        self._kv_value = np.zeros(kv_shape, dtype=np.float32)
        self._mtp_past_length = 0

    def reset(self):
        """Reset MTP KV cache. Call at the start of each new prompt."""
        self._kv_key[:] = 0.0
        self._kv_value[:] = 0.0
        self._mtp_past_length = 0

    def predict(self, hidden_states: np.ndarray, token_id: int,
                embed_table: np.ndarray, seq_position: int) -> tuple[int, np.ndarray]:
        """Run MTP block and project to vocab to get a draft token.

        Uses persistent KV cache that accumulates across decode steps.
        The cache_position tracks where to write in the fixed-size KV cache,
        and the attention mask allows attending to all previously written positions.

        Args:
            hidden_states: [1, 1, hidden_size] from main model (pre-head-norm).
            token_id: The token predicted by the main model at this step.
            embed_table: [vocab_size, hidden_size] float32 embedding matrix.
            seq_position: Absolute sequence position (for RoPE).

        Returns:
            (draft_token_id, mtp_logits[1, vocab_size])
        """
        # Embed the predicted token -> [1, 1, hidden_size]
        input_embeds = embed_table[np.array([[token_id]], dtype=np.int64)]

        # Position IDs for mRoPE: [3, B, S]
        position_ids = np.full((3, 1, 1), seq_position, dtype=np.int64)

        # Cache position: write at current mtp_past_length
        cache_position = np.array([self._mtp_past_length], dtype=np.int64)

        # Attention mask: attend to positions 0..mtp_past_length (inclusive),
        # mask everything else with -65504.0 (fp16 min).
        mask = np.full(
            (1, 1, 1, self._max_cache_len), np.float32(-65504.0),
        )
        mask[0, 0, 0, : self._mtp_past_length + 1] = 0.0

        # Run MTP block with persistent KV cache
        req = self._request
        req.set_input_tensor(0, ov.Tensor(np.ascontiguousarray(hidden_states)))
        req.set_input_tensor(1, ov.Tensor(np.ascontiguousarray(input_embeds)))
        req.set_input_tensor(2, ov.Tensor(np.ascontiguousarray(position_ids)))
        req.set_input_tensor(3, ov.Tensor(np.ascontiguousarray(self._kv_key)))
        req.set_input_tensor(4, ov.Tensor(np.ascontiguousarray(self._kv_value)))
        req.set_input_tensor(5, ov.Tensor(np.ascontiguousarray(cache_position)))
        req.set_input_tensor(6, ov.Tensor(np.ascontiguousarray(mask)))
        req.infer()

        # MTP output hidden is 2D [B, hidden_size] due to tracing squeeze
        mtp_hidden = req.get_output_tensor(0).data.copy()  # [1, 1024]

        # Store updated KV caches for next call
        self._kv_key = req.get_output_tensor(1).data.copy()
        self._kv_value = req.get_output_tensor(2).data.copy()
        self._mtp_past_length += 1

        # Project to vocab using shared embedding weights (tied lm_head).
        # MTP already applied its own final_norm (mtp.norm), so we skip
        # the head block's norm and directly compute logits = hidden @ embed.T
        mtp_logits = mtp_hidden @ embed_table.T  # [1, vocab_size]

        draft_token = int(np.argmax(mtp_logits[0]))
        return draft_token, mtp_logits


# ---------------------------------------------------------------------------
# Modified forward that captures hidden states
# ---------------------------------------------------------------------------

def forward_with_hidden(model: Qwen35HybridModel,
                        token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run main model forward, returning (hidden_states, logits).

    Replicates model.forward() but also returns the hidden states before
    the head block (needed as MTP input).

    Args:
        model: The hybrid model.
        token_ids: [1, seq_len] int64 token IDs.

    Returns:
        hidden: [1, seq_len, hidden_size] float32 (pre-head-norm).
        logits: [1, seq_len, vocab_size] float32.
    """
    batch_size, seq_len = token_ids.shape

    # Embed tokens
    embeds = model._embed_table[token_ids]

    # Attention mask for GDN blocks
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

    # Position IDs for attention blocks (mRoPE: [3, B, seq_len])
    positions = np.arange(
        model._past_length, model._past_length + seq_len, dtype=np.int64,
    )
    position_ids = np.tile(
        positions[np.newaxis, np.newaxis, :], (3, batch_size, 1),
    )

    hidden = embeds
    use_prefill = seq_len in model._attn_prefill_requests

    for i in range(model._num_blocks):
        hidden = model._run_gdn_block(i, hidden, attention_mask)
        hidden = model._run_attn_block(
            i, hidden, position_ids, use_prefill=use_prefill,
        )

    # Capture hidden BEFORE head
    hidden_out = hidden.copy()

    # Run head
    model._head_request.set_input_tensor(
        0, ov.Tensor(np.ascontiguousarray(hidden)),
    )
    model._head_request.infer()
    logits = model._head_request.get_output_tensor(0).data.copy()

    model._past_length += seq_len
    return hidden_out, logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_f32_output(ir):
    """Add FP16->FP32 output conversion."""
    from openvino.preprocess import PrePostProcessor
    ppp = PrePostProcessor(ir)
    for i in range(len(ir.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    return ppp.build()


# ---------------------------------------------------------------------------
# Per-prompt test
# ---------------------------------------------------------------------------

def test_prompt(
    model: Qwen35HybridModel,
    mtp: MTPRunner,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Run generation with MTP acceptance tracking.

    Returns dict with keys: output, accepted, total, rate, top5_accepted,
    top5_rate, tokens_generated.
    """
    model.reset()
    mtp.reset()

    # Tokenize
    token_list = model._tokenizer.encode(prompt)
    input_ids = np.array([token_list], dtype=np.int64)

    # Prefill (we do not have hidden from prefill, so no MTP for first token)
    logits = model.prefill(input_ids)
    first_token = int(np.argmax(logits[0, -1, :]))

    eos_id = model._tokenizer.eos_token_id
    stop_ids = {eos_id, 151645, 151643} if eos_id else {151645, 151643}

    generated = [first_token]
    accepted = 0
    top5_accepted = 0
    total_drafts = 0
    pending_draft = None

    next_id = first_token
    for step in range(max_tokens - 1):
        if next_id in stop_ids:
            break

        # Decode one token, capturing hidden states
        token_input = np.array([[next_id]], dtype=np.int64)
        hidden, logits = forward_with_hidden(model, token_input)
        next_id = int(np.argmax(logits[0, -1, :]))
        generated.append(next_id)

        # Check the pending draft from the previous step
        if pending_draft is not None:
            total_drafts += 1
            draft_id, draft_logits = pending_draft
            if draft_id == next_id:
                accepted += 1
            # Top-5 check: is the actual token in the draft's top 5?
            top5_ids = set(np.argsort(draft_logits[0])[-5:].tolist())
            if next_id in top5_ids:
                top5_accepted += 1

        # Make MTP prediction for the next step.
        # hidden is [1, 1, H], next_id is the just-predicted token.
        # The MTP predicts the token at position P+1, so position_ids = _past_length
        # (forward_with_hidden already incremented _past_length by 1).
        seq_pos = model._past_length
        draft_token, draft_logits = mtp.predict(
            hidden, next_id, model._embed_table, seq_pos,
        )
        pending_draft = (draft_token, draft_logits)

    # Output text
    output_tokens = list(generated)
    if output_tokens and output_tokens[-1] in stop_ids:
        output_tokens = output_tokens[:-1]
    output_text = model._tokenizer.decode(output_tokens, skip_special_tokens=True)

    rate = accepted / total_drafts if total_drafts > 0 else 0.0
    top5_rate = top5_accepted / total_drafts if total_drafts > 0 else 0.0

    return {
        "output": output_text,
        "accepted": accepted,
        "top5_accepted": top5_accepted,
        "total": total_drafts,
        "rate": rate,
        "top5_rate": top5_rate,
        "tokens_generated": len(generated),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Test MTP acceptance rate for Qwen3.5 speculative decoding",
    )
    parser.add_argument(
        "--model-dir",
        default="models/qwen35/Qwen3.5-0.8B-hybrid",
        help="Model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="GPU_ONLY",
        choices=["HYBRID", "GPU_ONLY", "CPU_ONLY"],
        help="Main model device config (default: %(default)s)",
    )
    parser.add_argument(
        "--mtp-device",
        default="GPU",
        help="Device for MTP block (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max new tokens per prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt (overrides built-in test suite)",
    )
    parser.add_argument(
        "--attn-past-seq",
        type=int,
        default=256,
        help="Static KV cache size (default: %(default)s)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=16,
        help="Prefill chunk size (default: %(default)s)",
    )
    args = parser.parse_args()

    # Device map
    device_map = {
        "HYBRID": ("GPU", "NPU", "GPU"),
        "GPU_ONLY": ("GPU", "GPU", "GPU"),
        "CPU_ONLY": ("CPU", "CPU", "CPU"),
    }
    gdn_dev, attn_dev, head_dev = device_map[args.device]
    attn_stateful = args.device != "HYBRID"

    # Load main model
    logger.info("Loading main model from %s (device=%s)", args.model_dir, args.device)
    model = Qwen35HybridModel(
        model_dir=args.model_dir,
        gdn_device=gdn_dev,
        attn_device=attn_dev,
        head_device=head_dev,
        attn_past_seq=args.attn_past_seq,
        attn_stateful=attn_stateful,
        prefill_chunk_size=args.prefill_chunk_size,
    )

    # Load MTP block
    logger.info("Loading MTP block on %s", args.mtp_device)
    mtp = MTPRunner(args.model_dir, device=args.mtp_device)

    # Select prompts
    if args.prompt:
        prompts = [("custom", args.prompt)]
    else:
        prompts = TEST_PROMPTS

    # Run tests
    print("=" * 70)
    print(f"MTP Acceptance Rate Test")
    print(f"  Prompts   : {len(prompts)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Model     : {args.model_dir}")
    print(f"  Device    : {args.device} (MTP on {args.mtp_device})")
    print("=" * 70)

    total_accepted = 0
    total_top5 = 0
    total_drafts = 0
    all_results = []

    for idx, (name, prompt) in enumerate(prompts):
        print(f"\n[{idx + 1}/{len(prompts)}] {name}")
        print("-" * 50)

        t0 = time.time()
        result = test_prompt(model, mtp, prompt, args.max_tokens)
        elapsed = time.time() - t0

        total_accepted += result["accepted"]
        total_top5 += result["top5_accepted"]
        total_drafts += result["total"]
        all_results.append((name, result))

        output_preview = result["output"]
        if len(output_preview) > 120:
            output_preview = output_preview[:120] + "..."

        print(f"  Prompt : {prompt}")
        print(f"  Output : {output_preview}")
        print(f"  Tokens : {result['tokens_generated']} generated, "
              f"{result['total']} MTP drafts")
        print(f"  Top-1  : {result['accepted']}/{result['total']} "
              f"({result['rate'] * 100:.1f}%)")
        print(f"  Top-5  : {result['top5_accepted']}/{result['total']} "
              f"({result['top5_rate'] * 100:.1f}%)")
        print(f"  Time   : {elapsed:.1f}s")

    # Summary table
    overall_rate = total_accepted / total_drafts if total_drafts > 0 else 0.0
    overall_top5 = total_top5 / total_drafts if total_drafts > 0 else 0.0

    print()
    print("=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"{'Prompt':<16} | {'Tokens':>6} | {'Top-1':>12} | {'Top-5':>12}")
    print("-" * 70)
    for name, result in all_results:
        print(f"{name:<16} | {result['total']:>6} | "
              f"{result['rate'] * 100:>5.1f}% ({result['accepted']:>3}) | "
              f"{result['top5_rate'] * 100:>5.1f}% ({result['top5_accepted']:>3})")
    print("-" * 70)
    print(f"{'OVERALL':<16} | {total_drafts:>6} | "
          f"{overall_rate * 100:>5.1f}% ({total_accepted:>3}) | "
          f"{overall_top5 * 100:>5.1f}% ({total_top5:>3})")
    print("=" * 70)

    # Verdict
    print()
    if overall_rate >= 0.6:
        print(f"VERDICT: MTP speculative decoding is WORTH implementing")
        print(f"  {overall_rate * 100:.0f}% top-1 acceptance >= 60% threshold")
        print(f"  Expected speedup with mtp_steps=1: ~{1 / (1 - overall_rate * 0.3):.1f}x")
    elif overall_rate >= 0.4:
        print(f"VERDICT: MTP speculative decoding is MARGINAL")
        print(f"  {overall_rate * 100:.0f}% top-1 acceptance, between 40-60%")
        print(f"  May break even depending on MTP/verify overhead")
    else:
        print(f"VERDICT: MTP speculative decoding is NOT worth it")
        print(f"  {overall_rate * 100:.0f}% top-1 acceptance < 40% threshold")
        print(f"  Verify overhead would outweigh the few accepted drafts")


if __name__ == "__main__":
    main()
