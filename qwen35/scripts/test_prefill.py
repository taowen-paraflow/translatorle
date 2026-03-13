"""Compare prefill strategies: chunk-major (old) vs layer-major (new).

Tests correctness (output identity) and performance.

Usage:
  powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\Apps\translatorle; uv run python -m qwen35.scripts.test_prefill'
"""

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prefill_chunk_major(model, input_ids):
    """Old approach: forward() per chunk (chunk-major ordering)."""
    model.reset()
    prompt_len = input_ids.shape[1]
    chunk_size = model._prefill_chunk_size
    pos = 0
    while pos < prompt_len:
        remaining = prompt_len - pos
        cs = chunk_size
        while cs > remaining:
            cs //= 2
        if cs < 1:
            cs = 1
        logits = model.forward(input_ids[:, pos:pos + cs])
        pos += cs
    return logits


def prefill_layer_major(model, input_ids):
    """New approach: prefill() with layer-major ordering."""
    model.reset()
    return model.prefill(input_ids)


def main():
    from qwen35.inference_hybrid import Qwen35HybridModel

    model_dir = "models/qwen35/Qwen3.5-0.8B-hybrid"

    prompts = {
        "short": "Hello",
        "medium": "The capital of France is",
        "long_cn": "请用中文详细解释量子计算的基本原理，包括量子比特、量子纠缠和量子门的概念。",
        "reasoning": "If a train travels at 60 km/h for 2 hours and then at 80 km/h for 3 hours, what is the total distance traveled? Let me think step by step.",
    }

    # --- Test HYBRID mode ---
    logger.info("=" * 60)
    logger.info("Testing HYBRID mode (GDN=GPU, Attn=NPU)")
    logger.info("=" * 60)

    model = Qwen35HybridModel(
        model_dir=model_dir,
        gdn_device="GPU",
        attn_device="NPU",
        head_device="GPU",
        attn_stateful=False,
        prefill_chunk_size=16,
    )

    NUM_RUNS = 3
    for name, prompt in prompts.items():
        token_list = model._tokenizer.encode(prompt)
        input_ids = np.array([token_list], dtype=np.int64)
        prompt_len = input_ids.shape[1]

        logger.info("-" * 60)
        logger.info("Prompt: %s (%d tokens)", name, prompt_len)

        # Warmup + benchmark (3 runs, report best)
        times_old, times_new = [], []
        for run in range(NUM_RUNS):
            t0 = time.time()
            logits_old = prefill_chunk_major(model, input_ids)
            times_old.append(time.time() - t0)

            t0 = time.time()
            logits_new = prefill_layer_major(model, input_ids)
            times_new.append(time.time() - t0)

        t_old = min(times_old)
        t_new = min(times_new)
        next_token_old = int(np.argmax(logits_old[0, -1, :]))
        next_token_new = int(np.argmax(logits_new[0, -1, :]))

        # Compare
        tokens_match = next_token_old == next_token_new
        logit_diff = np.max(np.abs(
            logits_old[0, -1, :] - logits_new[0, -1, :]))

        status = "MATCH" if tokens_match else "MISMATCH"
        speedup = t_old / t_new if t_new > 0 else float('inf')

        logger.info(
            "  chunk-major: %.1fms (%.1f tok/s) → token %d  [runs: %s]",
            t_old * 1000, prompt_len / t_old, next_token_old,
            ", ".join(f"{t*1000:.0f}" for t in times_old),
        )
        logger.info(
            "  layer-major: %.1fms (%.1f tok/s) → token %d  [runs: %s]",
            t_new * 1000, prompt_len / t_new, next_token_new,
            ", ".join(f"{t*1000:.0f}" for t in times_new),
        )
        logger.info(
            "  %s | max logit diff: %.6f | speedup: %.2fx",
            status, logit_diff, speedup,
        )

    # --- Also test GPU_ONLY mode ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Testing GPU_ONLY mode (all GPU, stateful)")
    logger.info("=" * 60)

    model_gpu = Qwen35HybridModel(
        model_dir=model_dir,
        gdn_device="GPU",
        attn_device="GPU",
        head_device="GPU",
        attn_stateful=True,
        prefill_chunk_size=16,
    )

    for name, prompt in prompts.items():
        token_list = model_gpu._tokenizer.encode(prompt)
        input_ids = np.array([token_list], dtype=np.int64)
        prompt_len = input_ids.shape[1]

        logger.info("-" * 60)
        logger.info("Prompt: %s (%d tokens)", name, prompt_len)

        times_old, times_new = [], []
        for run in range(NUM_RUNS):
            t0 = time.time()
            logits_old = prefill_chunk_major(model_gpu, input_ids)
            times_old.append(time.time() - t0)

            t0 = time.time()
            logits_new = prefill_layer_major(model_gpu, input_ids)
            times_new.append(time.time() - t0)

        t_old = min(times_old)
        t_new = min(times_new)
        next_token_old = int(np.argmax(logits_old[0, -1, :]))
        next_token_new = int(np.argmax(logits_new[0, -1, :]))

        tokens_match = next_token_old == next_token_new
        logit_diff = np.max(np.abs(
            logits_old[0, -1, :] - logits_new[0, -1, :]))
        status = "MATCH" if tokens_match else "MISMATCH"
        speedup = t_old / t_new if t_new > 0 else float('inf')

        logger.info(
            "  chunk-major: %.1fms (%.1f tok/s) → token %d  [runs: %s]",
            t_old * 1000, prompt_len / t_old, next_token_old,
            ", ".join(f"{t*1000:.0f}" for t in times_old),
        )
        logger.info(
            "  layer-major: %.1fms (%.1f tok/s) → token %d  [runs: %s]",
            t_new * 1000, prompt_len / t_new, next_token_new,
            ", ".join(f"{t*1000:.0f}" for t in times_new),
        )
        logger.info(
            "  %s | max logit diff: %.6f | speedup: %.2fx",
            status, logit_diff, speedup,
        )


if __name__ == "__main__":
    main()
