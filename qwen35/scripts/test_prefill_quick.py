"""Quick prefill comparison: Loop-based vs chunkwise parallel GDN.

Usage:
  powershell.exe -Command 'cd C:\Apps\translatorle; uv run python -m qwen35.scripts.test_prefill_quick'
"""

import time
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

from qwen35.inference_hybrid import Qwen35HybridModel


def bench_prefill(model, prompts, label, warmup=2, runs=3):
    tokenizer = model._tokenizer

    # Warmup
    for _ in range(warmup):
        model.reset()
        ids = tokenizer.encode("warmup test", return_tensors="np")
        model.prefill(ids)

    results = {}
    for name, prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="np")
        prompt_len = ids.shape[1]

        times = []
        for _ in range(runs):
            model.reset()
            t0 = time.perf_counter()
            logits = model.prefill(ids)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        best = min(times)
        results[name] = (prompt_len, best, logits)

    return results


def main():
    prompts = [
        ("short", "The capital of France is"),
        ("medium", "Explain the theory of relativity in simple terms"),
        ("long_cn", "详细分析人工智能对全球劳动力市场的经济影响"),
        ("reasoning", "A farmer has 15 chickens and 12 cows. Each chicken eats 0.5kg of feed "
                      "per day and each cow eats 8kg of feed per day. How much feed for a week? "
                      "Show your step-by-step calculation."),
    ]

    import os
    loop_dir = "models/qwen35/Qwen3.5-0.8B-hybrid"
    chunk_dir = "models/qwen35/Qwen3.5-0.8B-hybrid-chunkwise"

    if not os.path.exists(os.path.join(chunk_dir, "gdn_prefill_block_0.xml")):
        print("ERROR: chunkwise model not found at", chunk_dir)
        return

    # Load models (GPU_ONLY = all devices on GPU)
    print("Loading Loop-based model...")
    model_loop = Qwen35HybridModel(loop_dir, gdn_device="GPU", attn_device="GPU", head_device="GPU")
    print("Loading Chunkwise model...")
    model_chunk = Qwen35HybridModel(chunk_dir, gdn_device="GPU", attn_device="GPU", head_device="GPU")

    # Benchmark
    loop_res = bench_prefill(model_loop, prompts, "Loop")
    chunk_res = bench_prefill(model_chunk, prompts, "Chunk")

    # Results table
    print("\n=== Prefill Benchmark: Loop vs Chunkwise (GPU_ONLY) ===\n")
    header = f"  {'prompt':12s}  {'tok':>4s}  {'Loop':>9s}  {'Chunk':>9s}  {'Speedup':>7s}  {'Match':>5s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, _ in prompts:
        tok = loop_res[name][0]
        loop_ms = loop_res[name][1]
        loop_logits = loop_res[name][2]
        chunk_ms = chunk_res[name][1]
        chunk_logits = chunk_res[name][2]
        speedup = loop_ms / chunk_ms

        # Compare output tokens
        tok_loop = np.argmax(loop_logits[0, 0])
        tok_chunk = np.argmax(chunk_logits[0, 0])
        match = "YES" if tok_loop == tok_chunk else f"NO({tok_loop}vs{tok_chunk})"

        print(f"  {name:12s}  {tok:4d}  {loop_ms:7.1f}ms  {chunk_ms:7.1f}ms  {speedup:5.2f}x  {match:>5s}")


if __name__ == "__main__":
    main()
