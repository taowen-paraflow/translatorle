"""Benchmark GPU attention prefill vs NPU chunked prefill.

Compares:
1. HYBRID + GPU attn prefill (default): GDN→GPU, Attn prefill→GPU, Attn decode→NPU
2. HYBRID + NPU attn prefill: GDN→GPU, Attn prefill→NPU chunked, Attn decode→NPU
3. GPU_ONLY (baseline): all on GPU, stateful

Tests correctness (first token match) and speed across multiple prompts.
"""

import logging
import time
import sys
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from qwen35.inference_hybrid import Qwen35HybridModel

MODEL_DIR = "models/qwen35/Qwen3.5-0.8B-hybrid"

PROMPTS = [
    ("very_short", "Hi"),
    ("short_en", "The capital of France is"),
    ("math", "What is 1+1? Answer:"),
    ("medium_en", "Explain the theory of relativity in simple terms."),
    ("chinese", "请用中文解释什么是机器学习"),
    ("long_code", "Write a Python function that takes a list of integers and returns the two numbers that sum to a target value. Use a hash map for O(n) time complexity."),
]

WARMUP = 2
RUNS = 3

configs = [
    {
        "name": "GPU_ONLY (stateful)",
        "kwargs": {
            "gdn_device": "GPU", "attn_device": "GPU", "head_device": "GPU",
            "attn_stateful": True, "prefill_chunk_size": 16,
        },
    },
    {
        "name": "HYBRID + GPU attn prefill",
        "kwargs": {
            "gdn_device": "GPU", "attn_device": "NPU", "head_device": "GPU",
            "attn_stateful": False, "prefill_chunk_size": 16,
            "attn_gpu_prefill": True,
        },
    },
    {
        "name": "HYBRID + NPU attn prefill",
        "kwargs": {
            "gdn_device": "GPU", "attn_device": "NPU", "head_device": "GPU",
            "attn_stateful": False, "prefill_chunk_size": 16,
            "attn_gpu_prefill": False,
        },
    },
]


def benchmark_config(config):
    """Load model, run warmup + timed runs, return results."""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    model = Qwen35HybridModel(model_dir=MODEL_DIR, **config["kwargs"])

    results = {}
    for prompt_name, prompt_text in PROMPTS:
        token_list = model._tokenizer.encode(prompt_text)
        input_ids = np.array([token_list], dtype=np.int64)
        prompt_len = input_ids.shape[1]

        # Warmup
        for _ in range(WARMUP):
            model.reset()
            logits = model.prefill(input_ids)

        # Timed runs
        times = []
        first_token = None
        for _ in range(RUNS):
            model.reset()
            t0 = time.perf_counter()
            logits = model.prefill(input_ids)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if first_token is None:
                first_token = int(np.argmax(logits[0, -1, :]))

        best_ms = min(times) * 1000
        tok_s = prompt_len / min(times)
        results[prompt_name] = {
            "prompt_len": prompt_len,
            "best_ms": best_ms,
            "tok_s": tok_s,
            "first_token": first_token,
            "first_token_text": model._tokenizer.decode([first_token]),
        }
        print(f"  {prompt_name:15s} ({prompt_len:2d} tok): {best_ms:7.1f}ms = {tok_s:5.1f} tok/s  -> '{results[prompt_name]['first_token_text']}'")

    del model
    return results


def main():
    all_results = {}
    for config in configs:
        all_results[config["name"]] = benchmark_config(config)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Prefill Performance Comparison")
    print(f"{'='*80}")

    baseline_name = configs[0]["name"]
    baseline = all_results[baseline_name]

    header = f"{'Prompt':15s} {'Tok':>3s}"
    for config in configs:
        header += f" | {config['name']:>25s}"
    header += " | Match?"
    print(header)
    print("-" * len(header))

    for prompt_name, _ in PROMPTS:
        row = f"{prompt_name:15s} {baseline[prompt_name]['prompt_len']:3d}"
        first_tokens = []
        for config in configs:
            r = all_results[config["name"]][prompt_name]
            speedup = r["tok_s"] / baseline[prompt_name]["tok_s"]
            row += f" | {r['best_ms']:6.0f}ms {speedup:4.2f}x"
            first_tokens.append(r["first_token"])
        match = "OK" if len(set(first_tokens)) == 1 else "DIFF"
        row += f" | {match}"
        print(row)

    # Averages
    print("-" * len(header))
    row = f"{'AVERAGE':15s} {'':3s}"
    for config in configs:
        avg_ms = np.mean([all_results[config["name"]][pn]["best_ms"] for pn, _ in PROMPTS])
        avg_baseline = np.mean([baseline[pn]["best_ms"] for pn, _ in PROMPTS])
        speedup = avg_baseline / avg_ms
        row += f" | {avg_ms:6.0f}ms {speedup:4.2f}x"
    row += " |"
    print(row)


if __name__ == "__main__":
    main()
