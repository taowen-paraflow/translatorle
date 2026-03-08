"""Benchmark comparison: INT4_ASYM (original) vs INT4_SYM (requantized) on NPU.

Tests both models with openvino_genai.LLMPipeline on NPU to measure the
performance impact of switching from INT4_ASYM to INT4_SYM quantization.

Each model gets:
  - 1 warmup run (discarded)
  - 3 timed runs with streaming callback to measure TTFT and decode tok/s

Usage:
    uv run python hymt/scripts/bench_mt_compare.py
"""

from __future__ import annotations

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Model configs: (name, model_dir, cache_dir)
MODEL_CONFIGS = [
    ("INT4_ASYM (original)", os.path.join(MODELS_DIR, "hy_mt_ov"), os.path.join(MODELS_DIR, "hy_mt_cache_asym")),
    ("INT4_SYM  (new)",      os.path.join(MODELS_DIR, "hy_mt_int4sym"), os.path.join(MODELS_DIR, "hy_mt_cache_sym")),
]

NPU_CONFIG = {
    "MAX_PROMPT_LEN": 512,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": 512,
    "GENERATE_HINT": "BEST_PERF",
}

PROMPT = "Translate to Chinese: The quick brown fox jumps over the lazy dog."
MAX_NEW_TOKENS = 128
NUM_WARMUP = 1
NUM_RUNS = 3


def run_single(pipe, prompt: str, max_new_tokens: int) -> dict:
    """Run a single generation and return timing metrics."""
    token_times: list[float] = []
    first_token_time: float | None = None

    def streamer(subword: str) -> bool:
        nonlocal first_token_time
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        token_times.append(now)
        return False  # continue generation

    t_start = time.perf_counter()
    result = pipe.generate(prompt, max_new_tokens=max_new_tokens, streamer=streamer)
    t_end = time.perf_counter()

    n_tokens = len(token_times)
    ttft = (first_token_time - t_start) if first_token_time is not None else 0.0
    total = t_end - t_start
    decode_time = total - ttft

    if n_tokens > 1 and decode_time > 0:
        tok_per_sec = (n_tokens - 1) / decode_time
    else:
        tok_per_sec = 0.0

    return {
        "result": result.strip() if isinstance(result, str) else str(result),
        "n_tokens": n_tokens,
        "ttft": ttft,
        "decode_time": decode_time,
        "total": total,
        "tok_per_sec": tok_per_sec,
    }


def benchmark_model(name: str, model_dir: str, cache_dir: str) -> dict:
    """Benchmark a single model, return average metrics."""
    import openvino_genai as ov_genai

    print(f"\n{'=' * 60}")
    print(f"  Benchmarking: {name}")
    print(f"  Model dir:    {model_dir}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"{'=' * 60}")

    # Check model dir exists
    if not os.path.exists(os.path.join(model_dir, "openvino_model.xml")):
        print(f"  ERROR: Model not found at {model_dir}")
        return {}

    # Check required files
    required = ["openvino_tokenizer.xml", "openvino_detokenizer.xml", "generation_config.json"]
    for f in required:
        path = os.path.join(model_dir, f)
        if not os.path.exists(path):
            print(f"  ERROR: Required file missing: {f}")
            return {}

    # Report model .bin size
    bin_path = os.path.join(model_dir, "openvino_model.bin")
    bin_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"  Model size:   {bin_size_mb:.1f} MB")
    print()

    # Initialize pipeline
    print("  Initializing LLMPipeline on NPU...")
    os.makedirs(cache_dir, exist_ok=True)
    config = {**NPU_CONFIG, "CACHE_DIR": cache_dir}

    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(model_dir, "NPU", **config)
    init_time = time.perf_counter() - t0
    print(f"  Init time: {init_time:.1f}s")
    print()

    # Warmup
    print(f"  Warmup ({NUM_WARMUP} run)...")
    for i in range(NUM_WARMUP):
        r = run_single(pipe, PROMPT, MAX_NEW_TOKENS)
        print(f"    Warmup {i+1}: {r['n_tokens']} tokens, {r['tok_per_sec']:.1f} tok/s")
        print(f"    Output: {r['result'][:80]}...")
    print()

    # Timed runs
    print(f"  Timed runs ({NUM_RUNS} runs)...")
    runs = []
    for i in range(NUM_RUNS):
        r = run_single(pipe, PROMPT, MAX_NEW_TOKENS)
        runs.append(r)
        print(f"    Run {i+1}: TTFT={r['ttft']:.3f}s, decode={r['tok_per_sec']:.1f} tok/s, "
              f"tokens={r['n_tokens']}, total={r['total']:.2f}s")

    # Compute averages
    avg_ttft = sum(r["ttft"] for r in runs) / len(runs)
    avg_tok_per_sec = sum(r["tok_per_sec"] for r in runs) / len(runs)
    avg_total = sum(r["total"] for r in runs) / len(runs)
    avg_tokens = sum(r["n_tokens"] for r in runs) / len(runs)

    summary = {
        "name": name,
        "bin_size_mb": bin_size_mb,
        "init_time": init_time,
        "avg_ttft": avg_ttft,
        "avg_tok_per_sec": avg_tok_per_sec,
        "avg_total": avg_total,
        "avg_tokens": avg_tokens,
        "output": runs[0]["result"],
    }

    print(f"\n  Average: TTFT={avg_ttft:.3f}s, decode={avg_tok_per_sec:.1f} tok/s, total={avg_total:.2f}s")

    # Cleanup
    del pipe
    return summary


def main():
    import openvino_genai as ov_genai

    print("=" * 60)
    print("  HY-MT NPU Benchmark: INT4_ASYM vs INT4_SYM")
    print("=" * 60)
    print(f"  openvino_genai version: {ov_genai.__version__}")
    print(f"  Prompt: {PROMPT}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Warmup: {NUM_WARMUP}, Timed runs: {NUM_RUNS}")

    results = []
    for name, model_dir, cache_dir in MODEL_CONFIGS:
        r = benchmark_model(name, model_dir, cache_dir)
        if r:
            results.append(r)

    if len(results) < 2:
        print("\nERROR: Could not benchmark both models. Check errors above.")
        return

    # Comparison table
    print("\n")
    print("=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Metric':<25s} | {'INT4_ASYM (orig)':>18s} | {'INT4_SYM (new)':>18s} | {'Speedup':>8s}")
    print(f"  {'-'*25}-+-{'-'*18}-+-{'-'*18}-+-{'-'*8}")

    r0, r1 = results[0], results[1]

    # Model size
    print(f"  {'Model .bin size (MB)':<25s} | {r0['bin_size_mb']:>17.1f} | {r1['bin_size_mb']:>17.1f} | {r0['bin_size_mb']/r1['bin_size_mb']:>7.2f}x")

    # Init time
    print(f"  {'Init time (s)':<25s} | {r0['init_time']:>17.1f} | {r1['init_time']:>17.1f} | {r0['init_time']/r1['init_time']:>7.2f}x")

    # TTFT
    ttft_speedup = r0['avg_ttft'] / r1['avg_ttft'] if r1['avg_ttft'] > 0 else float('inf')
    print(f"  {'Avg TTFT (s)':<25s} | {r0['avg_ttft']:>17.3f} | {r1['avg_ttft']:>17.3f} | {ttft_speedup:>7.2f}x")

    # Decode speed
    decode_speedup = r1['avg_tok_per_sec'] / r0['avg_tok_per_sec'] if r0['avg_tok_per_sec'] > 0 else float('inf')
    print(f"  {'Avg decode (tok/s)':<25s} | {r0['avg_tok_per_sec']:>17.1f} | {r1['avg_tok_per_sec']:>17.1f} | {decode_speedup:>7.2f}x")

    # Total time
    total_speedup = r0['avg_total'] / r1['avg_total'] if r1['avg_total'] > 0 else float('inf')
    print(f"  {'Avg total time (s)':<25s} | {r0['avg_total']:>17.2f} | {r1['avg_total']:>17.2f} | {total_speedup:>7.2f}x")

    print()
    print("  Outputs:")
    for r in results:
        print(f"    {r['name']}: {r['output'][:80]}")
    print()

    # Verdict
    print("  VERDICT:", end=" ")
    if decode_speedup > 1.2:
        print(f"INT4_SYM is {decode_speedup:.1f}x FASTER for decode. Switch recommended!")
    elif decode_speedup > 0.9:
        print(f"INT4_SYM is roughly the same speed ({decode_speedup:.2f}x).")
    else:
        print(f"INT4_SYM is {1/decode_speedup:.1f}x SLOWER. Keep INT4_ASYM.")
    print()


if __name__ == "__main__":
    main()
