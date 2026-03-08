"""Benchmark NPU Dynamic Quantization on HY-MT INT4_SYM model.

Tests three NPU configurations:
  A) Baseline INT4_SYM (current config)
  B) INT4_SYM + NPU_COMPILER_DYNAMIC_QUANTIZATION=YES
  C) INT4_SYM + NPU_COMPILER_DYNAMIC_QUANTIZATION=YES + NPU_QDQ_OPTIMIZATION=YES

Each config uses a separate cache dir to avoid conflicts.

Usage:
    uv run python hymt/scripts/bench_dynquant.py
"""

import gc
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path

MODELS_DIR = Path(PROJECT_ROOT) / "models"
MODEL_DIR = str(MODELS_DIR / "hy_mt_int4sym")

PROMPT = (
    "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n"
    "The quick brown fox jumps over the lazy dog."
)

CONFIGS = {
    "A: Baseline INT4_SYM": {
        "npu_config": {
            "MAX_PROMPT_LEN": 512,
            "NPUW_LLM_PREFILL_CHUNK_SIZE": 512,
            "GENERATE_HINT": "BEST_PERF",
        },
        "cache_dir": str(MODELS_DIR / "hy_mt_cache_sym"),
    },
    "B: + DynamicQuant": {
        "npu_config": {
            "MAX_PROMPT_LEN": 512,
            "NPUW_LLM_PREFILL_CHUNK_SIZE": 512,
            "GENERATE_HINT": "BEST_PERF",
            "NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES",
        },
        "cache_dir": str(MODELS_DIR / "hy_mt_cache_dynq"),
    },
    "C: + DynQuant + QDQ": {
        "npu_config": {
            "MAX_PROMPT_LEN": 512,
            "NPUW_LLM_PREFILL_CHUNK_SIZE": 512,
            "GENERATE_HINT": "BEST_PERF",
            "NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES",
            "NPU_QDQ_OPTIMIZATION": "YES",
        },
        "cache_dir": str(MODELS_DIR / "hy_mt_cache_dynq_qdq"),
    },
}

NUM_WARMUP = 1
NUM_TIMED = 3


def bench_one_run(pipe, prompt, max_new_tokens=128):
    """Run one generation and return (ttft, decode_time, n_tokens, output)."""
    token_times = []
    first_token_time = [None]

    def streamer(subword):
        now = time.perf_counter()
        if first_token_time[0] is None:
            first_token_time[0] = now
        token_times.append(now)
        return False

    t_start = time.perf_counter()
    result = pipe.generate(prompt, max_new_tokens=max_new_tokens, streamer=streamer)
    t_end = time.perf_counter()

    n_tokens = len(token_times)
    ttft = first_token_time[0] - t_start if first_token_time[0] else 0
    total = t_end - t_start
    decode_time = total - ttft

    if n_tokens > 1 and decode_time > 0:
        tok_per_sec = (n_tokens - 1) / decode_time
    else:
        tok_per_sec = 0.0

    return {
        "ttft": ttft,
        "decode_time": decode_time,
        "total": total,
        "n_tokens": n_tokens,
        "tok_per_sec": tok_per_sec,
        "output": result.strip(),
    }


def bench_config(name, npu_config, cache_dir):
    """Benchmark a single NPU config. Returns dict of averaged metrics or error string."""
    import openvino_genai as ov_genai

    print(f"\n{'='*70}")
    print(f"  Config: {name}")
    print(f"  NPU properties: {npu_config}")
    print(f"  Cache dir: {cache_dir}")
    print(f"{'='*70}")

    # Build pipeline kwargs
    kwargs = {**npu_config, "CACHE_DIR": cache_dir}

    # Create pipeline
    try:
        t_init_start = time.perf_counter()
        pipe = ov_genai.LLMPipeline(MODEL_DIR, "NPU", **kwargs)
        t_init = time.perf_counter() - t_init_start
        print(f"  Pipeline created in {t_init:.1f}s")
    except Exception as e:
        msg = f"  FAILED to create pipeline: {e}"
        print(msg)
        traceback.print_exc()
        return {"error": str(e)}

    # Warmup
    print(f"  Warmup ({NUM_WARMUP} run)...")
    try:
        for i in range(NUM_WARMUP):
            r = bench_one_run(pipe, PROMPT)
            print(f"    warmup {i+1}: {r['n_tokens']} tokens, {r['tok_per_sec']:.1f} tok/s")
            print(f"    output: {r['output'][:80]}")
    except Exception as e:
        msg = f"  FAILED during warmup: {e}"
        print(msg)
        traceback.print_exc()
        del pipe
        gc.collect()
        return {"error": str(e)}

    # Timed runs
    print(f"  Timed runs ({NUM_TIMED})...")
    runs = []
    for i in range(NUM_TIMED):
        try:
            r = bench_one_run(pipe, PROMPT)
            runs.append(r)
            print(
                f"    run {i+1}: {r['n_tokens']} tokens, "
                f"TTFT={r['ttft']:.3f}s, "
                f"decode={r['decode_time']:.3f}s, "
                f"tok/s={r['tok_per_sec']:.1f}"
            )
        except Exception as e:
            print(f"    run {i+1} FAILED: {e}")
            traceback.print_exc()

    # Cleanup
    del pipe
    gc.collect()

    if not runs:
        return {"error": "All timed runs failed"}

    # Average metrics
    avg = {
        "init_time": t_init,
        "ttft": sum(r["ttft"] for r in runs) / len(runs),
        "decode_time": sum(r["decode_time"] for r in runs) / len(runs),
        "total": sum(r["total"] for r in runs) / len(runs),
        "tok_per_sec": sum(r["tok_per_sec"] for r in runs) / len(runs),
        "n_tokens": runs[0]["n_tokens"],
        "n_runs": len(runs),
    }
    return avg


def main():
    import openvino_genai as ov_genai

    print(f"openvino_genai version: {ov_genai.__version__}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Prompt: {PROMPT[:60]}...")
    print(f"Warmup runs: {NUM_WARMUP}, Timed runs: {NUM_TIMED}")

    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = bench_config(name, cfg["npu_config"], cfg["cache_dir"])

    # Summary table
    print(f"\n\n{'='*80}")
    print("  SUMMARY: Dynamic Quantization Benchmark on NPU")
    print(f"{'='*80}")
    print(
        f"{'Config':<30s} | {'Init':>6s} | {'TTFT':>7s} | {'Decode':>7s} | "
        f"{'Total':>7s} | {'tok/s':>6s} | {'Tokens':>6s}"
    )
    print("-" * 85)

    for name, r in results.items():
        if "error" in r:
            print(f"{name:<30s} | FAILED: {r['error'][:50]}")
        else:
            print(
                f"{name:<30s} | {r['init_time']:5.1f}s | {r['ttft']:6.3f}s | "
                f"{r['decode_time']:6.3f}s | {r['total']:6.3f}s | "
                f"{r['tok_per_sec']:5.1f} | {r['n_tokens']:6d}"
            )

    # Comparison vs baseline
    baseline = results.get("A: Baseline INT4_SYM")
    if baseline and "error" not in baseline:
        print(f"\n  Speedup vs baseline ({baseline['tok_per_sec']:.1f} tok/s):")
        for name, r in results.items():
            if "error" not in r and name != "A: Baseline INT4_SYM":
                speedup = r["tok_per_sec"] / baseline["tok_per_sec"] if baseline["tok_per_sec"] > 0 else 0
                delta = r["tok_per_sec"] - baseline["tok_per_sec"]
                print(f"    {name}: {r['tok_per_sec']:.1f} tok/s ({delta:+.1f}, {speedup:.2f}x)")

    print()


if __name__ == "__main__":
    main()
