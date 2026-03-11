#!/usr/bin/env python3
"""Test various OpenVINO GPU optimization configurations for Qwen3.5.

Measures first-token latency and throughput across multiple GPU configs.

Usage (from WSL2):
    powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\\Apps\\translatorle; C:\\Users\\taowen\\.local\\bin\\uv.exe run python -m qwen35.scripts._test_gpu_opts'
"""

import gc
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.config import MODELS_DIR
from qwen35.inference import Qwen35OVModel

MODEL_PATH = str(MODELS_DIR / "Qwen3.5-0.8B-ov")
PROMPT = "Explain quantum computing in simple terms."
MAX_NEW_TOKENS = 50
DEVICE = "GPU"

# ---------------------------------------------------------------------------
# GPU configurations to test
# ---------------------------------------------------------------------------

CONFIGS = [
    ("Baseline", {}),
    ("LATENCY hint", {"PERFORMANCE_HINT": "LATENCY"}),
    ("THROUGHPUT hint", {"PERFORMANCE_HINT": "THROUGHPUT"}),
    ("Model cache", {"CACHE_DIR": str(MODELS_DIR / "qwen35" / "cache")}),
    ("LATENCY+Cache", {
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": str(MODELS_DIR / "qwen35" / "cache"),
    }),
    ("FP16 precision", {"INFERENCE_PRECISION_HINT": "f16"}),
]


def run_single_test(config_name: str, ov_config: dict) -> dict:
    """Load model with given config, generate tokens, return metrics."""
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"  ov_config: {ov_config}")
    print(f"{'='*60}")

    # Load model
    t_load_start = time.time()
    try:
        model = Qwen35OVModel.from_pretrained(
            MODEL_PATH,
            device=DEVICE,
            ov_config=ov_config if ov_config else None,
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return {
            "config": config_name,
            "load_time": -1,
            "first_token_ms": -1,
            "throughput": -1,
            "total_tokens": 0,
            "total_time": -1,
            "output": f"LOAD ERROR: {e}",
        }
    t_load = time.time() - t_load_start
    print(f"  Model loaded in {t_load:.1f}s")

    # Prepare input using chat template
    messages = [{"role": "user", "content": PROMPT}]
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = model.tokenizer(text, return_tensors="pt")
    n_input = inputs["input_ids"].shape[1]

    # --- Warmup run (discard) ---
    print("  Warmup run...")
    model.reset()
    try:
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    except Exception as e:
        print(f"  FAILED warmup: {e}")
        del model
        gc.collect()
        return {
            "config": config_name,
            "load_time": t_load,
            "first_token_ms": -1,
            "throughput": -1,
            "total_tokens": 0,
            "total_time": -1,
            "output": f"INFER ERROR: {e}",
        }

    # --- Timed run ---
    model.reset()

    # Measure first-token latency: generate just 1 token
    t_first_start = time.time()
    out_first = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    t_first = time.time() - t_first_start
    first_token_ms = t_first * 1000

    # Full generation: reset and generate MAX_NEW_TOKENS
    model.reset()
    t_gen_start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    t_gen = time.time() - t_gen_start

    generated = outputs[0][n_input:]
    n_gen = len(generated)
    throughput = n_gen / t_gen if t_gen > 0 else 0
    result_text = model.tokenizer.decode(generated, skip_special_tokens=True)

    print(f"  First token: {first_token_ms:.0f} ms")
    print(f"  Generated {n_gen} tokens in {t_gen:.2f}s = {throughput:.1f} tok/s")
    print(f"  Output: {result_text[:120]}...")

    # Cleanup
    del model
    gc.collect()

    return {
        "config": config_name,
        "load_time": t_load,
        "first_token_ms": first_token_ms,
        "throughput": throughput,
        "total_tokens": n_gen,
        "total_time": t_gen,
        "output": result_text[:80],
    }


def print_table(results: list):
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 90)
    print("  GPU Optimization Benchmark Results")
    print(f"  Model: Qwen3.5-0.8B-ov | Device: {DEVICE} | Prompt: {PROMPT!r}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print("=" * 90)

    # Header
    header = f"{'Config':<20} {'Load(s)':>8} {'1st tok(ms)':>12} {'tok/s':>8} {'Tokens':>7} {'Time(s)':>8}"
    print(header)
    print("-" * 90)

    for r in results:
        if r["throughput"] < 0:
            print(f"{r['config']:<20} {'FAILED':>8} {'---':>12} {'---':>8} {'---':>7} {'---':>8}  {r['output'][:30]}")
        else:
            print(
                f"{r['config']:<20} {r['load_time']:>8.1f} {r['first_token_ms']:>12.0f} "
                f"{r['throughput']:>8.1f} {r['total_tokens']:>7d} {r['total_time']:>8.2f}"
            )

    print("-" * 90)

    # Find best throughput (excluding failures)
    valid = [r for r in results if r["throughput"] > 0]
    if valid:
        best_tp = max(valid, key=lambda r: r["throughput"])
        best_lat = min(valid, key=lambda r: r["first_token_ms"])
        print(f"\n  Best throughput:     {best_tp['config']} ({best_tp['throughput']:.1f} tok/s)")
        print(f"  Best first-token:   {best_lat['config']} ({best_lat['first_token_ms']:.0f} ms)")

        baseline = next((r for r in valid if r["config"] == "Baseline"), None)
        if baseline and baseline["throughput"] > 0:
            print(f"\n  Speedup vs baseline:")
            for r in valid:
                speedup = r["throughput"] / baseline["throughput"]
                lat_ratio = baseline["first_token_ms"] / r["first_token_ms"] if r["first_token_ms"] > 0 else 0
                print(f"    {r['config']:<20} throughput: {speedup:.2f}x  first-token: {lat_ratio:.2f}x")

    print()


def main():
    print(f"GPU Optimization Benchmark for Qwen3.5-0.8B")
    print(f"Model path: {MODEL_PATH}")
    print(f"Testing {len(CONFIGS)} configurations...\n")

    results = []
    for name, config in CONFIGS:
        result = run_single_test(name, config)
        results.append(result)

    print_table(results)


if __name__ == "__main__":
    main()
