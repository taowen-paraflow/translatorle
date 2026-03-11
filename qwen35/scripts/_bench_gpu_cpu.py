#!/usr/bin/env python3
"""Benchmark Qwen3.5 inference: GPU vs CPU with various OpenVINO configs.

Tests 5 configurations, each with a fresh model load:
  1. CPU default
  2. GPU default
  3. GPU + CACHE_DIR
  4. GPU + INFERENCE_PRECISION_HINT=f16
  5. GPU + CACHE_DIR + f16

Usage:
    uv run python -m qwen35.scripts._bench_gpu_cpu
"""

import gc
import sys
import time
from dataclasses import replace
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.config import MODELS_DIR, QWEN35_MODEL_CPU, QWEN35_MODEL_GPU, Qwen35ModelConfig

PROMPT = "Explain the theory of relativity in detail."
WARMUP_TOKENS = 10
BENCH_TOKENS = 100

GPU_CACHE_DIR = str(MODELS_DIR / "gpu_cache")


def build_configs() -> list[tuple[str, Qwen35ModelConfig]]:
    """Return (label, config) pairs for each benchmark variant."""
    configs = []

    # 1. CPU default
    configs.append(("CPU default", QWEN35_MODEL_CPU))

    # 2. GPU default
    configs.append(("GPU default", QWEN35_MODEL_GPU))

    # 3. GPU + CACHE_DIR
    configs.append((
        "GPU + cache",
        replace(QWEN35_MODEL_GPU, ov_config={
            "CACHE_DIR": GPU_CACHE_DIR,
        }),
    ))

    # 4. GPU + INFERENCE_PRECISION_HINT f16
    configs.append((
        "GPU + f16 hint",
        replace(QWEN35_MODEL_GPU, ov_config={
            "INFERENCE_PRECISION_HINT": "f16",
        }),
    ))

    # 5. GPU + CACHE_DIR + f16
    configs.append((
        "GPU + cache + f16",
        replace(QWEN35_MODEL_GPU, ov_config={
            "CACHE_DIR": GPU_CACHE_DIR,
            "INFERENCE_PRECISION_HINT": "f16",
        }),
    ))

    return configs


def run_benchmark(label: str, cfg: Qwen35ModelConfig) -> dict:
    """Load model, warmup, then generate BENCH_TOKENS and return metrics."""
    from qwen35.inference import Qwen35OVModel

    model_dir = str(Path(cfg.model_xml).parent)

    # --- Load ---
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  device={cfg.device}  ov_config={cfg.ov_config or '{}'}")
    print(f"{'=' * 60}")

    t_load_start = time.perf_counter()
    model = Qwen35OVModel.from_pretrained(
        model_dir, device=cfg.device, ov_config=cfg.ov_config or None,
    )
    t_load = time.perf_counter() - t_load_start
    print(f"  Loaded in {t_load:.1f}s")

    # --- Warmup ---
    print(f"  Warming up ({WARMUP_TOKENS} tokens)...")
    inputs = model.tokenizer(PROMPT, return_tensors="pt")
    model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)
    # Reset state for the real run
    model._request.reset_state()

    # --- Timed run ---
    print(f"  Generating {BENCH_TOKENS} tokens...")
    inputs = model.tokenizer(PROMPT, return_tensors="pt")
    num_input = inputs["input_ids"].shape[1]

    t_gen_start = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=BENCH_TOKENS, do_sample=False)
    t_gen_total = time.perf_counter() - t_gen_start

    generated_ids = outputs[0][num_input:]
    num_generated = len(generated_ids)
    text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # First-token latency: approximate from total time
    # (model.generate doesn't expose per-token timing, so we compute average)
    if num_generated > 1:
        # Rough estimate: first token is slower, rest at steady state
        avg_tok_s = num_generated / t_gen_total
    else:
        avg_tok_s = 0

    # --- Cleanup ---
    del model
    gc.collect()

    result = {
        "label": label,
        "device": cfg.device,
        "load_s": t_load,
        "gen_tokens": num_generated,
        "gen_total_s": t_gen_total,
        "tok_per_s": avg_tok_s,
        "text_preview": text[:80],
    }
    print(f"  => {num_generated} tokens in {t_gen_total:.1f}s = {avg_tok_s:.1f} tok/s")
    print(f"  => \"{text[:80]}...\"")
    return result


def main():
    configs = build_configs()
    results = []

    print(f"Qwen3.5-0.8B GPU vs CPU Benchmark")
    print(f"Prompt: \"{PROMPT}\"")
    print(f"Warmup: {WARMUP_TOKENS} tokens, Bench: {BENCH_TOKENS} tokens")
    print(f"Configs to test: {len(configs)}")

    for label, cfg in configs:
        try:
            r = run_benchmark(label, cfg)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR in '{label}': {e}")
            results.append({
                "label": label,
                "device": cfg.device,
                "load_s": -1,
                "gen_tokens": 0,
                "gen_total_s": -1,
                "tok_per_s": 0,
                "text_preview": f"ERROR: {e}",
            })

    # --- Print markdown table ---
    print(f"\n\n{'=' * 80}")
    print("## Results\n")
    print("| Config | Device | Load (s) | Tokens | Time (s) | tok/s | Output preview |")
    print("|--------|--------|----------|--------|----------|-------|----------------|")
    for r in results:
        load = f"{r['load_s']:.1f}" if r['load_s'] >= 0 else "ERR"
        gen_t = f"{r['gen_total_s']:.1f}" if r['gen_total_s'] >= 0 else "ERR"
        tps = f"{r['tok_per_s']:.1f}" if r['tok_per_s'] > 0 else "-"
        preview = r["text_preview"][:50].replace("|", "/")
        print(f"| {r['label']:<20s} | {r['device']:<6s} | {load:>8s} | {r['gen_tokens']:>6d} | {gen_t:>8s} | {tps:>5s} | {preview} |")

    print()


if __name__ == "__main__":
    main()
