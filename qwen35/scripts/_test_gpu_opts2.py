#!/usr/bin/env python3
"""GPU optimization round 2: cache combos + INT4 quantized model.

Tests:
  1. Cache only (control, ~11.3 tok/s from round 1)
  2. Cache + FP16 precision hint
  3. Cache + THROUGHPUT hint
  4. Cache + FP16 + THROUGHPUT
  5. INT4_SYM quantized model + Cache
  6. INT4_SYM quantized model + Cache + FP16

Usage (from WSL2):
    powershell.exe -Command '$env:PYTHONIOENCODING="utf-8"; cd C:\\Apps\\translatorle; C:\\Users\\taowen\\.local\\bin\\uv.exe run python -m qwen35.scripts._test_gpu_opts2'
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
INT4_MODEL_PATH = str(MODELS_DIR / "Qwen3.5-0.8B-ov-int4")
CACHE_DIR = str(MODELS_DIR / "qwen35" / "cache")
PROMPT = "Explain quantum computing in simple terms."
MAX_NEW_TOKENS = 50
DEVICE = "GPU"


# ---------------------------------------------------------------------------
# INT4 quantization helper
# ---------------------------------------------------------------------------

def ensure_int4_model():
    """Create INT4_SYM quantized model if it doesn't exist."""
    int4_dir = Path(INT4_MODEL_PATH)
    int4_xml = int4_dir / "openvino_model.xml"
    if int4_xml.exists():
        print(f"  INT4 model already exists: {int4_xml}")
        return True

    print("  Creating INT4_SYM quantized model...")
    try:
        import nncf
        import openvino as ov
        import shutil

        src_dir = Path(MODEL_PATH)
        src_xml = src_dir / "openvino_model.xml"

        if not src_xml.exists():
            print(f"  ERROR: Source model not found: {src_xml}")
            return False

        core = ov.Core()
        t0 = time.time()
        model = core.read_model(str(src_xml))
        print(f"  Read model in {time.time() - t0:.1f}s")

        t0 = time.time()
        compressed = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT4_SYM,
            group_size=128,
        )
        print(f"  Compressed weights in {time.time() - t0:.1f}s")

        int4_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ov.save_model(compressed, str(int4_xml))
        print(f"  Saved INT4 model in {time.time() - t0:.1f}s")

        # Copy tokenizer and config files
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                       "embed_tokens.npy", "generation_config.json",
                       "special_tokens_map.json"]:
            src_file = src_dir / fname
            if src_file.exists():
                shutil.copy2(str(src_file), str(int4_dir / fname))

        print(f"  INT4 model saved to: {int4_dir}")

        # Show size comparison
        src_bin = src_dir / "openvino_model.bin"
        int4_bin = int4_dir / "openvino_model.bin"
        if src_bin.exists() and int4_bin.exists():
            src_mb = src_bin.stat().st_size / 1024 / 1024
            int4_mb = int4_bin.stat().st_size / 1024 / 1024
            print(f"  Size: FP16 {src_mb:.0f} MB -> INT4 {int4_mb:.0f} MB ({int4_mb/src_mb*100:.0f}%)")

        return True

    except Exception as e:
        print(f"  ERROR creating INT4 model: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test runner (same structure as round 1)
# ---------------------------------------------------------------------------

def run_single_test(config_name: str, model_path: str, ov_config: dict) -> dict:
    """Load model with given config, generate tokens, return metrics."""
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")
    print(f"  Model:  {Path(model_path).name}")
    print(f"  ov_config: {ov_config}")
    print(f"{'='*60}")

    # Load model
    t_load_start = time.time()
    try:
        model = Qwen35OVModel.from_pretrained(
            model_path,
            device=DEVICE,
            ov_config=ov_config if ov_config else None,
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()
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
    print(f"  Output: {result_text[:200]}")

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
        "output": result_text,
    }


def print_table(results: list):
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 100)
    print("  GPU Optimization Benchmark — Round 2")
    print(f"  Device: {DEVICE} | Prompt: {PROMPT!r}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print("=" * 100)

    # Header
    header = f"{'Config':<30} {'Load(s)':>8} {'1st(ms)':>8} {'tok/s':>8} {'Tokens':>7} {'Time(s)':>8}  {'Output (first 60 chars)'}"
    print(header)
    print("-" * 100)

    for r in results:
        if r["throughput"] < 0:
            print(f"{r['config']:<30} {'FAIL':>8} {'---':>8} {'---':>8} {'---':>7} {'---':>8}  {r['output'][:60]}")
        else:
            output_preview = r["output"][:60].replace("\n", " ")
            print(
                f"{r['config']:<30} {r['load_time']:>8.1f} {r['first_token_ms']:>8.0f} "
                f"{r['throughput']:>8.1f} {r['total_tokens']:>7d} {r['total_time']:>8.2f}  {output_preview}"
            )

    print("-" * 100)

    # Find best throughput (excluding failures)
    valid = [r for r in results if r["throughput"] > 0]
    if valid:
        best_tp = max(valid, key=lambda r: r["throughput"])
        best_lat = min(valid, key=lambda r: r["first_token_ms"])
        print(f"\n  Best throughput:   {best_tp['config']} ({best_tp['throughput']:.1f} tok/s)")
        print(f"  Best first-token:  {best_lat['config']} ({best_lat['first_token_ms']:.0f} ms)")

        # Speedup vs control (Cache only)
        control = next((r for r in valid if "Cache only" in r["config"]), None)
        if control and control["throughput"] > 0:
            print(f"\n  Speedup vs Cache-only control ({control['throughput']:.1f} tok/s):")
            for r in valid:
                speedup = r["throughput"] / control["throughput"]
                lat_ratio = control["first_token_ms"] / r["first_token_ms"] if r["first_token_ms"] > 0 else 0
                marker = " <-- best" if r is best_tp else ""
                print(f"    {r['config']:<30} throughput: {speedup:.2f}x  first-token: {lat_ratio:.2f}x{marker}")

    # Quality assessment
    print(f"\n  Output quality check:")
    for r in results:
        if r["throughput"] > 0:
            text = r["output"]
            # Simple coherence heuristics
            is_repetitive = len(set(text.split())) < len(text.split()) * 0.3 if text.split() else True
            has_garbage = any(c in text for c in ['\x00', '\ufffd'])
            is_short = len(text.strip()) < 10
            if is_repetitive or has_garbage or is_short:
                quality = "POOR (repetitive/garbage)"
            else:
                quality = "OK (coherent)"
            print(f"    {r['config']:<30} {quality}")
    print()


def main():
    print("=" * 60)
    print("  GPU Optimization Benchmark — Round 2")
    print("  Qwen3.5-0.8B on Intel Arc GPU")
    print("=" * 60)

    # --- Step 1: Ensure INT4 model exists ---
    print("\n--- Step 1: Prepare INT4 quantized model ---")
    has_int4 = ensure_int4_model()

    # --- Step 2: Define test configurations ---
    configs = []

    # FP16 model configs
    configs.append(("FP16 Cache only (control)", MODEL_PATH, {
        "CACHE_DIR": CACHE_DIR,
    }))

    configs.append(("FP16 Cache+FP16hint", MODEL_PATH, {
        "CACHE_DIR": CACHE_DIR,
        "INFERENCE_PRECISION_HINT": "f16",
    }))

    configs.append(("FP16 Cache+THROUGHPUT", MODEL_PATH, {
        "CACHE_DIR": CACHE_DIR,
        "PERFORMANCE_HINT": "THROUGHPUT",
    }))

    configs.append(("FP16 Cache+FP16+THROUGHPUT", MODEL_PATH, {
        "CACHE_DIR": CACHE_DIR,
        "INFERENCE_PRECISION_HINT": "f16",
        "PERFORMANCE_HINT": "THROUGHPUT",
    }))

    # INT4 model configs
    if has_int4:
        configs.append(("INT4 Cache only", INT4_MODEL_PATH, {
            "CACHE_DIR": CACHE_DIR,
        }))

        configs.append(("INT4 Cache+FP16hint", INT4_MODEL_PATH, {
            "CACHE_DIR": CACHE_DIR,
            "INFERENCE_PRECISION_HINT": "f16",
        }))

        configs.append(("INT4 Cache+THROUGHPUT", INT4_MODEL_PATH, {
            "CACHE_DIR": CACHE_DIR,
            "PERFORMANCE_HINT": "THROUGHPUT",
        }))

        configs.append(("INT4 Cache+FP16+THROUGHPUT", INT4_MODEL_PATH, {
            "CACHE_DIR": CACHE_DIR,
            "INFERENCE_PRECISION_HINT": "f16",
            "PERFORMANCE_HINT": "THROUGHPUT",
        }))

    print(f"\n--- Step 2: Running {len(configs)} test configurations ---")

    results = []
    for name, mpath, config in configs:
        result = run_single_test(name, mpath, config)
        results.append(result)

    # --- Step 3: Print results ---
    print_table(results)


if __name__ == "__main__":
    main()
