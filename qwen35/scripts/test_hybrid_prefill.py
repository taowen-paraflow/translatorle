r"""Benchmark HYBRID vs GPU_ONLY prefill performance.

Loads two Qwen35HybridModel instances (GPU_ONLY and HYBRID) and compares
prefill latency across prompts of varying lengths. For each prompt, runs
2 warmup iterations followed by 3 timed runs (reports best time). Also
generates 10 tokens to verify decode works after prefill.

Usage:
  powershell.exe -Command 'cd C:\Apps\translatorle; $env:PYTHONIOENCODING="utf-8"; C:\Users\taowen\.local\bin\uv.exe run python -m qwen35.scripts.test_hybrid_prefill'
"""

import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models/qwen35/Qwen3.5-0.8B-hybrid"
NUM_WARMUP = 2
NUM_TIMED = 3
NUM_DECODE_TOKENS = 10

PROMPTS = [
    ("very_short", "Hello"),
    ("short_en", "The capital of France is"),
    ("math_tiny", "1+1="),
    ("medium_en", "Explain the theory of relativity in simple terms. Albert Einstein proposed"),
    ("short_cn", "\u8bf7\u7528\u4e00\u53e5\u8bdd\u89e3\u91ca\u4ec0\u4e48\u662f\u673a\u5668\u5b66\u4e60"),
    ("long_math", (
        "A farmer has 15 chickens and 12 cows. Each chicken eats 0.5kg of feed "
        "per day and each cow eats 8kg of feed per day. How much feed does the "
        "farmer need for a week? Show your step-by-step calculation."
    )),
    ("long_code", (
        "Write a Python function that takes a list of integers and returns the "
        "second largest unique element. Handle edge cases like empty lists and "
        "lists with all duplicate values."
    )),
]


def run_prefill(model, input_ids):
    """Reset model and run prefill, returning logits and elapsed seconds."""
    model.reset()
    t0 = time.perf_counter()
    logits = model.prefill(input_ids)
    elapsed = time.perf_counter() - t0
    return logits, elapsed


def generate_tokens(model, first_token_id, num_tokens):
    """Decode num_tokens after prefill (model state already primed).

    Returns list of generated token IDs (including first_token_id).
    """
    generated = [first_token_id]
    next_id = first_token_id
    eos_ids = {151645, 151643}  # <|im_end|>, <|endoftext|>
    if model._tokenizer.eos_token_id:
        eos_ids.add(model._tokenizer.eos_token_id)

    for _ in range(num_tokens - 1):
        if next_id in eos_ids:
            break
        token_input = np.array([[next_id]], dtype=np.int64)
        logits = model.forward(token_input)
        next_id = int(np.argmax(logits[0, -1, :]))
        generated.append(next_id)
    return generated


def benchmark_prompt(model, input_ids, prompt_len):
    """Run warmup + timed prefill runs.

    Returns (best_time_sec, logits_from_best_run).
    """
    # Warmup
    for _ in range(NUM_WARMUP):
        run_prefill(model, input_ids)

    # Timed runs
    best_time = float("inf")
    best_logits = None
    for _ in range(NUM_TIMED):
        logits, elapsed = run_prefill(model, input_ids)
        if elapsed < best_time:
            best_time = elapsed
            best_logits = logits
    return best_time, best_logits


def main():
    from qwen35.inference_hybrid import Qwen35HybridModel

    # --- Load GPU_ONLY model ---
    logger.info("=" * 70)
    logger.info("Loading GPU_ONLY model (GDN=GPU, Attn=GPU, Head=GPU, stateful)")
    logger.info("=" * 70)
    model_gpu = Qwen35HybridModel(
        model_dir=MODEL_DIR,
        gdn_device="GPU",
        attn_device="GPU",
        head_device="GPU",
        attn_stateful=True,
        prefill_chunk_size=16,
    )

    # --- Load HYBRID model ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("Loading HYBRID model (GDN=GPU, Attn=NPU, Head=GPU, explicit I/O)")
    logger.info("=" * 70)
    model_hybrid = Qwen35HybridModel(
        model_dir=MODEL_DIR,
        gdn_device="GPU",
        attn_device="NPU",
        head_device="GPU",
        attn_stateful=False,
        prefill_chunk_size=16,
    )

    tokenizer = model_gpu._tokenizer

    # --- Per-prompt results ---
    print()
    print("=" * 110)
    print(f"{'Prompt':<12} {'Tok':>4} | {'GPU_ONLY ms':>11} {'tok/s':>7} | "
          f"{'HYBRID ms':>10} {'tok/s':>7} | {'Speedup':>7} {'Match':>5} | Generated text")
    print("-" * 110)

    results = []  # (name, tok, gpu_ms, gpu_tps, hyb_ms, hyb_tps, speedup, match)

    for name, prompt in PROMPTS:
        token_list = tokenizer.encode(prompt)
        input_ids = np.array([token_list], dtype=np.int64)
        prompt_len = input_ids.shape[1]

        # Benchmark GPU_ONLY
        gpu_time, gpu_logits = benchmark_prompt(model_gpu, input_ids, prompt_len)
        gpu_first_token = int(np.argmax(gpu_logits[0, -1, :]))

        # Generate 10 tokens with GPU_ONLY (model state is from last prefill)
        model_gpu.reset()
        model_gpu.prefill(input_ids)
        gpu_gen_ids = generate_tokens(model_gpu, gpu_first_token, NUM_DECODE_TOKENS)
        gpu_gen_text = tokenizer.decode(gpu_gen_ids, skip_special_tokens=True)

        # Benchmark HYBRID
        hyb_time, hyb_logits = benchmark_prompt(model_hybrid, input_ids, prompt_len)
        hyb_first_token = int(np.argmax(hyb_logits[0, -1, :]))

        # Generate 10 tokens with HYBRID
        model_hybrid.reset()
        model_hybrid.prefill(input_ids)
        hyb_gen_ids = generate_tokens(model_hybrid, hyb_first_token, NUM_DECODE_TOKENS)
        hyb_gen_text = tokenizer.decode(hyb_gen_ids, skip_special_tokens=True)

        # Compute metrics
        gpu_ms = gpu_time * 1000
        hyb_ms = hyb_time * 1000
        gpu_tps = prompt_len / gpu_time if gpu_time > 0 else 0
        hyb_tps = prompt_len / hyb_time if hyb_time > 0 else 0
        speedup = gpu_time / hyb_time if hyb_time > 0 else float("inf")
        match = "OK" if gpu_first_token == hyb_first_token else "DIFF"

        results.append((name, prompt_len, gpu_ms, gpu_tps, hyb_ms, hyb_tps, speedup, match))

        # Truncate generated text for display
        gen_snippet = hyb_gen_text[:50].replace("\n", " ")

        print(f"{name:<12} {prompt_len:>4} | {gpu_ms:>9.1f}ms {gpu_tps:>6.1f} | "
              f"{hyb_ms:>8.1f}ms {hyb_tps:>6.1f} | {speedup:>6.2f}x {match:>5} | {gen_snippet}")

    # --- Summary ---
    print("-" * 110)
    avg_gpu_ms = sum(r[2] for r in results) / len(results)
    avg_hyb_ms = sum(r[4] for r in results) / len(results)
    avg_gpu_tps = sum(r[3] for r in results) / len(results)
    avg_hyb_tps = sum(r[5] for r in results) / len(results)
    avg_speedup = avg_gpu_ms / avg_hyb_ms if avg_hyb_ms > 0 else float("inf")
    num_match = sum(1 for r in results if r[7] == "OK")
    total = len(results)

    print(f"{'AVERAGE':<12} {'':>4} | {avg_gpu_ms:>9.1f}ms {avg_gpu_tps:>6.1f} | "
          f"{avg_hyb_ms:>8.1f}ms {avg_hyb_tps:>6.1f} | {avg_speedup:>6.2f}x "
          f"{num_match}/{total:>3} |")
    print("=" * 110)


if __name__ == "__main__":
    main()
