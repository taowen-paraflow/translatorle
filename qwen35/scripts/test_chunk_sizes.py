r"""Benchmark HYBRID prefill performance with different NPU attention chunk sizes.

Compares prefill latency across chunk sizes (16, 32, 64, 128, 256) for the
HYBRID configuration (GDN=GPU, Attn=NPU, Head=GPU, explicit I/O). Includes
a GPU_ONLY baseline for reference. Models are loaded one at a time to save
memory.

For each prompt, runs 2 warmup + 3 timed iterations (reports best time).
Verifies first-token correctness against GPU_ONLY baseline and generates 5
decode tokens to show output.

Usage:
  powershell.exe -Command 'cd C:\Apps\translatorle; $env:PYTHONIOENCODING="utf-8"; C:\Users\taowen\.local\bin\uv.exe run python -m qwen35.scripts.test_chunk_sizes'
"""

import logging
import time
from collections import OrderedDict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "models/qwen35/Qwen3.5-0.8B-hybrid"
NUM_WARMUP = 2
NUM_TIMED = 3
NUM_DECODE_TOKENS = 5

CHUNK_SIZES = [16, 32, 64, 128, 256]

PROMPTS = [
    ("hello", "Hello"),
    ("short_en", "The capital of France is"),
    ("long_math", (
        "A farmer has 15 chickens and 12 cows. Each chicken eats 0.5kg of feed "
        "per day and each cow eats 8kg of feed per day. How much feed does the "
        "farmer need for a week? Show your step-by-step calculation."
    )),
    ("long_code", (
        "Write a Python function that takes a list of integers and returns the "
        "second largest unique element. Handle edge cases like empty lists and "
        "lists with all duplicate values. Include docstrings and type hints."
    )),
]

# Config name -> constructor kwargs
# GPU_ONLY first (baseline), then HYBRID with each chunk size.
CONFIGS = OrderedDict()
CONFIGS["GPU_ONLY"] = dict(
    gdn_device="GPU",
    attn_device="GPU",
    head_device="GPU",
    attn_stateful=True,
    prefill_chunk_size=16,
)
for _cs in CHUNK_SIZES:
    CONFIGS[f"HYBRID cs={_cs}"] = dict(
        gdn_device="GPU",
        attn_device="NPU",
        head_device="GPU",
        attn_stateful=False,
        prefill_chunk_size=_cs,
    )


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


def benchmark_prompt(model, input_ids):
    """Run warmup + timed prefill runs.

    Returns (best_time_sec, logits_from_best_run).
    """
    for _ in range(NUM_WARMUP):
        run_prefill(model, input_ids)

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

    # results[config_name][prompt_name] = {
    #     "prompt_len", "prefill_ms", "tok_per_s",
    #     "first_token", "output_text"
    # }
    results = OrderedDict()

    # Tokenize all prompts once (tokenizer is identical across configs).
    # We grab it from the first model loaded.
    tokenizer = None
    prompt_inputs = OrderedDict()  # prompt_name -> (input_ids, prompt_len)

    for config_name, kwargs in CONFIGS.items():
        logger.info("")
        logger.info("=" * 70)
        logger.info("Loading config: %s", config_name)
        logger.info("  kwargs: %s", kwargs)
        logger.info("=" * 70)

        model = Qwen35HybridModel(model_dir=MODEL_DIR, **kwargs)

        # Grab tokenizer from first model
        if tokenizer is None:
            tokenizer = model._tokenizer
            for prompt_name, prompt_text in PROMPTS:
                token_list = tokenizer.encode(prompt_text)
                input_ids = np.array([token_list], dtype=np.int64)
                prompt_inputs[prompt_name] = (input_ids, input_ids.shape[1])

        config_results = OrderedDict()

        for prompt_name, prompt_text in PROMPTS:
            input_ids, prompt_len = prompt_inputs[prompt_name]
            logger.info("  Benchmarking prompt '%s' (%d tokens) ...",
                        prompt_name, prompt_len)

            # Benchmark prefill
            best_time, best_logits = benchmark_prompt(model, input_ids)
            first_token = int(np.argmax(best_logits[0, -1, :]))

            # Generate decode tokens (run fresh prefill to prime state)
            model.reset()
            model.prefill(input_ids)
            gen_ids = generate_tokens(model, first_token, NUM_DECODE_TOKENS)
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            prefill_ms = best_time * 1000
            tok_per_s = prompt_len / best_time if best_time > 0 else 0.0

            config_results[prompt_name] = {
                "prompt_len": prompt_len,
                "prefill_ms": prefill_ms,
                "tok_per_s": tok_per_s,
                "first_token": first_token,
                "output_text": gen_text,
            }

            logger.info("    -> %.1f ms (%.1f tok/s), first_token=%d",
                        prefill_ms, tok_per_s, first_token)

        results[config_name] = config_results

        # Free model memory before loading next config
        del model
        logger.info("  Model unloaded.")

    # ------------------------------------------------------------------
    # Print per-prompt comparison tables
    # ------------------------------------------------------------------
    gpu_results = results.get("GPU_ONLY", {})

    for prompt_name, prompt_text in PROMPTS:
        _, prompt_len = prompt_inputs[prompt_name]
        prompt_display = (prompt_text if len(prompt_text) <= 40
                          else prompt_text[:37] + "...")

        print()
        print(f"=== Prompt: {prompt_name} ({prompt_len} tokens) ===")
        print(f"    \"{prompt_display}\"")
        print(f"{'Config':<16}| {'Prefill ms':>10} | {'tok/s':>8} | "
              f"{'1st tok':>7} | Output")
        print("-" * 90)

        for config_name in CONFIGS:
            r = results[config_name][prompt_name]

            if config_name == "GPU_ONLY":
                match_str = "(ref)"
            else:
                gpu_r = gpu_results.get(prompt_name)
                if gpu_r and r["first_token"] == gpu_r["first_token"]:
                    match_str = "OK"
                else:
                    match_str = "DIFF"

            output_snippet = r["output_text"][:45].replace("\n", " ")
            print(f"{config_name:<16}| {r['prefill_ms']:>10.1f} | "
                  f"{r['tok_per_s']:>8.1f} | {match_str:>7} | "
                  f"{output_snippet}")

        print()

    # ------------------------------------------------------------------
    # Summary table: average prefill performance per config
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY: Average prefill performance across all prompts")
    print("=" * 80)
    print(f"{'Config':<16}| {'Avg ms':>8} | {'Avg tok/s':>10} | "
          f"{'vs GPU_ONLY':>11} | {'Token match':>11}")
    print("-" * 80)

    gpu_avg_ms = 0.0
    if gpu_results:
        gpu_avg_ms = (sum(gpu_results[pn]["prefill_ms"] for pn, _ in PROMPTS)
                      / len(PROMPTS))

    for config_name in CONFIGS:
        cr = results[config_name]
        avg_ms = (sum(cr[pn]["prefill_ms"] for pn, _ in PROMPTS)
                  / len(PROMPTS))
        avg_tps = (sum(cr[pn]["tok_per_s"] for pn, _ in PROMPTS)
                   / len(PROMPTS))

        if config_name == "GPU_ONLY":
            speedup_str = "(baseline)"
        elif gpu_avg_ms > 0:
            speedup = gpu_avg_ms / avg_ms
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        if config_name == "GPU_ONLY":
            match_str = "(ref)"
        else:
            num_match = sum(
                1 for pn, _ in PROMPTS
                if (gpu_results.get(pn)
                    and cr[pn]["first_token"] == gpu_results[pn]["first_token"])
            )
            match_str = f"{num_match}/{len(PROMPTS)}"

        print(f"{config_name:<16}| {avg_ms:>8.1f} | {avg_tps:>10.1f} | "
              f"{speedup_str:>11} | {match_str:>11}")

    print("=" * 80)

    # ------------------------------------------------------------------
    # Per-prompt speedup matrix (HYBRID configs vs GPU_ONLY)
    # ------------------------------------------------------------------
    print()
    print("Per-prompt speedup vs GPU_ONLY:")

    header = f"{'Prompt':<12} {'Tok':>4}"
    for config_name in CONFIGS:
        if config_name == "GPU_ONLY":
            continue
        label = config_name.replace("HYBRID ", "")
        header += f" | {label:>8}"
    print(header)
    print("-" * len(header))

    for prompt_name, _ in PROMPTS:
        _, prompt_len = prompt_inputs[prompt_name]
        gpu_ms = (gpu_results[prompt_name]["prefill_ms"]
                  if gpu_results else 0)
        row = f"{prompt_name:<12} {prompt_len:>4}"
        for config_name in CONFIGS:
            if config_name == "GPU_ONLY":
                continue
            hyb_ms = results[config_name][prompt_name]["prefill_ms"]
            if hyb_ms > 0 and gpu_ms > 0:
                speedup = gpu_ms / hyb_ms
                row += f" | {speedup:>7.2f}x"
            else:
                row += f" | {'N/A':>8}"
        print(row)

    print()


if __name__ == "__main__":
    main()
