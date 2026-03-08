"""Benchmark ASR decoder: FP16 vs INT4 on NPU (NPUW_LLM).

Measures prefill latency, per-token decode time, and throughput for both the
original FP16 model and the INT4-quantized model.  Uses the same NPUW_LLM
compilation config as the production OVDecoder.

If a model fails to compile on NPU with NPUW_LLM, the script tries:
  1. NPU with NPUW (no LLM mode)
  2. CPU with LATENCY hint
and reports which device was actually used.

Usage (from WSL, targeting Windows Python):
    powershell.exe -Command '
        $env:PYTHONIOENCODING = "utf-8";
        cd C:\\Apps\\translatorle;
        C:\\Users\\taowen\\.local\\bin\\uv.exe run python asr/scripts/bench_decoder.py
    '
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import openvino as ov

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

MODELS = {
    "FP16": {
        "xml": MODELS_DIR / "decoder_stateful_embeds" / "openvino_model.xml",
        "embed": MODELS_DIR / "embed_tokens.npy",
    },
    "INT4": {
        "xml": MODELS_DIR / "decoder_stateful_int4" / "openvino_model.xml",
        "embed": MODELS_DIR / "decoder_stateful_int4" / "embed_tokens.npy",
    },
}

# Same config as asr/config.py -> NPU_DECODER_CONFIG
NPU_DECODER_CONFIG = {
    "NPU_USE_NPUW": "YES",
    "NPUW_LLM": "YES",
    "NPUW_LLM_BATCH_DIM": 0,
    "NPUW_LLM_SEQ_LEN_DIM": 2,
    "NPUW_LLM_MAX_PROMPT_LEN": 256,
    "NPUW_LLM_MIN_RESPONSE_LEN": 64,
}

# Fallback configs to try if NPUW_LLM fails
NPU_NPUW_CONFIG = {
    "NPU_USE_NPUW": "YES",
}

CPU_CONFIG = {
    "PERFORMANCE_HINT": "LATENCY",
}

# Benchmark parameters
PREFILL_SEQ_LEN = 128   # Simulates a typical prompt length
EMBED_DIM = 1024
NUM_SEQUENCES = 5        # Number of full sequences to run
DECODE_TOKENS = 20       # Tokens to decode per sequence
WARMUP_SEQUENCES = 1     # Warmup runs (excluded from timing)


def compile_model_with_fallback(
    core: ov.Core, xml_path: Path, label: str
) -> tuple[ov.CompiledModel, str]:
    """Try to compile on NPU with NPUW_LLM, falling back if needed.

    Returns (compiled_model, device_description).
    """
    strategies = [
        ("NPU (NPUW_LLM)", "NPU", NPU_DECODER_CONFIG),
        ("NPU (NPUW only)", "NPU", NPU_NPUW_CONFIG),
        ("CPU (LATENCY)", "CPU", CPU_CONFIG),
    ]

    for desc, device, config in strategies:
        print(f"  Trying {desc}...")
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(str(xml_path), device, config)
            elapsed = time.perf_counter() - t0
            print(f"    OK - compile time: {elapsed:.2f}s")
            return compiled, desc
        except RuntimeError as e:
            err_short = str(e).split("\n")[0][:120]
            print(f"    FAILED: {err_short}")
            continue

    raise RuntimeError(f"All compilation strategies failed for {label}")


def run_prefill(request: ov.InferRequest, inputs_embeds: np.ndarray) -> tuple[int, float]:
    """Run a prefill step and return (token_id, elapsed_seconds)."""
    seq_len = inputs_embeds.shape[1]
    t0 = time.perf_counter()
    request.infer({
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
        "beam_idx": np.array([0], dtype=np.int32),
    })
    elapsed = time.perf_counter() - t0
    logits = request.get_output_tensor(0).data
    token_id = int(np.argmax(logits[0, -1, :]))
    return token_id, elapsed


def run_decode_step(
    request: ov.InferRequest,
    embed_table: np.ndarray,
    token_id: int,
    past_len: int,
) -> tuple[int, float]:
    """Run one decode step and return (next_token_id, elapsed_seconds)."""
    token_embed = embed_table[token_id][np.newaxis, np.newaxis, :]  # [1,1,1024]
    t0 = time.perf_counter()
    request.infer({
        "inputs_embeds": token_embed.astype(np.float32),
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
        "position_ids": np.array([[past_len]], dtype=np.int64),
        "beam_idx": np.array([0], dtype=np.int32),
    })
    elapsed = time.perf_counter() - t0
    logits = request.get_output_tensor(0).data
    next_token = int(np.argmax(logits[0, -1, :]))
    return next_token, elapsed


def benchmark_model(
    core: ov.Core,
    label: str,
    xml_path: Path,
    embed_path: Path,
) -> dict | None:
    """Benchmark a single model and return timing results."""
    print()
    print("-" * 60)
    print(f"  Benchmarking: {label}")
    print("-" * 60)

    # Load embedding table
    print(f"  Loading embed_tokens from {embed_path}...")
    embed_table = np.load(str(embed_path))
    print(f"    Shape: {embed_table.shape}, dtype: {embed_table.dtype}")

    # Compile model (with fallback)
    print(f"  Compiling {label} model...")
    print(f"    XML: {xml_path}")
    try:
        compiled, device_desc = compile_model_with_fallback(core, xml_path, label)
    except RuntimeError as e:
        print(f"  ERROR: Could not compile {label} on any device: {e}")
        return None

    request = compiled.create_infer_request()

    # Create dummy inputs_embeds for prefill (random, fp32)
    rng = np.random.default_rng(42)
    dummy_embeds = rng.standard_normal(
        (1, PREFILL_SEQ_LEN, EMBED_DIM)
    ).astype(np.float32) * 0.01

    # Warmup
    print(f"  Warmup ({WARMUP_SEQUENCES} sequence)...")
    for _ in range(WARMUP_SEQUENCES):
        request.reset_state()
        token_id, _ = run_prefill(request, dummy_embeds)
        past_len = PREFILL_SEQ_LEN
        for _ in range(DECODE_TOKENS):
            token_id, _ = run_decode_step(request, embed_table, token_id, past_len)
            past_len += 1
    print("    Done.")

    # Benchmark
    print(f"  Running {NUM_SEQUENCES} sequences "
          f"(prefill {PREFILL_SEQ_LEN} tokens + decode {DECODE_TOKENS} tokens)...")

    prefill_times = []
    decode_times = []  # per-token decode times

    for seq_i in range(NUM_SEQUENCES):
        request.reset_state()

        # Prefill
        token_id, prefill_t = run_prefill(request, dummy_embeds)
        prefill_times.append(prefill_t)

        # Decode loop
        past_len = PREFILL_SEQ_LEN
        seq_decode_times = []
        for _ in range(DECODE_TOKENS):
            token_id, decode_t = run_decode_step(
                request, embed_table, token_id, past_len
            )
            seq_decode_times.append(decode_t)
            past_len += 1

        decode_times.extend(seq_decode_times)

        avg_decode = np.mean(seq_decode_times) * 1000
        print(f"    Seq {seq_i + 1}/{NUM_SEQUENCES}: "
              f"prefill={prefill_t * 1000:.1f}ms, "
              f"avg decode={avg_decode:.1f}ms/tok")

    # Compute statistics
    prefill_arr = np.array(prefill_times) * 1000  # ms
    decode_arr = np.array(decode_times) * 1000    # ms

    total_tokens = NUM_SEQUENCES * (1 + DECODE_TOKENS)  # +1 for prefill output
    total_time = sum(prefill_times) + sum(decode_times)
    tok_per_sec = total_tokens / total_time

    results = {
        "label": f"{label} [{device_desc}]",
        "device": device_desc,
        "prefill_mean_ms": float(np.mean(prefill_arr)),
        "prefill_std_ms": float(np.std(prefill_arr)),
        "prefill_min_ms": float(np.min(prefill_arr)),
        "prefill_max_ms": float(np.max(prefill_arr)),
        "decode_mean_ms": float(np.mean(decode_arr)),
        "decode_std_ms": float(np.std(decode_arr)),
        "decode_min_ms": float(np.min(decode_arr)),
        "decode_max_ms": float(np.max(decode_arr)),
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "tok_per_sec": tok_per_sec,
    }

    print()
    print(f"  Results for {results['label']}:")
    print(f"    Prefill (first token): {results['prefill_mean_ms']:.1f} +/- "
          f"{results['prefill_std_ms']:.1f} ms  "
          f"[min={results['prefill_min_ms']:.1f}, max={results['prefill_max_ms']:.1f}]")
    print(f"    Decode  (per token):   {results['decode_mean_ms']:.1f} +/- "
          f"{results['decode_std_ms']:.1f} ms  "
          f"[min={results['decode_min_ms']:.1f}, max={results['decode_max_ms']:.1f}]")
    print(f"    Throughput:            {results['tok_per_sec']:.1f} tok/s "
          f"({results['total_tokens']} tokens in {results['total_time_s']:.2f}s)")

    return results


def print_comparison(results: list[dict]):
    """Print a comparison table of all benchmarked models."""
    print()
    print("=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print()

    # Header
    col_width = max(len(r["label"]) for r in results) + 2
    col_width = max(col_width, 16)
    header = f"{'Metric':<30}"
    for r in results:
        header += f"  {r['label']:>{col_width}}"
    print(header)
    print("-" * (30 + (col_width + 2) * len(results)))

    # Rows
    metrics = [
        ("Prefill mean (ms)", "prefill_mean_ms", ".1f"),
        ("Prefill std (ms)", "prefill_std_ms", ".1f"),
        ("Decode mean (ms/tok)", "decode_mean_ms", ".1f"),
        ("Decode std (ms/tok)", "decode_std_ms", ".1f"),
        ("Decode min (ms/tok)", "decode_min_ms", ".1f"),
        ("Decode max (ms/tok)", "decode_max_ms", ".1f"),
        ("Throughput (tok/s)", "tok_per_sec", ".1f"),
        ("Total time (s)", "total_time_s", ".2f"),
    ]

    for metric_name, key, fmt in metrics:
        row = f"{metric_name:<30}"
        for r in results:
            row += f"  {r[key]:>{col_width}{fmt}}"
        print(row)

    # Speedup (if exactly 2 results)
    if len(results) == 2:
        r0, r1 = results[0], results[1]
        print()
        print("-" * (30 + (col_width + 2) * len(results)))

        # Show which is faster
        prefill_ratio = r0["prefill_mean_ms"] / r1["prefill_mean_ms"]
        decode_ratio = r0["decode_mean_ms"] / r1["decode_mean_ms"]
        throughput_ratio = r1["tok_per_sec"] / r0["tok_per_sec"]

        lbl0 = results[0]["label"]
        lbl1 = results[1]["label"]
        print(f"  Prefill ratio ({lbl1} vs {lbl0}): {prefill_ratio:.2f}x")
        print(f"  Decode ratio  ({lbl1} vs {lbl0}): {decode_ratio:.2f}x")
        print(f"  Throughput ratio: {throughput_ratio:.2f}x")

        # Note if different devices were used
        if results[0]["device"] != results[1]["device"]:
            print()
            print(f"  NOTE: Models ran on different devices!")
            print(f"    {lbl0} -> {results[0]['device']}")
            print(f"    {lbl1} -> {results[1]['device']}")
            print(f"    Comparison reflects device difference, not just quantization.")

    print()
    print("=" * 70)


def main():
    print()
    print("=" * 70)
    print("  ASR Decoder Benchmark: FP16 vs INT4 on NPU (NPUW_LLM)")
    print("=" * 70)
    print(f"  Prefill length: {PREFILL_SEQ_LEN} tokens")
    print(f"  Decode length:  {DECODE_TOKENS} tokens/sequence")
    print(f"  Sequences:      {NUM_SEQUENCES} (+ {WARMUP_SEQUENCES} warmup)")
    print(f"  Embed dim:      {EMBED_DIM}")
    print()

    core = ov.Core()

    # Show available devices
    devices = core.available_devices
    print(f"  Available devices: {devices}")
    if "NPU" in devices:
        npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
        print(f"  NPU: {npu_name}")
    print()

    all_results = []

    for label, paths in MODELS.items():
        xml_path = paths["xml"]
        embed_path = paths["embed"]

        if not xml_path.exists():
            print(f"  SKIP {label}: model not found at {xml_path}")
            continue

        if not embed_path.exists():
            print(f"  SKIP {label}: embedding table not found at {embed_path}")
            continue

        result = benchmark_model(core, label, xml_path, embed_path)
        if result is not None:
            all_results.append(result)

    if len(all_results) >= 2:
        print_comparison(all_results)
    elif len(all_results) == 1:
        print()
        print("  Only one model was benchmarked successfully.")
        print("  No side-by-side comparison available.")
    else:
        print()
        print("  ERROR: No models were benchmarked successfully.")


if __name__ == "__main__":
    main()
