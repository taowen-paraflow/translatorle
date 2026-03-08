#!/usr/bin/env python
"""Benchmark script for ASR and HY-MT engines across CPU/GPU/NPU devices.

Usage:
    python benchmark.py                          # benchmark all components on all devices
    python benchmark.py --component mt           # MT only
    python benchmark.py --component asr-decoder  # ASR decoder only
    python benchmark.py --component asr-encoder  # ASR encoder only
    python benchmark.py --device NPU             # NPU only
    python benchmark.py --component mt --device CPU
"""

import sys

sys.path.insert(0, ".")

import argparse
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DEVICES = ["CPU", "GPU", "NPU"]

# MT benchmark
MT_INPUT_TEXT = "今天天气真好，我们一起去公园散步吧。春天的花开得特别美丽。"
MT_TARGET_LANG = "English"
MT_ITERATIONS = 3

# ASR Decoder benchmark
DECODER_SEQ_LEN = 128  # prefill sequence length (tokens)
DECODER_HIDDEN_DIM = 1024
DECODER_MAX_NEW_TOKENS = 20
DECODER_ITERATIONS = 3

# ASR Encoder benchmark
ENCODER_MEL_BINS = 128
ENCODER_MEL_FRAMES = 800  # fixed 5-second input (T=800)
ENCODER_ITERATIONS = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_time(seconds: float) -> str:
    """Format a duration in human-readable form."""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple aligned ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(cells):
        return " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))
    print()


def print_banner(title: str, details: list[str] | None = None) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    if details:
        for d in details:
            print(f"  {d}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# MT Benchmark
# ---------------------------------------------------------------------------


def bench_mt(devices: list[str]) -> None:
    print_banner(
        "HY-MT Translation Benchmark",
        [
            f"Input:      {MT_INPUT_TEXT}",
            f"Target:     {MT_TARGET_LANG}",
            f"Iterations: {MT_ITERATIONS} (+ 1 warmup)",
        ],
    )

    from hymt.engine import MTEngine

    rows: list[list[str]] = []

    for device in devices:
        print(f"\n--- {device} ---")
        try:
            # Load
            t0 = time.perf_counter()
            engine = MTEngine(device=device)
            load_time = time.perf_counter() - t0
            print(f"  Load time: {fmt_time(load_time)}")

            # Warmup (first generate compiles/caches on NPU)
            warmup_t0 = time.perf_counter()
            _ = engine.translate(MT_INPUT_TEXT, target_lang=MT_TARGET_LANG)
            warmup_time = time.perf_counter() - warmup_t0
            print(f"  Warmup:    {fmt_time(warmup_time)}")

            # Timed runs
            times: list[float] = []
            sample_output = ""
            for i in range(MT_ITERATIONS):
                t0 = time.perf_counter()
                result = engine.translate(MT_INPUT_TEXT, target_lang=MT_TARGET_LANG)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                if i == 0:
                    sample_output = result
                preview = result[:60] + ("..." if len(result) > 60 else "")
                print(f"  Run {i + 1}:    {fmt_time(elapsed)}  ->  {preview}")

            avg_t = sum(times) / len(times)
            out_preview = (
                sample_output[:40] + "..."
                if len(sample_output) > 40
                else sample_output
            )
            rows.append(
                [
                    device,
                    fmt_time(load_time),
                    fmt_time(warmup_time),
                    fmt_time(avg_t),
                    fmt_time(min(times)),
                    fmt_time(max(times)),
                    out_preview,
                ]
            )

            del engine

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            rows.append([device, "ERROR", "", str(e)[:50], "", "", ""])

    print()
    print_table(
        [
            "Device",
            "Load",
            "Warmup",
            "Avg",
            "Min",
            "Max",
            "Sample Output",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# ASR Decoder Benchmark
# ---------------------------------------------------------------------------


def bench_asr_decoder(devices: list[str]) -> None:
    print_banner(
        "ASR Decoder Benchmark",
        [
            f"Prefill seq_len: {DECODER_SEQ_LEN}",
            f"Decode tokens:   {DECODER_MAX_NEW_TOKENS}",
            f"Hidden dim:      {DECODER_HIDDEN_DIM}",
            f"Iterations:      {DECODER_ITERATIONS} (+ 1 warmup)",
        ],
    )

    from asr.ov_decoder import OVDecoder

    rows: list[list[str]] = []

    for device in devices:
        print(f"\n--- {device} ---")
        try:
            # Load and compile
            t0 = time.perf_counter()
            decoder = OVDecoder(device=device)
            load_time = time.perf_counter() - t0
            print(f"  Load/compile: {fmt_time(load_time)}")

            # Dummy inputs_embeds [seq_len, 1024] (no batch dim -- OVDecoder adds it)
            dummy_embeds = (
                np.random.randn(DECODER_SEQ_LEN, DECODER_HIDDEN_DIM).astype(
                    np.float32
                )
                * 0.01
            )

            # Warmup
            decoder.generate(dummy_embeds, max_new_tokens=5)

            # Timed runs
            prefill_times: list[float] = []
            decode_times_per_tok: list[float] = []

            for i in range(DECODER_ITERATIONS):
                decoder.reset()

                # Prefill
                t0 = time.perf_counter()
                first_token = decoder.prefill(dummy_embeds)
                prefill_t = time.perf_counter() - t0
                prefill_times.append(prefill_t)

                # Decode steps
                token_id = first_token
                step_times: list[float] = []
                for _ in range(DECODER_MAX_NEW_TOKENS):
                    t0 = time.perf_counter()
                    token_id = decoder.decode_step(token_id)
                    step_times.append(time.perf_counter() - t0)

                avg_step = sum(step_times) / len(step_times)
                decode_times_per_tok.append(avg_step)
                tok_s = 1.0 / avg_step if avg_step > 0 else 0
                print(
                    f"  Run {i + 1}:    prefill={fmt_time(prefill_t)}, "
                    f"decode={avg_step * 1000:.1f} ms/tok, "
                    f"{tok_s:.1f} tok/s"
                )

            avg_prefill = sum(prefill_times) / len(prefill_times)
            avg_decode = sum(decode_times_per_tok) / len(decode_times_per_tok)
            tok_s = 1.0 / avg_decode if avg_decode > 0 else 0

            rows.append(
                [
                    device,
                    fmt_time(load_time),
                    fmt_time(avg_prefill),
                    f"{avg_decode * 1000:.1f} ms",
                    f"{tok_s:.1f}",
                ]
            )

            del decoder

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            rows.append([device, "ERROR", str(e)[:50], "", ""])

    print()
    print_table(
        ["Device", "Load Time", "Avg Prefill", "Avg Decode/tok", "Tok/s"],
        rows,
    )


# ---------------------------------------------------------------------------
# ASR Encoder Benchmark
# ---------------------------------------------------------------------------


def bench_asr_encoder(devices: list[str]) -> None:
    print_banner(
        "ASR Encoder Benchmark",
        [
            f"Input shape: [1, {ENCODER_MEL_BINS}, {ENCODER_MEL_FRAMES}]",
            f"Iterations:  {ENCODER_ITERATIONS} (+ 1 warmup)",
        ],
    )

    from asr.ov_encoder import OVEncoder

    rows: list[list[str]] = []

    for device in devices:
        print(f"\n--- {device} ---")
        try:
            # Load and compile
            t0 = time.perf_counter()
            encoder = OVEncoder(device=device)
            load_time = time.perf_counter() - t0
            print(f"  Load/compile: {fmt_time(load_time)}")

            # Dummy mel input [1, 128, 800]
            dummy_mel = np.random.randn(
                1, ENCODER_MEL_BINS, ENCODER_MEL_FRAMES
            ).astype(np.float32)

            # Warmup
            _ = encoder(dummy_mel)

            # Timed runs
            times: list[float] = []
            output_shape = ""
            for i in range(ENCODER_ITERATIONS):
                t0 = time.perf_counter()
                output = encoder(dummy_mel)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                output_shape = str(output.shape)
                print(f"  Run {i + 1}:    {fmt_time(elapsed)}  output={output_shape}")

            avg_t = sum(times) / len(times)
            rows.append(
                [
                    device,
                    fmt_time(load_time),
                    fmt_time(avg_t),
                    fmt_time(min(times)),
                    fmt_time(max(times)),
                    output_shape,
                ]
            )

            del encoder

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            rows.append([device, "ERROR", str(e)[:50], "", "", ""])

    print()
    print_table(
        ["Device", "Load Time", "Avg Infer", "Min Infer", "Max Infer", "Output Shape"],
        rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMPONENTS = {
    "mt": bench_mt,
    "asr-decoder": bench_asr_decoder,
    "asr-encoder": bench_asr_encoder,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ASR and MT engines across devices"
    )
    parser.add_argument(
        "--component",
        choices=list(COMPONENTS.keys()) + ["all"],
        default="all",
        help="Component to benchmark (default: all)",
    )
    parser.add_argument(
        "--device",
        choices=ALL_DEVICES + ["all"],
        default="all",
        help="Device to test (default: all)",
    )
    args = parser.parse_args()

    devices = ALL_DEVICES if args.device == "all" else [args.device]

    if args.component == "all":
        bench_fns = list(COMPONENTS.values())
    else:
        bench_fns = [COMPONENTS[args.component]]

    print(f"Devices: {devices}")
    print(f"Components: {args.component}")

    for fn in bench_fns:
        fn(devices)

    print("Done.")


if __name__ == "__main__":
    main()
