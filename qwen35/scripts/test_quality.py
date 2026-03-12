#!/usr/bin/env python3
"""Test model output quality across multiple diverse prompts.

Runs Qwen3.5 hybrid inference on a curated set of prompts and reports
timing + output for each. Optionally compares two model directories
side by side.

Usage (root venv):
    powershell.exe -Command 'cd C:\\Apps\\translatorle; uv run python -m qwen35.scripts.test_quality'
    powershell.exe -Command 'cd C:\\Apps\\translatorle; uv run python -m qwen35.scripts.test_quality --device GPU_ONLY'
    powershell.exe -Command 'cd C:\\Apps\\translatorle; uv run python -m qwen35.scripts.test_quality --baseline-dir models/qwen35/Qwen3.5-0.8B-hybrid-v2'
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.inference_hybrid import Qwen35HybridModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test prompts: (name, prompt_text)
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    ("factual_en", "The capital of France is"),
    ("factual_cn", "中国的首都是"),
    ("reasoning", "If I have 3 apples and buy 5 more, then give away 2, how many do I have?"),
    ("code", "Write a Python function that checks if a number is prime:"),
    (
        "long_context",
        "Explain the difference between machine learning and deep learning in detail."
        " Cover the key concepts, advantages, and use cases of each approach.",
    ),
    ("translation", "Translate to English: 今天天气很好，我们去公园散步吧。"),
]


# ---------------------------------------------------------------------------
# Log-capture handler: extracts prefill/decode metrics from model logger
# ---------------------------------------------------------------------------

class _MetricsCapture(logging.Handler):
    """Captures prefill/decode timing logged by Qwen35HybridModel.generate()."""

    # Prefill: 44 tokens in 2930.7ms (15.0 tok/s, chunk=16)
    _PREFILL_RE = re.compile(
        r"Prefill:\s+(\d+)\s+tokens?\s+in\s+([\d.]+)ms\s+\(([\d.]+)\s+tok/s"
    )
    # Decode: 50 tokens in 3354.5ms (14.9 tok/s)
    _DECODE_RE = re.compile(
        r"Decode:\s+(\d+)\s+tokens?\s+in\s+([\d.]+)ms\s+\(([\d.]+)\s+tok/s"
    )

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.prefill_tokens: int = 0
        self.prefill_ms: float = 0.0
        self.prefill_tps: float = 0.0
        self.decode_tokens: int = 0
        self.decode_ms: float = 0.0
        self.decode_tps: float = 0.0

    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        m = self._PREFILL_RE.search(msg)
        if m:
            self.prefill_tokens = int(m.group(1))
            self.prefill_ms = float(m.group(2))
            self.prefill_tps = float(m.group(3))
        m = self._DECODE_RE.search(msg)
        if m:
            self.decode_tokens = int(m.group(1))
            self.decode_ms = float(m.group(2))
            self.decode_tps = float(m.group(3))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(
    model_dir: str,
    device: str,
    attn_past_seq: int,
    attn_stateful: bool,
    prefill_chunk_size: int,
) -> Qwen35HybridModel:
    """Construct a Qwen35HybridModel from CLI parameters."""
    device_map = {
        "HYBRID": ("GPU", "NPU", "GPU"),
        "GPU_ONLY": ("GPU", "GPU", "GPU"),
        "CPU_ONLY": ("CPU", "CPU", "CPU"),
    }
    gdn_dev, attn_dev, head_dev = device_map[device]
    logger.info(
        "Loading model from %s  (GDN=%s, Attn=%s, Head=%s)",
        model_dir, gdn_dev, attn_dev, head_dev,
    )
    return Qwen35HybridModel(
        model_dir=model_dir,
        gdn_device=gdn_dev,
        attn_device=attn_dev,
        head_device=head_dev,
        attn_past_seq=attn_past_seq,
        attn_stateful=attn_stateful,
        prefill_chunk_size=prefill_chunk_size,
    )


def _run_prompt(
    model: Qwen35HybridModel,
    prompt: str,
    max_tokens: int,
    capture: _MetricsCapture,
) -> tuple[str, dict]:
    """Generate from a single prompt and return (output_text, metrics_dict)."""
    capture.reset()
    t0 = time.time()
    output = model.generate(prompt, max_new_tokens=max_tokens)
    wall_ms = (time.time() - t0) * 1000

    metrics = {
        "prefill_tokens": capture.prefill_tokens,
        "prefill_tps": capture.prefill_tps,
        "decode_tokens": capture.decode_tokens,
        "decode_tps": capture.decode_tps,
        "wall_ms": wall_ms,
    }
    return output, metrics


def _print_separator(char: str = "=", width: int = 80):
    print(char * width)


def _print_output_block(label: str, prompt: str, output: str, metrics: dict):
    """Pretty-print one generation result."""
    print(f"  Prompt : {prompt}")
    print(f"  Output : {output}")
    print(
        f"  Timing : prefill {metrics['prefill_tps']:.1f} tok/s "
        f"({metrics['prefill_tokens']} tokens), "
        f"decode {metrics['decode_tps']:.1f} tok/s "
        f"({metrics['decode_tokens']} tokens), "
        f"wall {metrics['wall_ms']:.0f}ms"
    )


def _print_summary_table(
    results: list[tuple[str, dict]],
    baseline_results: Optional[list[tuple[str, dict]]] = None,
):
    """Print a summary table at the end of the run."""
    _print_separator()
    print("SUMMARY")
    _print_separator("-")

    has_baseline = baseline_results is not None

    if has_baseline:
        header = (
            f"{'Prompt':<16} | {'Prefill':>8} {'Decode':>8} {'Tokens':>6} "
            f"| {'B-Prefill':>9} {'B-Decode':>9} {'B-Tokens':>8}"
        )
        print(header)
        print("-" * len(header))
        for i, (name, metrics) in enumerate(results):
            b_name, b_metrics = baseline_results[i]
            print(
                f"{name:<16} | "
                f"{metrics['prefill_tps']:>7.1f}  {metrics['decode_tps']:>7.1f}  "
                f"{metrics['decode_tokens']:>5}  | "
                f"{b_metrics['prefill_tps']:>8.1f}  {b_metrics['decode_tps']:>8.1f}  "
                f"{b_metrics['decode_tokens']:>7}"
            )
    else:
        header = f"{'Prompt':<16} | {'Prefill tok/s':>14} {'Decode tok/s':>13} {'Output tokens':>14}"
        print(header)
        print("-" * len(header))
        for name, metrics in results:
            print(
                f"{name:<16} | "
                f"{metrics['prefill_tps']:>13.1f}  {metrics['decode_tps']:>12.1f}  "
                f"{metrics['decode_tokens']:>13}"
            )

    _print_separator()


def _flag_differences(
    results: list[tuple[str, str, dict]],
    baseline_results: list[tuple[str, str, dict]],
):
    """Flag prompt pairs where outputs differ significantly.

    Uses a simple heuristic: if the first 40 characters of the output
    diverge, flag it. This catches gross quality regressions without
    requiring a heavy similarity metric.
    """
    flagged = []
    for (name, out_a, _), (_, out_b, _) in zip(results, baseline_results):
        prefix_len = min(40, len(out_a), len(out_b))
        if out_a[:prefix_len] != out_b[:prefix_len]:
            flagged.append(name)

    if flagged:
        _print_separator()
        print(f"DIVERGENT OUTPUTS ({len(flagged)}/{len(results)}):")
        for name in flagged:
            print(f"  - {name}")
        _print_separator()
    else:
        print("All outputs match (first 40 chars identical).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Test Qwen3.5 hybrid model quality across diverse prompts",
    )
    parser.add_argument(
        "--model-dir",
        default="models/qwen35/Qwen3.5-0.8B-hybrid",
        help="Model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Optional baseline model directory for side-by-side comparison",
    )
    parser.add_argument(
        "--device",
        default="HYBRID",
        choices=["HYBRID", "GPU_ONLY", "CPU_ONLY"],
        help="Device config (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Max new tokens per generation (default: %(default)s)",
    )
    parser.add_argument(
        "--attn-past-seq",
        type=int,
        default=256,
        help="Static KV cache size for NPU (default: %(default)s)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=16,
        help="Prefill chunk size (default: %(default)s)",
    )
    args = parser.parse_args()

    # Determine stateful mode: HYBRID uses explicit I/O for NPU attention
    attn_stateful = args.device != "HYBRID"

    # Attach metrics capture handler to the inference_hybrid logger
    capture = _MetricsCapture()
    inference_logger = logging.getLogger("qwen35.inference_hybrid")
    inference_logger.addHandler(capture)

    # --- Load primary model ---
    model = _build_model(
        args.model_dir, args.device, args.attn_past_seq,
        attn_stateful, args.prefill_chunk_size,
    )

    # --- Load baseline model (if requested) ---
    baseline_model: Optional[Qwen35HybridModel] = None
    baseline_capture: Optional[_MetricsCapture] = None
    if args.baseline_dir:
        baseline_capture = _MetricsCapture()
        inference_logger.addHandler(baseline_capture)
        baseline_model = _build_model(
            args.baseline_dir, args.device, args.attn_past_seq,
            attn_stateful, args.prefill_chunk_size,
        )

    # --- Run all prompts ---
    _print_separator()
    print(f"Quality test: {len(TEST_PROMPTS)} prompts, max_tokens={args.max_tokens}")
    print(f"Model : {args.model_dir}")
    print(f"Device: {args.device} (stateful={'yes' if attn_stateful else 'no'})")
    if baseline_model:
        print(f"Baseline: {args.baseline_dir}")
    _print_separator()

    # Collect results: list of (name, output_text, metrics)
    all_results: list[tuple[str, str, dict]] = []
    all_baseline: list[tuple[str, str, dict]] = []

    for idx, (name, prompt) in enumerate(TEST_PROMPTS):
        print(f"\n[{idx + 1}/{len(TEST_PROMPTS)}] {name}")
        _print_separator("-", 60)

        # --- Primary model ---
        output, metrics = _run_prompt(model, prompt, args.max_tokens, capture)
        all_results.append((name, output, metrics))

        if baseline_model and baseline_capture:
            # --- Baseline model ---
            b_output, b_metrics = _run_prompt(
                baseline_model, prompt, args.max_tokens, baseline_capture,
            )
            all_baseline.append((name, b_output, b_metrics))

            print(f"  Prompt   : {prompt}")
            print(f"  [Model]  : {output}")
            print(
                f"             prefill {metrics['prefill_tps']:.1f} tok/s, "
                f"decode {metrics['decode_tps']:.1f} tok/s, "
                f"{metrics['decode_tokens']} tokens"
            )
            print(f"  [Baseline]: {b_output}")
            print(
                f"             prefill {b_metrics['prefill_tps']:.1f} tok/s, "
                f"decode {b_metrics['decode_tps']:.1f} tok/s, "
                f"{b_metrics['decode_tokens']} tokens"
            )
        else:
            _print_output_block(name, prompt, output, metrics)

    # --- Summary ---
    print()
    summary_results = [(name, m) for name, _, m in all_results]
    summary_baseline = (
        [(name, m) for name, _, m in all_baseline] if all_baseline else None
    )
    _print_summary_table(summary_results, summary_baseline)

    # --- Flag divergent outputs (baseline mode only) ---
    if all_baseline:
        _flag_differences(all_results, all_baseline)


if __name__ == "__main__":
    main()
