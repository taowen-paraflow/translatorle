#!/usr/bin/env python3
"""Run Qwen3.5 inference on an exported OpenVINO model.

Usage:
    uv run python -m qwen35.scripts.run_inference [--model-path PATH] [--prompt TEXT]
"""

import argparse
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.config import MODELS_DIR
from qwen35.inference import Qwen35OVModel


def main():
    default_model = str(MODELS_DIR / "Qwen3.5-0.8B-ov")

    parser = argparse.ArgumentParser(description="Run Qwen3.5 OpenVINO inference")
    parser.add_argument(
        "--model-path",
        default=default_model,
        help="Path to exported OpenVINO model directory",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=30,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        help="OpenVINO device (CPU, GPU, etc.)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} on {args.device}...")
    model = Qwen35OVModel.from_pretrained(args.model_path, device=args.device)
    print("Model loaded.")

    inputs = model.tokenizer(args.prompt, return_tensors="pt")
    num_input_tokens = inputs["input_ids"].shape[1]

    print(f"\nPrompt: {args.prompt!r}")
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    elapsed = time.time() - start

    generated = outputs[0][num_input_tokens:]
    num_new = len(generated)
    tok_per_sec = num_new / elapsed if elapsed > 0 else 0

    result = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {result}")
    print(f"  ({num_new} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s)")


if __name__ == "__main__":
    main()
