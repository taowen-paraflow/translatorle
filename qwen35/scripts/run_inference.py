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
    parser = argparse.ArgumentParser(description="Run Qwen3.5 OpenVINO inference")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to exported OpenVINO model directory (auto-detected from --device)",
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
        help="OpenVINO device (CPU, GPU, NPU, NPUW, HYBRID, MULTISUB, NPUV2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (enables sampling when set)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold",
    )
    args = parser.parse_args()

    # Auto-detect model path and config based on device
    ov_config = None
    actual_device = args.device

    if args.device == "NPUW":
        if args.model_path is None:
            args.model_path = str(MODELS_DIR / "Qwen3.5-0.8B-npuw")
        actual_device = "NPU"
        from qwen35.config import NPUW_LLM_OV_CONFIG
        ov_config = NPUW_LLM_OV_CONFIG
    elif args.device == "HYBRID":
        if args.model_path is None:
            args.model_path = str(MODELS_DIR / "Qwen3.5-0.8B-hybrid")
        actual_device = "NPU"
        from qwen35.config import HYBRID_OV_CONFIG
        ov_config = HYBRID_OV_CONFIG
    elif args.device == "MULTISUB":
        if args.model_path is None:
            args.model_path = str(MODELS_DIR / "Qwen3.5-0.8B-multisub")
        actual_device = "NPU"
        from qwen35.config import MULTISUB_OV_CONFIG
        ov_config = MULTISUB_OV_CONFIG
    elif args.device in ("NPUV2", "NPUV2_CPU"):
        if args.model_path is None:
            args.model_path = str(MODELS_DIR / "Qwen3.5-0.8B-npu-v2")
        if args.device == "NPUV2_CPU":
            actual_device = "CPU"
        else:
            actual_device = "NPU"
            from qwen35.config import NPU_V2_OV_CONFIG
            ov_config = NPU_V2_OV_CONFIG
    elif args.device == "NPU":
        if args.model_path is None:
            args.model_path = str(MODELS_DIR / "Qwen3.5-0.8B-npu")
    else:
        if args.model_path is None:
            args.model_path = str(MODELS_DIR / "Qwen3.5-0.8B-ov")

    if args.device in ("NPUV2", "NPUV2_CPU"):
        from qwen35.inference_npu_v2 import Qwen35NPUv2Model
        print(f"Loading NPU v2 model from {args.model_path} on {actual_device}...")
        if ov_config:
            print(f"  OV config: {ov_config}")
        model = Qwen35NPUv2Model.from_pretrained(
            args.model_path, device=actual_device, ov_config=ov_config
        )
    elif args.device == "MULTISUB":
        from qwen35.inference import Qwen35MultiSubgraphModel
        print(f"Loading multi-subgraph model from {args.model_path} on {actual_device}...")
        if ov_config:
            print(f"  OV config: {ov_config}")
        model = Qwen35MultiSubgraphModel.from_pretrained(
            args.model_path, device=actual_device, ov_config=ov_config
        )
    else:
        print(f"Loading model from {args.model_path} on {actual_device}...")
        if ov_config:
            print(f"  OV config: {ov_config}")
        model = Qwen35OVModel.from_pretrained(
            args.model_path, device=actual_device, ov_config=ov_config
        )
    print("Model loaded.")

    inputs = model.tokenizer(args.prompt, return_tensors="pt")
    num_input_tokens = inputs["input_ids"].shape[1]

    print(f"\nPrompt: {args.prompt!r}")
    start = time.time()
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.temperature is not None:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generate_kwargs["top_p"] = args.top_p
    else:
        generate_kwargs["do_sample"] = False
    outputs = model.generate(
        **inputs,
        **generate_kwargs,
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
