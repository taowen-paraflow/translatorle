#!/usr/bin/env python3
"""Run Qwen3.5 inference on an exported OpenVINO model.

Usage:
    uv run python -m qwen35.scripts.run_inference [--model-path PATH] [--prompt TEXT]
"""

import argparse
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from qwen35.config import MODELS_DIR, ARCH_CONFIGS
from qwen35.inference import Qwen35OVModel


def _parse_response(text: str) -> tuple[str, str]:
    """Extract thinking and response from model output.

    Returns (thinking, response) where thinking may be empty.
    """
    # Match <think>...</think> block
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        response = text[m.end():].strip()
    else:
        thinking = ""
        response = text.strip()
    return thinking, response


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3.5 OpenVINO inference")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to exported OpenVINO model directory (auto-detected from --model-size)",
    )
    parser.add_argument(
        "--model-size",
        default="0.8B",
        choices=list(ARCH_CONFIGS.keys()),
        help="Model size (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Input prompt",
    )
    parser.add_argument(
        "--chatml",
        action="store_true",
        help="Wrap prompt in ChatML template using tokenizer.apply_chat_template",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable thinking mode (only with --chatml)",
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
        help="OpenVINO device (CPU, GPU)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides ChatML defaults)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold (overrides ChatML defaults)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (overrides ChatML defaults)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Force greedy decoding (not recommended for ChatML)",
    )
    args = parser.parse_args()

    # Auto-detect model path based on model size
    if args.model_path is None:
        ov_dir_name = ARCH_CONFIGS[args.model_size]["ov_dir_name"]
        args.model_path = str(MODELS_DIR / ov_dir_name)

    print(f"Loading model from {args.model_path} on {args.device}...")
    model = Qwen35OVModel.from_pretrained(args.model_path, device=args.device)
    print("Model loaded.")

    # --- Build input tokens ---
    if args.chatml:
        messages = [{"role": "user", "content": args.prompt}]
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if args.no_think:
            template_kwargs["enable_thinking"] = False
        chat_text = model.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = model.tokenizer(chat_text, return_tensors="pt")
    else:
        inputs = model.tokenizer(args.prompt, return_tensors="pt")
    num_input_tokens = inputs["input_ids"].shape[1]

    # --- Build generation kwargs ---
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens)

    # Stop tokens: always include both <|endoftext|> and <|im_end|>
    endoftext_id = 248044  # <|endoftext|>
    im_end_id = model.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    generate_kwargs["eos_token_id"] = list(set([endoftext_id] + im_end_id))

    if args.greedy:
        # Explicit greedy override
        generate_kwargs["do_sample"] = False
    elif args.chatml and args.temperature is None:
        # ChatML mode: use Qwen3.5 recommended sampling parameters
        if args.no_think:
            # Non-thinking: temperature=0.7, top_p=0.8, top_k=20
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = 0.7
            generate_kwargs["top_p"] = 0.8
            generate_kwargs["top_k"] = 20
        else:
            # Thinking: temperature=0.6, top_p=0.95, top_k=20
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = 0.6
            generate_kwargs["top_p"] = 0.95
            generate_kwargs["top_k"] = 20
        generate_kwargs["repetition_penalty"] = 1.05
    elif args.temperature is not None:
        # Manual sampling parameters
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p or 0.95
        generate_kwargs["top_k"] = args.top_k or 20
        generate_kwargs["repetition_penalty"] = 1.05
    else:
        # Raw text mode, no ChatML: greedy by default
        generate_kwargs["do_sample"] = False

    # Allow explicit overrides
    if args.top_p is not None and "top_p" not in generate_kwargs:
        generate_kwargs["top_p"] = args.top_p
    if args.top_k is not None and "top_k" not in generate_kwargs:
        generate_kwargs["top_k"] = args.top_k

    # --- Generate ---
    sampling_desc = "greedy" if not generate_kwargs.get("do_sample") else (
        f"T={generate_kwargs.get('temperature')}, "
        f"top_p={generate_kwargs.get('top_p')}, "
        f"top_k={generate_kwargs.get('top_k')}"
    )
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Sampling: {sampling_desc}")

    start = time.time()
    outputs = model.generate(**inputs, **generate_kwargs)
    elapsed = time.time() - start

    # --- Decode output ---
    generated_ids = outputs[0][num_input_tokens:]
    num_new = len(generated_ids)
    tok_per_sec = num_new / elapsed if elapsed > 0 else 0

    generated_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

    if args.chatml:
        thinking, response = _parse_response(generated_text)
        if thinking:
            print(f"\n<think>\n{thinking}\n</think>\n")
        print(f"Response: {response}")
    else:
        print(f"Output: {generated_text}")

    print(f"  ({num_new} tokens in {elapsed:.1f}s = {tok_per_sec:.1f} tok/s)")


if __name__ == "__main__":
    main()
