"""Benchmark Qwen3-0.6B on NPU/GPU/CPU using openvino_genai LLMPipeline.

Usage:
    uv run python -m qwen3.scripts.benchmark --device NPU
    uv run python -m qwen3.scripts.benchmark --device GPU
    uv run python -m qwen3.scripts.benchmark --device CPU
    uv run python -m qwen3.scripts.benchmark --device NPU --max-new-tokens 4096
    uv run python -m qwen3.scripts.benchmark --device NPU --max-new-tokens 4096 --ignore-eos
"""

import argparse
import time
from pathlib import Path

import openvino_genai as ov_genai

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "qwen3"
MODEL_DIR = str(MODELS_DIR / "Qwen3-0.6B-ov")

NPU_CONFIG_BASE = {
    "MAX_PROMPT_LEN": 512,
    "NPUW_LLM_PREFILL_CHUNK_SIZE": 512,
    "GENERATE_HINT": "BEST_PERF",
}


class TokenCounter:
    """Streamer callback that counts generated tokens and records first-token time."""

    def __init__(self):
        self.token_count = 0
        self.first_token_time = None
        self.start_time = None

    def __call__(self, subword):
        if self.token_count == 0:
            self.first_token_time = time.perf_counter()
        self.token_count += 1
        return False  # don't stop

    @property
    def ttft(self):
        """Time to first token in seconds."""
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-0.6B")
    parser.add_argument("--device", default="NPU", help="NPU, GPU, or CPU")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Tokens to generate")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Ignore EOS token to force generating exactly max-new-tokens")
    parser.add_argument("--prompt", default="Explain the theory of relativity in simple terms.",
                        help="Input prompt")
    args = parser.parse_args()

    if args.device == "NPU":
        config = {**NPU_CONFIG_BASE}
        # Set MIN_RESPONSE_LEN to ensure enough KV-cache for the requested generation
        config["MIN_RESPONSE_LEN"] = args.max_new_tokens
        cache_dir = str(MODELS_DIR / "qwen3_npu_cache")
        config["CACHE_DIR"] = cache_dir
    elif args.device in ("CPU", "GPU"):
        config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
    else:
        config = {}

    print(f"Loading Qwen3-0.6B on {args.device} ...")
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(MODEL_DIR, args.device, **config)
    load_time = time.perf_counter() - t0
    print(f"Loaded in {load_time:.1f}s")

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Ignore EOS: {args.ignore_eos}")
    print("-" * 60)

    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.do_sample = False
    if args.ignore_eos:
        gen_config.ignore_eos = True

    counter = TokenCounter()
    counter.start_time = time.perf_counter()
    result = pipe.generate(args.prompt, gen_config, counter)
    elapsed = time.perf_counter() - counter.start_time

    actual_tokens = counter.token_count
    output_chars = len(result)
    tok_per_sec = actual_tokens / elapsed if elapsed > 0 else 0

    # Decode-only throughput (excludes prefill/TTFT)
    decode_time = elapsed - (counter.ttft or 0)
    decode_tokens = max(actual_tokens - 1, 1)
    decode_tok_per_sec = decode_tokens / decode_time if decode_time > 0 else 0

    print(f"\nOutput ({output_chars} chars, {actual_tokens} tokens):")
    print(result[:500])
    if output_chars > 500:
        print(f"... ({output_chars - 500} more chars)")
    print("-" * 60)
    print(f"Tokens generated: {actual_tokens}")
    print(f"Total time:       {elapsed:.1f}s")
    if counter.ttft is not None:
        print(f"TTFT:             {counter.ttft * 1000:.0f}ms")
    print(f"Throughput:       {tok_per_sec:.1f} tok/s (total)")
    print(f"Decode speed:     {decode_tok_per_sec:.1f} tok/s (excl. prefill)")


if __name__ == "__main__":
    main()
