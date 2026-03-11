"""Test NPU compilation and inference of static-cache model."""
import logging, time, sys
logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(0, ".")
from qwen35.inference import Qwen35OVModel

print("Loading on NPU...", flush=True)
t0 = time.time()
try:
    m = Qwen35OVModel.from_pretrained("models/qwen35/Qwen3.5-0.8B-npu", device="NPU")
    elapsed = time.time() - t0
    print(f"Compiled in {elapsed:.1f}s", flush=True)
    print(f"is_static_cache={m._is_static_cache}", flush=True)

    # Try inference
    print("\nRunning inference...", flush=True)
    t1 = time.time()
    outputs = m.generate(
        **m.tokenizer("Hello", return_tensors="pt"),
        max_new_tokens=10,
        do_sample=False,
    )
    elapsed2 = time.time() - t1
    text = m.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {text}", flush=True)
    num_tokens = outputs.shape[1] - 1  # subtract input
    print(f"  ({num_tokens} tokens in {elapsed2:.1f}s = {num_tokens/elapsed2:.1f} tok/s)", flush=True)

except Exception as e:
    elapsed = time.time() - t0
    print(f"FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
