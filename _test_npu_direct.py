"""Quick test: try Direct NPU compile + inference on LL2 model."""
import time
import numpy as np
import openvino as ov

print(f"OpenVINO: {ov.__version__}")
core = ov.Core()
print(f"Devices: {core.available_devices}")

model = core.read_model("models/qwen35/Qwen3.5-0.8B-ll2/openvino_model.xml")
print(f"Inputs: {len(model.inputs)}, Outputs: {len(model.outputs)}")

# Check for boolean types
for op in model.get_ordered_ops():
    if op.get_type_name() == "ReadValue":
        etype = op.get_output_element_type(0)
        if "bool" in str(etype).lower():
            print(f"  BOOLEAN ReadValue: {op.get_friendly_name()} type={etype}")

for inp in model.inputs:
    etype = inp.get_element_type()
    if "bool" in str(etype).lower() or "u8" in str(etype).lower():
        print(f"  Input {inp.get_any_name()}: type={etype}")

# Try Direct NPU
print("\n[1] Direct NPU compile")
t0 = time.time()
try:
    compiled = core.compile_model(model, "NPU")
    print(f"  OK in {time.time() - t0:.1f}s")
except Exception as e:
    print(f"  FAILED: {str(e)[:500]}")
    exit(1)

# Try create_infer_request
print("\n[2] Create infer request")
try:
    request = compiled.create_infer_request()
    print("  OK")
except Exception as e:
    print(f"  FAILED: {str(e)[:500]}")
    exit(1)

# Try one inference step
print("\n[3] Single inference step")
embed_table = np.load("models/qwen35/Qwen3.5-0.8B-ll2/embed_tokens.npy").astype(np.float32)

# Token "Hello" = let's find it
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("models/qwen35/Qwen3.5-0.8B-ll2", trust_remote_code=True)
token_id = tokenizer("Hello", return_tensors="np")["input_ids"][0, 0]
embeds = embed_table[token_id:token_id+1].reshape(1, 1, -1)

KV_CACHE_LEN = 2048
inp = {"inputs_embeds": embeds}

# Build mask
mask = np.zeros((1, KV_CACHE_LEN + 1), dtype=np.int64)
mask[0, -1] = 1
inp["attention_mask"] = mask
inp["position_ids"] = np.zeros((3, 1, 1), dtype=np.int64)

# Add all other inputs
for model_inp in model.inputs:
    name = model_inp.get_any_name()
    if name in inp:
        continue
    ps = model_inp.get_partial_shape()
    shape = [d.get_length() for d in ps]
    etype = model_inp.get_element_type()
    if "i32" in str(etype):
        inp[name] = np.zeros(shape, dtype=np.int32)
    elif "i64" in str(etype):
        inp[name] = np.zeros(shape, dtype=np.int64)
    else:
        inp[name] = np.zeros(shape, dtype=np.float32)

print(f"  Input count: {len(inp)}")
t0 = time.time()
try:
    result = request.infer(inp)
    elapsed = time.time() - t0
    logits = None
    for out in compiled.outputs:
        if "logits" in out.get_any_name():
            logits = result[out]
            break
    if logits is not None:
        logits_1d = logits[0, -1, :]
        top_token = int(np.argmax(logits_1d))
        print(f"  OK in {elapsed:.2f}s, top_token={top_token} '{tokenizer.decode([top_token])}' "
              f"NaN={np.any(np.isnan(logits_1d))} logits=[{np.min(logits_1d):.1f}, {np.max(logits_1d):.1f}]")
    else:
        print(f"  OK but no logits found")
except Exception as e:
    print(f"  FAILED: {str(e)[:500]}")
