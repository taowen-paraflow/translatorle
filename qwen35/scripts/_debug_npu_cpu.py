"""Debug: run standard NPU model on CPU to verify export format."""
import sys
sys.path.insert(0, ".")

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

model_path = "models/qwen35/Qwen3.5-0.8B-npu"

core = ov.Core()
ov_model = core.read_model(f"{model_path}/openvino_model.xml")
compiled = core.compile_model(ov_model, "CPU")
request = compiled.create_infer_request()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
embed_table = np.load(f"{model_path}/embed_tokens.npy")
if embed_table.dtype == np.float16:
    embed_table = embed_table.astype(np.float32)

# Collect input/output names
input_names = set()
for inp in ov_model.inputs:
    input_names.update(inp.get_names())
output_names = set()
for out in ov_model.outputs:
    output_names.update(out.get_names())

print("=== INPUTS ===")
for i, inp in enumerate(ov_model.inputs):
    print(f"  [{i}] {inp.get_names()} shape={inp.get_partial_shape()}")
print()

conv_inputs = sorted(n for n in input_names if "cache_params.past.conv" in n)
recurrent_inputs = sorted(n for n in input_names if "cache_params.past.recurrent" in n)
key_inputs = sorted(n for n in input_names if "cache_params.past.key" in n)
value_inputs = sorted(n for n in input_names if "cache_params.past.value" in n)

conv_outputs = sorted(n for n in output_names if "cache_params.present.conv" in n)
recurrent_outputs = sorted(n for n in output_names if "cache_params.present.recurrent" in n)
key_outputs = sorted(n for n in output_names if "cache_params.present.key" in n)
value_outputs = sorted(n for n in output_names if "cache_params.present.value" in n)

max_cache_len = ov_model.input(key_inputs[0]).get_partial_shape()[2].get_length()
print(f"max_cache_len={max_cache_len}")

# Init states
gdn_conv = {}
for name in conv_inputs:
    shape = [d.get_length() for d in ov_model.input(name).get_partial_shape()]
    gdn_conv[name] = np.zeros(shape, dtype=np.float32)

recurrent = {}
for name in recurrent_inputs:
    shape = [d.get_length() for d in ov_model.input(name).get_partial_shape()]
    recurrent[name] = np.zeros(shape, dtype=np.float32)

kv_buffers = {}
for name in key_inputs + value_inputs:
    shape = [d.get_length() for d in ov_model.input(name).get_partial_shape()]
    kv_buffers[name] = np.zeros(shape, dtype=np.float32)

# Tokenize
prompt = "Hello"
token_ids = tokenizer(prompt, return_tensors="np")["input_ids"][0]
print(f"Prompt: '{prompt}' -> tokens: {token_ids}")

past_length = 0
generated_tokens = []

for step in range(len(token_ids) + 30):
    if step < len(token_ids):
        token_id = token_ids[step]
    else:
        token_id = generated_tokens[-1]

    embed = embed_table[token_id:token_id+1][np.newaxis, :, :]

    # Build 4D mask
    total_len = max_cache_len + 1
    mask = np.full((1, 1, 1, total_len), -np.inf, dtype=np.float32)
    t = min(past_length, max_cache_len)
    if t > 0:
        mask[0, 0, 0, :t] = 0.0
    mask[0, 0, 0, -1] = 0.0

    pos = np.full((3, 1, 1), min(past_length, max_cache_len - 1), dtype=np.int64)

    inp = {
        "inputs_embeds": embed,
        "attention_mask": mask,
        "position_ids": pos,
    }
    for name in conv_inputs:
        inp[name] = gdn_conv[name]
    for name in recurrent_inputs:
        inp[name] = recurrent[name]
    for name in key_inputs + value_inputs:
        inp[name] = kv_buffers[name]

    request.infer(inp)

    # Read conv states
    for past_n, present_n in zip(conv_inputs, conv_outputs):
        gdn_conv[past_n] = request.get_tensor(present_n).data.copy()
    # Read recurrent states (directly from model output)
    for past_n, present_n in zip(recurrent_inputs, recurrent_outputs):
        recurrent[past_n] = request.get_tensor(present_n).data.copy()

    # Read KV outputs
    p = min(past_length, max_cache_len - 1)
    if past_length >= max_cache_len:
        for name in key_inputs + value_inputs:
            kv_buffers[name][:, :, :-1, :] = kv_buffers[name][:, :, 1:, :]
        p = max_cache_len - 1

    for past_n, present_n in zip(key_inputs, key_outputs):
        new_kv = request.get_tensor(present_n).data.copy()
        kv_buffers[past_n][:, :, p:p+1, :] = new_kv
    for past_n, present_n in zip(value_inputs, value_outputs):
        new_kv = request.get_tensor(present_n).data.copy()
        kv_buffers[past_n][:, :, p:p+1, :] = new_kv

    past_length += 1

    logits = request.get_tensor("logits").data.copy()
    next_token = int(np.argmax(logits[0, -1]))

    if step >= len(token_ids) - 1:
        generated_tokens.append(next_token)

    if step < 5:
        top5_idx = np.argsort(logits[0, -1])[-5:][::-1]
        top5_logits = logits[0, -1][top5_idx]
        print(f"  step {step}: token={next_token} ({tokenizer.decode([next_token])!r}) "
              f"top5_logits={top5_logits[:3]}")

text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"\nGenerated ({len(generated_tokens)} tokens): '{text}'")
