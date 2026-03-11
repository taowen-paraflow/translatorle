"""Compare hybrid model: CPU shadow state vs model's own state output."""
import sys
sys.path.insert(0, ".")

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

model_path = "models/qwen35/Qwen3.5-0.8B-hybrid"

core = ov.Core()
ov_model = core.read_model(f"{model_path}/openvino_model.xml")
compiled = core.compile_model(ov_model, "CPU")
request = compiled.create_infer_request()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
embed_table = np.load(f"{model_path}/embed_tokens.npy").astype(np.float32)

# Collect names
input_names = set()
for inp in ov_model.inputs:
    input_names.update(inp.get_names())
output_names = set()
for out in ov_model.outputs:
    output_names.update(out.get_names())

conv_inputs = sorted(n for n in input_names if "cache_params.past.conv" in n)
recurrent_inputs = sorted(n for n in input_names if "cache_params.past.recurrent" in n)
key_inputs = sorted(n for n in input_names if "cache_params.past.key" in n)
value_inputs = sorted(n for n in input_names if "cache_params.past.value" in n)
conv_outputs = sorted(n for n in output_names if "cache_params.present.conv" in n)
recurrent_outputs = sorted(n for n in output_names if "cache_params.present.recurrent" in n)
key_outputs = sorted(n for n in output_names if "cache_params.present.key" in n)
value_outputs = sorted(n for n in output_names if "cache_params.present.value" in n)

max_cache_len = ov_model.input(key_inputs[0]).get_partial_shape()[2].get_length()

def init_states():
    conv = {n: np.zeros([d.get_length() for d in ov_model.input(n).get_partial_shape()], dtype=np.float32) for n in conv_inputs}
    rec = {n: np.zeros([d.get_length() for d in ov_model.input(n).get_partial_shape()], dtype=np.float32) for n in recurrent_inputs}
    kv = {n: np.zeros([d.get_length() for d in ov_model.input(n).get_partial_shape()], dtype=np.float32) for n in key_inputs + value_inputs}
    return conv, rec, kv

def build_mask(past_length):
    total_len = max_cache_len + 1
    mask = np.full((1, 1, 1, total_len), -np.inf, dtype=np.float32)
    t = min(past_length, max_cache_len)
    if t > 0:
        mask[0, 0, 0, :t] = 0.0
    mask[0, 0, 0, -1] = 0.0
    return mask

def run_step(conv, rec, kv, token_id, past_length):
    embed = embed_table[token_id:token_id+1][np.newaxis, :, :]
    pos = np.full((3, 1, 1), min(past_length, max_cache_len - 1), dtype=np.int64)
    inp = {"inputs_embeds": embed, "attention_mask": build_mask(past_length), "position_ids": pos}
    for n in conv_inputs: inp[n] = conv[n]
    for n in recurrent_inputs: inp[n] = rec[n]
    for n in key_inputs + value_inputs: inp[n] = kv[n]

    request.infer(inp)

    # Read conv
    for past_n, present_n in zip(conv_inputs, conv_outputs):
        conv[past_n] = request.get_tensor(present_n).data.copy()

    # Read KV
    p = min(past_length, max_cache_len - 1)
    if past_length >= max_cache_len:
        for n in key_inputs + value_inputs: kv[n][:, :, :-1, :] = kv[n][:, :, 1:, :]
        p = max_cache_len - 1
    for past_n, present_n in zip(key_inputs, key_outputs):
        kv[past_n][:, :, p:p+1, :] = request.get_tensor(present_n).data.copy()
    for past_n, present_n in zip(value_inputs, value_outputs):
        kv[past_n][:, :, p:p+1, :] = request.get_tensor(present_n).data.copy()

    # Read recurrent state from model output (direct)
    direct_rec = {}
    for past_n, present_n in zip(recurrent_inputs, recurrent_outputs):
        direct_rec[past_n] = request.get_tensor(present_n).data.copy()

    # Compute shadow state from intermediates
    shadow_rec = {}
    for i, past_n in enumerate(recurrent_inputs):
        S = rec[past_n].copy()  # Previous state
        g_t = request.get_tensor(f"gdn_intermediate.{i}.g_t").data.copy().astype(np.float32)
        k_t = request.get_tensor(f"gdn_intermediate.{i}.k_t").data.copy().astype(np.float32)
        v_t = request.get_tensor(f"gdn_intermediate.{i}.v_t").data.copy().astype(np.float32)
        beta_t = request.get_tensor(f"gdn_intermediate.{i}.beta_t").data.copy().astype(np.float32)

        S = S * g_t
        mem = np.einsum('bhkv,bhk->bhv', S, k_t)
        delta = (v_t - mem) * beta_t
        S = S + np.einsum('bhk,bhv->bhkv', k_t, delta)
        shadow_rec[past_n] = S

    logits = request.get_tensor("logits").data.copy()
    return logits, direct_rec, shadow_rec

# Run comparison
token_ids = tokenizer("Hello", return_tensors="np")["input_ids"][0]
print(f"Tokens: {token_ids}")

# Mode A: Use model's direct state output
conv_a, rec_a, kv_a = init_states()
# Mode B: Use shadow FP32 state
conv_b, rec_b, kv_b = init_states()

for step in range(5):
    if step == 0:
        token = token_ids[0]
    elif step == 1:
        token = next_a
    else:
        token = next_a  # Follow mode A's path for comparison

    # Run with mode A's state
    logits_a, direct_a, shadow_a = run_step(conv_a, rec_a, kv_a, token, step)
    next_a = int(np.argmax(logits_a[0, -1]))

    # Now re-run with mode B's state (separate state)
    # But we need to re-infer... let's use separate request
    # Actually, let's just compare the states

    # Compare direct vs shadow for step 0
    max_diffs = []
    for name in recurrent_inputs:
        diff = np.abs(direct_a[name] - shadow_a[name]).max()
        max_diffs.append(diff)

    print(f"Step {step}: token={next_a} ({tokenizer.decode([next_a])!r})")
    print(f"  Logit: {logits_a[0, -1].max():.4f}")
    print(f"  Direct vs Shadow state max diff: min={min(max_diffs):.8f} max={max(max_diffs):.8f} mean={np.mean(max_diffs):.8f}")
    if max(max_diffs) > 0:
        worst_layer = np.argmax(max_diffs)
        d = direct_a[recurrent_inputs[worst_layer]]
        s = shadow_a[recurrent_inputs[worst_layer]]
        print(f"  Worst layer {worst_layer}: direct range [{d.min():.6f}, {d.max():.6f}] shadow range [{s.min():.6f}, {s.max():.6f}]")
        # Show a few values
        idx = np.unravel_index(np.argmax(np.abs(d - s)), d.shape)
        print(f"    At index {idx}: direct={d[idx]:.8f} shadow={s[idx]:.8f}")

    # Use direct state for next step (mode A)
    for name in recurrent_inputs:
        rec_a[name] = direct_a[name]
