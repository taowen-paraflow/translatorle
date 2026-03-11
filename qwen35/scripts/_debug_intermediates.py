"""Debug: inspect intermediate tensor values at step 0."""
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

max_cache_len = ov_model.input(key_inputs[0]).get_partial_shape()[2].get_length()

# Run step 0
token_ids = tokenizer("Hello", return_tensors="np")["input_ids"][0]
token_id = token_ids[0]
embed = embed_table[token_id:token_id+1][np.newaxis, :, :]

total_len = max_cache_len + 1
mask = np.full((1, 1, 1, total_len), -np.inf, dtype=np.float32)
mask[0, 0, 0, -1] = 0.0

pos = np.full((3, 1, 1), 0, dtype=np.int64)

inp = {"inputs_embeds": embed, "attention_mask": mask, "position_ids": pos}
for n in conv_inputs:
    shape = [d.get_length() for d in ov_model.input(n).get_partial_shape()]
    inp[n] = np.zeros(shape, dtype=np.float32)
for n in recurrent_inputs:
    shape = [d.get_length() for d in ov_model.input(n).get_partial_shape()]
    inp[n] = np.zeros(shape, dtype=np.float32)
for n in key_inputs + value_inputs:
    shape = [d.get_length() for d in ov_model.input(n).get_partial_shape()]
    inp[n] = np.zeros(shape, dtype=np.float32)

request.infer(inp)

# Check intermediates for layer 0
print("=== Layer 0 intermediates (step 0) ===")
g_t = request.get_tensor("gdn_intermediate.0.g_t").data.copy()
k_t = request.get_tensor("gdn_intermediate.0.k_t").data.copy()
v_t = request.get_tensor("gdn_intermediate.0.v_t").data.copy()
beta_t = request.get_tensor("gdn_intermediate.0.beta_t").data.copy()

print(f"g_t: shape={g_t.shape} dtype={g_t.dtype} range=[{g_t.min():.6f}, {g_t.max():.6f}]")
print(f"k_t: shape={k_t.shape} dtype={k_t.dtype} range=[{k_t.min():.6f}, {k_t.max():.6f}] norm={np.linalg.norm(k_t):.6f}")
print(f"v_t: shape={v_t.shape} dtype={v_t.dtype} range=[{v_t.min():.6f}, {v_t.max():.6f}] norm={np.linalg.norm(v_t):.6f}")
print(f"beta_t: shape={beta_t.shape} dtype={beta_t.dtype} range=[{beta_t.min():.6f}, {beta_t.max():.6f}]")

# Compute shadow state from intermediates
S_shadow = np.zeros((1, 16, 128, 128), dtype=np.float32)
S_shadow = S_shadow * g_t.astype(np.float32)
mem = np.einsum('bhkv,bhk->bhv', S_shadow, k_t.astype(np.float32))
delta = (v_t.astype(np.float32) - mem) * beta_t.astype(np.float32)
S_shadow = S_shadow + np.einsum('bhk,bhv->bhkv', k_t.astype(np.float32), delta)

print(f"\nShadow S: range=[{S_shadow.min():.6f}, {S_shadow.max():.6f}]")

# Direct model output
S_direct = request.get_tensor("cache_params.present.recurrent.0").data.copy()
print(f"Direct S: range=[{S_direct.min():.6f}, {S_direct.max():.6f}]")

print(f"\nMax diff: {np.abs(S_shadow - S_direct).max():.6f}")

# The issue: k_t * delta should give the same result
# Let's manually compute what the model does internally:
# k_t outer delta = k_t.unsqueeze(-1) * delta.unsqueeze(-2)  in PyTorch
# equivalent to einsum('bhk,bhv->bhkv', k_t, delta) in NumPy
# Let's verify the shapes and values
print(f"\ndelta: range=[{delta.min():.6f}, {delta.max():.6f}]")
outer = np.einsum('bhk,bhv->bhkv', k_t.astype(np.float32), delta)
print(f"k_t outer delta: range=[{outer.min():.6f}, {outer.max():.6f}]")

# Let's check: is S_direct = k_t outer delta? (since initial state was zeros)
print(f"\nExpected (S_direct should equal k_t outer delta): max diff = {np.abs(S_direct - outer).max():.6f}")

# Check if k_t is L2-normalized (should be, since use_qk_l2norm_in_kernel=True)
k_t_norms = np.linalg.norm(k_t[0], axis=-1)  # per-head norms
print(f"\nk_t per-head norms (should be ~1.0): {k_t_norms[:4]}")
v_t_norms = np.linalg.norm(v_t[0], axis=-1)
print(f"v_t per-head norms: {v_t_norms[:4]}")

# Check layer 3 (worst layer from previous test)
print("\n=== Layer 3 intermediates (worst from comparison) ===")
g3 = request.get_tensor("gdn_intermediate.3.g_t").data.copy()
k3 = request.get_tensor("gdn_intermediate.3.k_t").data.copy()
v3 = request.get_tensor("gdn_intermediate.3.v_t").data.copy()
b3 = request.get_tensor("gdn_intermediate.3.beta_t").data.copy()
print(f"g_t: range=[{g3.min():.6f}, {g3.max():.6f}]")
print(f"k_t: range=[{k3.min():.6f}, {k3.max():.6f}] norm={np.linalg.norm(k3):.6f}")
print(f"v_t: range=[{v3.min():.6f}, {v3.max():.6f}] norm={np.linalg.norm(v3):.6f}")
print(f"beta_t: range=[{b3.min():.6f}, {b3.max():.6f}]")

S3_shadow = np.zeros((1, 16, 128, 128), dtype=np.float32)
delta3 = v3.astype(np.float32) * b3.astype(np.float32)
S3_shadow = np.einsum('bhk,bhv->bhkv', k3.astype(np.float32), delta3)
S3_direct = request.get_tensor("cache_params.present.recurrent.3").data.copy()
print(f"Shadow S3: range=[{S3_shadow.min():.6f}, {S3_shadow.max():.6f}]")
print(f"Direct S3: range=[{S3_direct.min():.6f}, {S3_direct.max():.6f}]")
print(f"Max diff: {np.abs(S3_shadow - S3_direct).max():.6f}")
