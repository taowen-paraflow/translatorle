"""Test NPUW model on CPU with token-by-token mode to verify IR correctness."""
import sys
import numpy as np
import openvino as ov
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from qwen35.config import NPUW_MODEL_DIR
from transformers import AutoTokenizer

# Load model
core = ov.Core()
model = core.read_model(str(NPUW_MODEL_DIR / "openvino_model.xml"))
compiled = core.compile_model(model, "CPU")
request = compiled.create_infer_request()

# Load tokenizer and embedding
tokenizer = AutoTokenizer.from_pretrained(str(NPUW_MODEL_DIR), trust_remote_code=True)
embed_table = np.load(str(NPUW_MODEL_DIR / "embed_tokens.npy"))

prompt = "Hello, what is your name?"
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"][0]
print(f"Prompt tokens: {input_ids.tolist()} ({len(input_ids)} tokens)")

# Initialize GDN states as zeros
conv_states = {}
recurrent_states = {}
for inp in model.inputs:
    name = inp.get_any_name()
    if "cache_params.past.conv" in name:
        shape = [1] + [d.get_length() for d in inp.get_partial_shape()[1:]]
        conv_states[name] = np.zeros(shape, dtype=np.float32)
    elif "cache_params.past.recurrent" in name:
        shape = [1] + [d.get_length() for d in inp.get_partial_shape()[1:]]
        recurrent_states[name] = np.zeros(shape, dtype=np.float32)

# Initialize KV cache as empty (0 seq_len)
kv_caches = {}
for inp in model.inputs:
    name = inp.get_any_name()
    if "past_key_values" in name:
        ps = inp.get_partial_shape()
        # [B, heads, 0, head_dim] - empty initial KV
        kv_caches[name] = np.zeros((1, ps[1].get_length(), 0, ps[3].get_length()), dtype=np.float32)

# Token-by-token inference
generated_ids = []
past_length = 0

all_tokens = list(input_ids)
for step in range(len(input_ids) + 10):  # prefill + 10 decode steps
    token_id = all_tokens[step]

    embed = embed_table[np.array([[token_id]])].astype(np.float32)  # [1, 1, 1024]

    inp = {
        "inputs_embeds": embed,
        "attention_mask": np.ones((1, past_length + 1), dtype=np.int64),
        "position_ids": np.full((3, 1, 1), past_length, dtype=np.int64),
        "beam_idx": np.array([0], dtype=np.int64),
    }

    # Feed GDN states
    inp.update(conv_states)
    inp.update(recurrent_states)

    # Feed KV caches
    inp.update(kv_caches)

    result = request.infer(inp)

    # Read logits
    logits = result["logits"]
    last_logits = logits[0, -1, :]
    next_token = int(np.argmax(last_logits))

    # Debug first 3 steps
    if step < 3 or step == len(input_ids) - 1:
        top5 = np.argsort(last_logits)[-5:][::-1]
        print(f"  Step {step}: logits range [{last_logits.min():.4f}, {last_logits.max():.4f}], "
              f"top5 tokens: {top5.tolist()}, "
              f"top5 texts: {[tokenizer.decode([t]) for t in top5]}")

    # Read updated GDN states
    for name in conv_states:
        out_name = name.replace("past", "present")
        conv_states[name] = np.array(result[out_name])
    for name in recurrent_states:
        out_name = name.replace("past", "present")
        recurrent_states[name] = np.array(result[out_name])

    # Read updated KV caches (present = full concat)
    for name in kv_caches:
        # past_key_values.0.key -> present.0.key
        idx = name.replace("past_key_values.", "").split(".")[0]
        kv_type = name.split(".")[-1]  # key or value
        out_name = f"present.{idx}.{kv_type}"
        kv_caches[name] = np.array(result[out_name])

    past_length += 1

    if step >= len(input_ids) - 1:
        generated_ids.append(next_token)
        all_tokens.append(next_token)
        text = tokenizer.decode([next_token])
        print(f"Step {step}: token={next_token} text={text!r}")

print()
print("Generated:", tokenizer.decode(generated_ids, skip_special_tokens=True))
