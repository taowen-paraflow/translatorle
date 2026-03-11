#!/usr/bin/env python3
"""Debug: compare multi-step prefill between stateful-CPU and static-cache-CPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

MAX_CACHE_LEN = 256

def prefill_stateful_cpu(xml_path, embed_path, token_ids):
    """Batch prefill on CPU stateful model."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()

    embed_table = np.load(str(embed_path)).astype(np.float32)
    ids = np.array([token_ids])
    embeds = embed_table[ids]  # [1, seq_len, 1024]
    seq_len = len(token_ids)

    positions = np.arange(seq_len, dtype=np.int64)[np.newaxis, np.newaxis, :]
    positions = np.tile(positions, (3, 1, 1))

    request.infer({
        "inputs_embeds": embeds,
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": positions,
        "beam_idx": np.zeros(1, dtype=np.int32),
    })
    return request.get_tensor("logits").data.copy()[0, -1, :]

def prefill_static_cache_on_cpu(xml_path, embed_path, token_ids):
    """Token-by-token prefill of static-cache IR on CPU."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()

    embed_table = np.load(str(embed_path)).astype(np.float32)

    # Initialize states
    gdn_states = {}
    kv_buffers = {}
    for inp_obj in compiled.inputs:
        name = inp_obj.get_any_name()
        if "cache_params.past" in name:
            shape = [d.get_length() for d in inp_obj.get_partial_shape()]
            if "key" in name or "value" in name:
                kv_buffers[name] = np.zeros(shape, dtype=np.float32)
            else:
                gdn_states[name] = np.zeros(shape, dtype=np.float32)

    # Discover output names for state readback
    kv_out_names = {}
    gdn_out_names = {}
    for out_obj in compiled.outputs:
        oname = out_obj.get_any_name()
        if "cache_params.present.key" in oname or "cache_params.present.value" in oname:
            past_name = oname.replace("present", "past")
            kv_out_names[past_name] = oname
        elif "cache_params.present" in oname:
            past_name = oname.replace("present", "past")
            gdn_out_names[past_name] = oname

    past_length = 0
    for i, tid in enumerate(token_ids):
        embed = embed_table[np.array([[tid]])]  # [1, 1, 1024]

        # Build 4D mask
        total_len = MAX_CACHE_LEN + 1
        mask = np.full((1, 1, 1, total_len), -np.inf, dtype=np.float32)
        t = min(past_length, MAX_CACHE_LEN)
        if t > 0:
            mask[0, 0, 0, :t] = 0.0
        mask[0, 0, 0, -1] = 0.0

        inp = {
            "inputs_embeds": embed,
            "attention_mask": mask,
            "position_ids": np.full((3, 1, 1), past_length, dtype=np.int64),
        }
        inp.update(gdn_states)
        inp.update(kv_buffers)

        request.infer(inp)

        # Read back GDN states
        for past_name, present_name in gdn_out_names.items():
            gdn_states[past_name] = request.get_tensor(present_name).data.copy()

        # Read back KV and write into buffer
        for past_name, present_name in kv_out_names.items():
            new_kv = request.get_tensor(present_name).data.copy()  # [1, H, 1, D]
            pos = past_length
            if pos < MAX_CACHE_LEN:
                kv_buffers[past_name][:, :, pos:pos+1, :] = new_kv

        past_length += 1

        if i < 3 or i == len(token_ids) - 1:
            logits = request.get_tensor("logits").data.copy()[0, 0, :]
            top1 = logits.argmax()
            print("  Step %d: top1=%d (%.4f)" % (i, top1, logits[top1]))

    return request.get_tensor("logits").data.copy()[0, 0, :]


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(
        "models/qwen35/Qwen3.5-0.8B-ov", trust_remote_code=True
    )
    token_ids = tok.encode("The capital of France is")
    print("Tokens:", token_ids, [tok.decode([t]) for t in token_ids])

    print("\n1. Stateful CPU (batch prefill)...")
    cpu_logits = prefill_stateful_cpu(
        "models/qwen35/Qwen3.5-0.8B-ov/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-ov/embed_tokens.npy",
        token_ids,
    )

    print("\n2. Static-cache IR on CPU (token-by-token)...")
    sc_logits = prefill_static_cache_on_cpu(
        "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-npu/embed_tokens.npy",
        token_ids,
    )

    print("\n=== Final logits comparison ===")
    cpu_top5 = np.argsort(cpu_logits)[-5:][::-1]
    sc_top5 = np.argsort(sc_logits)[-5:][::-1]

    print("Stateful CPU top-5:")
    for t in cpu_top5:
        print("  %d (%s): %.4f" % (t, repr(tok.decode([t])), cpu_logits[t]))

    print("Static-cache on CPU top-5:")
    for t in sc_top5:
        print("  %d (%s): %.4f" % (t, repr(tok.decode([t])), sc_logits[t]))

    diff = np.abs(cpu_logits - sc_logits)
    print("\nMax diff: %.4f  Mean diff: %.4f" % (diff.max(), diff.mean()))
