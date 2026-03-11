#!/usr/bin/env python3
"""Test if saving NPU model without FP16 compression improves precision.

Instead of re-exporting (slow), we load the FP16 IR and check if the
NPU compiler produces the same results regardless.
We can also try converting the existing model outputs to see if there's
a way to improve NPU precision.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor
from transformers import AutoTokenizer

MAX_CACHE_LEN = 256

def prefill_on_npu(xml_path, embed_path, token_ids, extra_config=None):
    """Token-by-token prefill on NPU with static-cache IR."""
    core = ov.Core()
    model = core.read_model(str(xml_path))

    ppp = PrePostProcessor(model)
    for i in range(len(model.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    config = {
        "NPU_USE_NPUW": "YES",
        "NPUW_FOLD": "NO",
        "CACHE_DIR": "models/qwen35/npu_cache",
    }
    if extra_config:
        config.update(extra_config)

    compiled = core.compile_model(model, "NPU", config)
    request = compiled.create_infer_request()

    embed_table = np.load(str(embed_path)).astype(np.float32)

    # Initialize states
    states = {}
    for inp_obj in compiled.inputs:
        name = inp_obj.get_any_name()
        if "cache_params.past" in name:
            shape = [d.get_length() for d in inp_obj.get_partial_shape()]
            states[name] = np.zeros(shape, dtype=np.float32)

    kv_inputs = sorted(n for n in states if "key" in n or "value" in n)
    gdn_inputs = sorted(n for n in states if "conv" in n or "recurrent" in n)

    kv_out_map = {}
    gdn_out_map = {}
    for out_obj in compiled.outputs:
        oname = out_obj.get_any_name()
        past_name = oname.replace("present", "past")
        if "key" in oname or "value" in oname:
            kv_out_map[past_name] = oname
        elif "cache_params.present" in oname:
            gdn_out_map[past_name] = oname

    past_length = 0
    t0 = time.time()
    for tid in token_ids:
        embed = embed_table[np.array([[tid]])]

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
        inp.update(states)

        request.infer(inp)

        for past_name, present_name in gdn_out_map.items():
            states[past_name] = request.get_tensor(present_name).data.copy()
        for past_name, present_name in kv_out_map.items():
            new_kv = request.get_tensor(present_name).data.copy()
            pos = past_length
            if pos < MAX_CACHE_LEN:
                states[past_name][:, :, pos:pos+1, :] = new_kv

        past_length += 1
    elapsed = time.time() - t0

    logits = request.get_tensor("logits").data.copy()[0, 0, :]
    return logits, elapsed


def prefill_on_cpu(xml_path, embed_path, token_ids):
    """Batch prefill on CPU stateful model (reference)."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()

    embed_table = np.load(str(embed_path)).astype(np.float32)
    ids = np.array([token_ids])
    embeds = embed_table[ids]
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


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(
        "models/qwen35/Qwen3.5-0.8B-ov", trust_remote_code=True
    )
    token_ids = tok.encode("The capital of France is")
    print("Tokens:", token_ids)

    print("\n1. CPU reference...")
    cpu_logits = prefill_on_cpu(
        "models/qwen35/Qwen3.5-0.8B-ov/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-ov/embed_tokens.npy",
        token_ids,
    )

    print("\n2. NPU default config...")
    npu_logits, npu_t = prefill_on_npu(
        "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-npu/embed_tokens.npy",
        token_ids,
    )

    print("\n3. NPU with NPUW_ONLINE_PIPELINE=NONE...")
    npu2_logits, npu2_t = prefill_on_npu(
        "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-npu/embed_tokens.npy",
        token_ids,
        extra_config={"NPUW_ONLINE_PIPELINE": "NONE"},
    )

    cpu_top1 = cpu_logits.argmax()
    npu_top1 = npu_logits.argmax()
    npu2_top1 = npu2_logits.argmax()

    print("\n=== Results ===")
    print("CPU top-1: %d (%s) = %.4f" % (cpu_top1, repr(tok.decode([cpu_top1])), cpu_logits[cpu_top1]))
    print("NPU top-1: %d (%s) = %.4f  (%.2fs)" % (npu_top1, repr(tok.decode([npu_top1])), npu_logits[npu_top1], npu_t))
    print("NPU2 top-1: %d (%s) = %.4f  (%.2fs)" % (npu2_top1, repr(tok.decode([npu2_top1])), npu2_logits[npu2_top1], npu2_t))

    diff1 = np.abs(cpu_logits - npu_logits)
    diff2 = np.abs(cpu_logits - npu2_logits)
    print("\nCPU vs NPU: max=%.4f mean=%.4f" % (diff1.max(), diff1.mean()))
    print("CPU vs NPU2: max=%.4f mean=%.4f" % (diff2.max(), diff2.mean()))
