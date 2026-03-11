#!/usr/bin/env python3
"""Test NPU with ACCURACY execution mode hint."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor
from transformers import AutoTokenizer

MAX_CACHE_LEN = 256

def prefill_npu(xml_path, embed_path, token_ids, config, label):
    """Token-by-token prefill on NPU."""
    core = ov.Core()
    model = core.read_model(str(xml_path))

    ppp = PrePostProcessor(model)
    for i in range(len(model.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    compiled = core.compile_model(model, "NPU", config)
    request = compiled.create_infer_request()

    embed_table = np.load(str(embed_path)).astype(np.float32)

    states = {}
    for inp_obj in compiled.inputs:
        name = inp_obj.get_any_name()
        if "cache_params.past" in name:
            shape = [d.get_length() for d in inp_obj.get_partial_shape()]
            states[name] = np.zeros(shape, dtype=np.float32)

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
        if past_length > 0:
            mask[0, 0, 0, :past_length] = 0.0
        mask[0, 0, 0, -1] = 0.0

        inp = {"inputs_embeds": embed, "attention_mask": mask,
               "position_ids": np.full((3, 1, 1), past_length, dtype=np.int64)}
        inp.update(states)
        request.infer(inp)

        for pn, prn in gdn_out_map.items():
            states[pn] = request.get_tensor(prn).data.copy()
        for pn, prn in kv_out_map.items():
            new_kv = request.get_tensor(prn).data.copy()
            if past_length < MAX_CACHE_LEN:
                states[pn][:, :, past_length:past_length+1, :] = new_kv
        past_length += 1
    elapsed = time.time() - t0

    logits = request.get_tensor("logits").data.copy()[0, 0, :]
    return logits, elapsed


def prefill_cpu(xml_path, embed_path, token_ids):
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
    tok = AutoTokenizer.from_pretrained("models/qwen35/Qwen3.5-0.8B-ov", trust_remote_code=True)
    token_ids = tok.encode("The capital of France is")

    npu_xml = "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml"
    npu_embed = "models/qwen35/Qwen3.5-0.8B-npu/embed_tokens.npy"

    base_config = {
        "NPU_USE_NPUW": "YES",
        "NPUW_FOLD": "NO",
        "CACHE_DIR": "models/qwen35/npu_cache",
    }

    print("CPU reference...")
    cpu_logits = prefill_cpu(
        "models/qwen35/Qwen3.5-0.8B-ov/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-ov/embed_tokens.npy",
        token_ids,
    )
    cpu_top1 = cpu_logits.argmax()
    print("  top1: %d (%s)" % (cpu_top1, repr(tok.decode([cpu_top1]))))

    configs = [
        ("default (PERF)", {}),
        ("ACCURACY mode", {"EXECUTION_MODE_HINT": "ACCURACY"}),
        ("NPU_TURBO", {"NPU_TURBO": "YES"}),
    ]

    for label, extra in configs:
        cfg = dict(base_config)
        cfg.update(extra)
        print("\nNPU %s..." % label)
        try:
            logits, elapsed = prefill_npu(npu_xml, npu_embed, token_ids, cfg, label)
            top1 = logits.argmax()
            diff = np.abs(cpu_logits - logits)
            print("  top1: %d (%s) = %.4f" % (top1, repr(tok.decode([top1])), logits[top1]))
            print("  time: %.2fs, max_diff: %.4f, mean_diff: %.4f" % (elapsed, diff.max(), diff.mean()))
        except Exception as e:
            print("  ERROR: %s" % e)
