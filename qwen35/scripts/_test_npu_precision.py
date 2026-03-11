#!/usr/bin/env python3
"""Test if converting model to FP32 before NPU compilation helps precision."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov
from openvino.preprocess import PrePostProcessor
from transformers import AutoTokenizer

MAX_CACHE_LEN = 256

def single_step_npu(model_obj, token_id, embed_table, config, past_length=0, states=None):
    """Run single step on NPU, return logits and updated states."""
    core = ov.Core()

    # Add output conversions
    ppp = PrePostProcessor(model_obj)
    for i in range(len(model_obj.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    processed_model = ppp.build()

    compiled = core.compile_model(processed_model, "NPU", config)
    request = compiled.create_infer_request()

    embed = embed_table[np.array([[token_id]])]

    total_len = MAX_CACHE_LEN + 1
    mask = np.full((1, 1, 1, total_len), -np.inf, dtype=np.float32)
    if past_length > 0:
        mask[0, 0, 0, :past_length] = 0.0
    mask[0, 0, 0, -1] = 0.0

    inp = {
        "inputs_embeds": embed,
        "attention_mask": mask,
        "position_ids": np.full((3, 1, 1), past_length, dtype=np.int64),
    }
    if states:
        inp.update(states)
    else:
        for inp_obj in compiled.inputs:
            name = inp_obj.get_any_name()
            if "cache_params.past" in name:
                shape = [d.get_length() for d in inp_obj.get_partial_shape()]
                inp[name] = np.zeros(shape, dtype=np.float32)

    request.infer(inp)
    logits = request.get_tensor("logits").data.copy()[0, 0, :]
    return logits


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(
        "models/qwen35/Qwen3.5-0.8B-ov", trust_remote_code=True
    )
    token_id = tok.encode("Hello")[0]

    xml_path = "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml"
    embed_path = "models/qwen35/Qwen3.5-0.8B-npu/embed_tokens.npy"
    embed_table = np.load(embed_path).astype(np.float32)

    config = {
        "NPU_USE_NPUW": "YES",
        "NPUW_FOLD": "NO",
        "CACHE_DIR": "models/qwen35/npu_cache",
    }

    core = ov.Core()
    print("Available NPU properties:")
    try:
        props = core.get_property("NPU", "SUPPORTED_PROPERTIES")
        for p in sorted(props):
            if "float" in p.lower() or "precision" in p.lower() or "hint" in p.lower():
                print("  %s" % p)
    except Exception as e:
        print("  (error: %s)" % e)

    # Try listing all properties
    print("\nAll NPU properties:")
    try:
        props = core.get_property("NPU", "SUPPORTED_PROPERTIES")
        for p in sorted(props):
            try:
                val = core.get_property("NPU", p)
                print("  %s = %s" % (p, val))
            except:
                print("  %s = (unreadable)" % p)
    except Exception as e:
        print("  (error: %s)" % e)
