#!/usr/bin/env python3
"""Debug: compare single-token logits between CPU and NPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

def run_single_token_cpu(xml_path, embed_path, token_id):
    """Run a single token through the CPU model."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()

    embed_table = np.load(str(embed_path)).astype(np.float32)
    embed = embed_table[np.array([[token_id]])]  # [1, 1, 1024]

    request.infer({
        "inputs_embeds": embed,
        "attention_mask": np.ones((1, 1), dtype=np.int64),
        "position_ids": np.zeros((3, 1, 1), dtype=np.int64),
        "beam_idx": np.zeros(1, dtype=np.int32),
    })

    logits = request.get_tensor("logits").data.copy()[0, 0, :]
    return logits

def run_single_token_npu(xml_path, embed_path, token_id, max_cache_len=256):
    """Run a single token through the NPU model."""
    core = ov.Core()
    model = core.read_model(str(xml_path))

    # Add f16->f32 conversion for NPU outputs
    from openvino.preprocess import PrePostProcessor
    ppp = PrePostProcessor(model)
    for i in range(len(model.outputs)):
        ppp.output(i).tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    compiled = core.compile_model(model, "NPU", {
        "NPU_USE_NPUW": "YES",
        "NPUW_FOLD": "NO",
        "CACHE_DIR": "models/qwen35/npu_cache",
    })
    request = compiled.create_infer_request()

    embed_table = np.load(str(embed_path)).astype(np.float32)
    embed = embed_table[np.array([[token_id]])]  # [1, 1, 1024]

    # Build inputs with zero KV buffers
    inp = {
        "inputs_embeds": embed,
        # 4D mask: only unmask the last position (current token)
        "attention_mask": np.full((1, 1, 1, max_cache_len + 1), -np.inf, dtype=np.float32),
        "position_ids": np.zeros((3, 1, 1), dtype=np.int64),
    }
    # Unmask only the current token (last position)
    inp["attention_mask"][0, 0, 0, -1] = 0.0

    # Zero KV cache buffers
    for inp_obj in compiled.inputs:
        name = inp_obj.get_any_name()
        if "cache_params.past" in name:
            shape = list(inp_obj.get_partial_shape())
            shape = [d.get_length() for d in shape]
            inp[name] = np.zeros(shape, dtype=np.float32)

    request.infer(inp)
    logits = request.get_tensor("logits").data.copy()[0, 0, :]
    return logits

def run_single_token_npu_on_cpu(xml_path, embed_path, token_id, max_cache_len=256):
    """Run the NPU model IR on CPU to isolate IR vs hardware issues."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()

    embed_table = np.load(str(embed_path)).astype(np.float32)
    embed = embed_table[np.array([[token_id]])]

    inp = {
        "inputs_embeds": embed,
        "attention_mask": np.full((1, 1, 1, max_cache_len + 1), -np.inf, dtype=np.float32),
        "position_ids": np.zeros((3, 1, 1), dtype=np.int64),
    }
    inp["attention_mask"][0, 0, 0, -1] = 0.0

    for inp_obj in compiled.inputs:
        name = inp_obj.get_any_name()
        if "cache_params.past" in name:
            shape = list(inp_obj.get_partial_shape())
            shape = [d.get_length() for d in shape]
            inp[name] = np.zeros(shape, dtype=np.float32)

    request.infer(inp)
    logits = request.get_tensor("logits").data.copy()[0, 0, :]
    return logits

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(
        "models/qwen35/Qwen3.5-0.8B-ov", trust_remote_code=True
    )
    token_id = tok.encode("Hello")[0]
    print("Token: %d (%s)" % (token_id, repr(tok.decode([token_id]))))

    print("\n1. CPU model (stateful IR)...")
    cpu_logits = run_single_token_cpu(
        "models/qwen35/Qwen3.5-0.8B-ov/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-ov/embed_tokens.npy",
        token_id,
    )

    print("2. NPU IR on CPU (static-cache IR, CPU device)...")
    npu_ir_on_cpu_logits = run_single_token_npu_on_cpu(
        "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-npu/embed_tokens.npy",
        token_id,
    )

    print("3. NPU model (static-cache IR, NPU device)...")
    npu_logits = run_single_token_npu(
        "models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml",
        "models/qwen35/Qwen3.5-0.8B-npu/embed_tokens.npy",
        token_id,
    )

    def show_top5(name, logits):
        top5 = np.argsort(logits)[-5:][::-1]
        print("%s top-5:" % name)
        for t in top5:
            print("  %d (%s): %.4f" % (t, repr(tok.decode([t])), logits[t]))

    print("\n=== Comparison ===")
    show_top5("CPU (stateful)", cpu_logits)
    show_top5("NPU-IR-on-CPU", npu_ir_on_cpu_logits)
    show_top5("NPU (device)", npu_logits)

    diff_ir = np.abs(cpu_logits - npu_ir_on_cpu_logits)
    diff_dev = np.abs(npu_ir_on_cpu_logits - npu_logits)
    print("\nCPU vs NPU-IR-on-CPU: max=%.4f mean=%.4f" % (diff_ir.max(), diff_ir.mean()))
    print("NPU-IR-on-CPU vs NPU: max=%.4f mean=%.4f" % (diff_dev.max(), diff_dev.mean()))
