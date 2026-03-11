#!/usr/bin/env python3
"""Inspect NPU v2 subgraph IR shapes and test CPU inference."""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_QWEN35_DIR = _SCRIPT_DIR.parent
_PROJECT_DIR = _QWEN35_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

import openvino as ov
import numpy as np

model_path = Path("models/qwen35/Qwen3.5-0.8B-npu-v2")
core = ov.Core()

# Check shapes for all subgraphs
for i in range(6):
    xml = model_path / f"subgraph_{i}.xml"
    if not xml.exists():
        print(f"subgraph_{i}.xml not found!")
        continue
    m = core.read_model(str(xml))
    print(f"\n=== Subgraph {i} ===")
    has_dynamic = False
    for inp in m.inputs:
        name = inp.get_any_name()
        ps = inp.get_partial_shape()
        is_static = ps.is_static
        if not is_static:
            has_dynamic = True
        print(f"  IN  {name}: {ps}  {'STATIC' if is_static else 'DYNAMIC!'}")
    for out in m.outputs:
        name = out.get_any_name()
        ps = out.get_partial_shape()
        is_static = ps.is_static
        if not is_static:
            has_dynamic = True
        print(f"  OUT {name}: {ps}  {'STATIC' if is_static else 'DYNAMIC!'}")
    if has_dynamic:
        print(f"  *** SUBGRAPH {i} HAS DYNAMIC SHAPES! ***")

# Try CPU inference with subgraph 0
print("\n\n=== CPU Inference Test (subgraph 0) ===")
try:
    m0 = core.read_model(str(model_path / "subgraph_0.xml"))
    compiled = core.compile_model(m0, "CPU")
    request = compiled.create_infer_request()

    # Build dummy inputs
    inp_dict = {}
    for inp in m0.inputs:
        name = inp.get_any_name()
        ps = inp.get_partial_shape()
        shape = []
        for d in ps:
            if d.is_static:
                shape.append(d.get_length())
            else:
                shape.append(1)
        inp_dict[name] = np.zeros(shape, dtype=np.float32)

    request.infer(inp_dict)

    for out in m0.outputs:
        name = out.get_any_name()
        data = request.get_tensor(name).data
        print(f"  OUT {name}: shape={data.shape}, mean={data.mean():.6f}, std={data.std():.6f}")
    print("  CPU inference OK!")
except Exception as e:
    print(f"  CPU inference FAILED: {e}")

# Try full chain CPU inference
print("\n\n=== Full Chain CPU Test ===")
try:
    from qwen35.inference_npu_v2 import Qwen35NPUv2Model
    model = Qwen35NPUv2Model.from_pretrained(str(model_path), device="CPU")
    print(f"Model loaded: {model}")

    inputs = model.tokenizer("The capital of France is", return_tensors="pt")
    print(f"Input tokens: {inputs['input_ids'].shape}")

    import time
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    elapsed = time.time() - t0

    result = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_new = outputs.shape[1] - inputs["input_ids"].shape[1]
    print(f"Output: {result}")
    print(f"  ({num_new} tokens in {elapsed:.1f}s = {num_new/elapsed:.1f} tok/s)")
except Exception as e:
    import traceback
    print(f"  Full chain FAILED:")
    traceback.print_exc()
