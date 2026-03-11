"""Debug NaN in NPUW model on CPU - single step."""
import numpy as np
import openvino as ov
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from qwen35.config import NPUW_MODEL_DIR

core = ov.Core()
model = core.read_model(str(NPUW_MODEL_DIR / "openvino_model.xml"))

# Check all input shapes
print("Model inputs:")
for inp in model.inputs:
    print(f"  {inp.get_any_name()}: {inp.get_partial_shape()} {inp.get_element_type()}")

# Build inputs for a single token
embed_table = np.load(str(NPUW_MODEL_DIR / "embed_tokens.npy")).astype(np.float32)
token_id = 9419  # "Hello"

inputs = {}
inputs["inputs_embeds"] = embed_table[np.array([[token_id]])]  # [1, 1, 1024]
inputs["attention_mask"] = np.ones((1, 1), dtype=np.int64)
inputs["position_ids"] = np.zeros((3, 1, 1), dtype=np.int64)
inputs["beam_idx"] = np.array([0], dtype=np.int64)

# Initialize all cache inputs
for inp_info in model.inputs:
    name = inp_info.get_any_name()
    if name in inputs:
        continue
    ps = inp_info.get_partial_shape()
    shape = []
    for i, d in enumerate(ps):
        is_dyn = d.is_dynamic if isinstance(d.is_dynamic, bool) else d.is_dynamic()
        if is_dyn:
            if i == 0:
                shape.append(1)  # batch
            elif "past_key_values" in name and i == 2:
                shape.append(0)  # seq_len for KV = 0 (empty)
            else:
                shape.append(1)  # default
        else:
            length = d.get_length() if hasattr(d, 'get_length') else int(d)
            shape.append(length)
    dtype = np.float32 if inp_info.get_element_type() == ov.Type.f32 else np.int64
    inputs[name] = np.zeros(shape, dtype=dtype)
    print(f"  Init {name}: shape={shape}")

print(f"\ninputs_embeds: shape={inputs['inputs_embeds'].shape}, "
      f"range=[{inputs['inputs_embeds'].min():.4f}, {inputs['inputs_embeds'].max():.4f}], "
      f"nan={np.any(np.isnan(inputs['inputs_embeds']))}")

# Compile and run
compiled = core.compile_model(model, "CPU")
request = compiled.create_infer_request()
result = request.infer(inputs)

# Check outputs
print("\nOutputs:")
for out in compiled.outputs:
    name = out.get_any_name()
    data = np.array(result[name])
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    print(f"  {name}: shape={data.shape}, nan={has_nan}, inf={has_inf}, "
          f"range=[{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")
