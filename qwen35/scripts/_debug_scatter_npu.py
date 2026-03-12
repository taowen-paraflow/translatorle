"""Diagnostic: 2-step explicit I/O to check if KV cache accumulates correctly."""
import numpy as np
import openvino as ov

core = ov.Core()
ir_path = "models/qwen35/Qwen3.5-0.8B-hybrid-v2/attn_block_0.xml"

B, S, H, D, MCL = 1, 1, 2, 256, 256

ir = core.read_model(ir_path)
shapes = {}
for i, inp in enumerate(ir.inputs):
    name = inp.get_any_name()
    if i == 0:
        shapes[name] = ov.PartialShape([B, S, 1024])
    elif i == 1:
        shapes[name] = ov.PartialShape([3, B, S])
    elif i in (2, 3):
        shapes[name] = ov.PartialShape([B, H, MCL, D])
    elif i == 4:
        shapes[name] = ov.PartialShape([S])
    elif i == 5:
        shapes[name] = ov.PartialShape([B, 1, S, MCL])
ir.reshape(shapes)

# Also test with f32 output conversion (like inference code does)
from openvino.preprocess import PrePostProcessor
ppp = PrePostProcessor(ir)
for i in range(len(ir.outputs)):
    ppp.output(i).tensor().set_element_type(ov.Type.f32)
ir = ppp.build()

npu_model = core.compile_model(ir, "NPU")
req = npu_model.create_infer_request()
print("NPU compiled OK")

# Step 1: write at position 0
np.random.seed(42)
hidden1 = np.random.randn(B, S, 1024).astype(np.float32) * 0.1
key_cache = np.zeros((B, H, MCL, D), dtype=np.float32)
value_cache = np.zeros((B, H, MCL, D), dtype=np.float32)
cache_pos = np.array([0], dtype=np.int64)
pos_ids = np.zeros((3, B, S), dtype=np.int64)
mask = np.full((B, 1, S, MCL), -65504.0, dtype=np.float32)
mask[:, :, :, 0] = 0.0

inputs = [hidden1, pos_ids, key_cache, value_cache, cache_pos, mask]
for i, arr in enumerate(inputs):
    req.set_input_tensor(i, ov.Tensor(np.ascontiguousarray(arr)))
req.infer()

out_hidden1 = req.get_output_tensor(0).data.copy()
out_key1 = req.get_output_tensor(1).data.copy()
out_val1 = req.get_output_tensor(2).data.copy()

print("\n=== Step 1 (write pos=0) ===")
print("Hidden mean=%.6f" % out_hidden1.mean())
print("Key pos=0 nonzero:", np.count_nonzero(out_key1[0, :, 0, :]))
print("Key pos=1 nonzero:", np.count_nonzero(out_key1[0, :, 1, :]))
print("Key pos=0 first 5:", out_key1[0, 0, 0, :5])

# Step 2: write at position 1, passing back the KV from step 1
hidden2 = np.random.randn(B, S, 1024).astype(np.float32) * 0.1
cache_pos2 = np.array([1], dtype=np.int64)
pos_ids2 = np.ones((3, B, S), dtype=np.int64)
mask2 = np.full((B, 1, S, MCL), -65504.0, dtype=np.float32)
mask2[:, :, :, :2] = 0.0  # attend to positions 0 and 1

inputs2 = [hidden2, pos_ids2, out_key1, out_val1, cache_pos2, mask2]
for i, arr in enumerate(inputs2):
    req.set_input_tensor(i, ov.Tensor(np.ascontiguousarray(arr)))
req.infer()

out_hidden2 = req.get_output_tensor(0).data.copy()
out_key2 = req.get_output_tensor(1).data.copy()

print("\n=== Step 2 (write pos=1, feed back KV from step 1) ===")
print("Hidden mean=%.6f" % out_hidden2.mean())
print("Key pos=0 nonzero:", np.count_nonzero(out_key2[0, :, 0, :]))
print("Key pos=1 nonzero:", np.count_nonzero(out_key2[0, :, 1, :]))
print("Key pos=2 nonzero:", np.count_nonzero(out_key2[0, :, 2, :]))
print("Key pos=0 first 5 (should match step1):", out_key2[0, 0, 0, :5])
print("Key pos=1 first 5 (newly written):", out_key2[0, 0, 1, :5])

# Verify position 0 was preserved from step 1
diff = np.abs(out_key1[0, :, 0, :] - out_key2[0, :, 0, :]).max()
print("\nPos=0 preserved from step1? Max diff:", diff)

# Now compare with CPU
print("\n=== CPU comparison ===")
ir_cpu = core.read_model(ir_path)
ir_cpu.reshape(shapes)
ppp = PrePostProcessor(ir_cpu)
for i in range(len(ir_cpu.outputs)):
    ppp.output(i).tensor().set_element_type(ov.Type.f32)
ir_cpu = ppp.build()
cpu_model = core.compile_model(ir_cpu, "CPU")
cpu_req = cpu_model.create_infer_request()

# Replay step 1 + step 2 on CPU
np.random.seed(42)
hidden1 = np.random.randn(B, S, 1024).astype(np.float32) * 0.1
key_cache = np.zeros((B, H, MCL, D), dtype=np.float32)
value_cache = np.zeros((B, H, MCL, D), dtype=np.float32)
inputs_cpu = [hidden1, pos_ids, key_cache, value_cache, cache_pos, mask]
for i, arr in enumerate(inputs_cpu):
    cpu_req.set_input_tensor(i, ov.Tensor(np.ascontiguousarray(arr)))
cpu_req.infer()
cpu_key1 = cpu_req.get_output_tensor(1).data.copy()
cpu_val1 = cpu_req.get_output_tensor(2).data.copy()
cpu_hidden1 = cpu_req.get_output_tensor(0).data.copy()

hidden2 = np.random.randn(B, S, 1024).astype(np.float32) * 0.1
inputs_cpu2 = [hidden2, pos_ids2, cpu_key1, cpu_val1, cache_pos2, mask2]
for i, arr in enumerate(inputs_cpu2):
    cpu_req.set_input_tensor(i, ov.Tensor(np.ascontiguousarray(arr)))
cpu_req.infer()
cpu_hidden2 = cpu_req.get_output_tensor(0).data.copy()
cpu_key2 = cpu_req.get_output_tensor(1).data.copy()

print("CPU step2 hidden mean=%.6f" % cpu_hidden2.mean())
print("NPU step2 hidden mean=%.6f" % out_hidden2.mean())
print("Step2 hidden max diff:", np.abs(cpu_hidden2 - out_hidden2).max())
print("Step2 key max diff:", np.abs(cpu_key2 - out_key2).max())
