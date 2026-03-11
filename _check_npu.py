"""Check why NPU doesn't appear in available devices."""
import openvino as ov
import os, sys

print(f"OpenVINO: {ov.__version__}")
print(f"Python: {sys.executable}")

core = ov.Core()
print(f"Available devices: {core.available_devices}")

# Check NPU plugin file
libs_dir = os.path.dirname(ov.__file__) + "/libs"
npu_dll = os.path.join(libs_dir, "openvino_intel_npu_plugin.dll")
npu_compiler = os.path.join(libs_dir, "openvino_intel_npu_compiler.dll")
print(f"\nNPU plugin exists: {os.path.exists(npu_dll)}")
print(f"NPU compiler exists: {os.path.exists(npu_compiler)}")

if os.path.exists(npu_dll):
    print(f"NPU plugin size: {os.path.getsize(npu_dll):,} bytes")
if os.path.exists(npu_compiler):
    print(f"NPU compiler size: {os.path.getsize(npu_compiler):,} bytes")

# Try to get NPU properties
if "NPU" not in core.available_devices:
    print("\nNPU not in available_devices. Trying to compile a dummy model on NPU...")
    try:
        import numpy as np
        from openvino import opset13
        param = opset13.parameter([1, 10], ov.Type.f32, name="input")
        result = opset13.result(param)
        model = ov.Model([result], [param])
        compiled = core.compile_model(model, "NPU")
        print("  NPU compile succeeded!")
    except Exception as e:
        print(f"  NPU compile failed: {e}")
else:
    print(f"\nNPU device name: {core.get_property('NPU', 'FULL_DEVICE_NAME')}")
