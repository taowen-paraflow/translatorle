"""Quick check if NPU is available."""
import openvino as ov

core = ov.Core()
print("Available devices:", core.available_devices)

try:
    ver = core.get_property("NPU", "FULL_DEVICE_NAME")
    print("NPU device:", ver)
except Exception as e:
    print("NPU error:", e)

# Check if plugin DLL is the right one
import os
libs_dir = os.path.dirname(ov.__file__) + "/libs"
npu_dll = os.path.join(libs_dir, "openvino_intel_npu_plugin.dll")
if os.path.exists(npu_dll):
    size = os.path.getsize(npu_dll)
    print(f"NPU DLL size: {size} bytes")
else:
    print("NPU DLL not found!")
