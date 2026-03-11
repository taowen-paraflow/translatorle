"""Inspect shapes of the exported NPU and CPU models."""
import openvino as ov

def inspect(xml_path):
    core = ov.Core()
    model = core.read_model(xml_path)

    print("=== INPUTS ===")
    for inp in model.inputs:
        name = inp.get_any_name()
        shape = inp.get_partial_shape()
        dtype = inp.get_element_type()
        print("  %s: %s %s" % (name, shape, dtype))

    print()
    print("=== OUTPUTS ===")
    for out in model.outputs:
        name = out.get_any_name()
        shape = out.get_partial_shape()
        dtype = out.get_element_type()
        print("  %s: %s %s" % (name, shape, dtype))

    print()
    print("=== ReadValue ops ===")
    rv_count = 0
    for op in model.get_ops():
        if op.get_type_name() == "ReadValue":
            rv_count += 1
            vid = op.get_variable_id() if hasattr(op, "get_variable_id") else "?"
            shape = op.get_output_partial_shape(0)
            if rv_count <= 5:
                print("  %s: %s" % (vid, shape))
    if rv_count > 5:
        print("  ... and %d more" % (rv_count - 5))
    print("  Total: %d" % rv_count)

    print()
    sinks = model.get_sinks()
    print("=== Sinks (Assign) === Count: %d" % len(sinks))
    for s in sinks[:3]:
        vid = s.get_variable_id() if hasattr(s, "get_variable_id") else "?"
        print("  Assign var=%s" % vid)
    if len(sinks) > 3:
        print("  ... and %d more" % (len(sinks) - 3))


if __name__ == "__main__":
    print("========== NPU MODEL ==========")
    inspect("models/qwen35/Qwen3.5-0.8B-npu/openvino_model.xml")
    print()
    print("========== CPU MODEL ==========")
    inspect("models/qwen35/Qwen3.5-0.8B-ov/openvino_model.xml")
