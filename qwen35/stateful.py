#  Standalone OpenVINO stateful transform for Qwen3.5 hybrid models.
#
#  Adapted from optimum-intel (optimum/exporters/openvino/stateful.py).
#  Original code Copyright 2023 The HuggingFace Team, Apache-2.0 license.
#
#  This module has NO dependency on optimum or optimum-intel.
#  It requires only: openvino, numpy.

import logging as log
from typing import List

import numpy as np

import openvino as ov
from openvino import opset13


def model_has_state(ov_model: ov.Model) -> bool:
    """Check whether the model already contains ReadValue (stateful) ops."""
    if isinstance(ov_model, ov.CompiledModel):
        return len(ov_model.query_state()) > 0
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str) -> bool:
    """Return True if *name* appears among the model's input or output tensor names."""
    return name in sum(
        [list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], []
    )


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List,
    key_value_input_names: List[str],
    gather_dim: int,
) -> None:
    """Insert a ``beam_idx`` parameter and per-state Gather ops for beam reorder.

    This must run **before** ``make_stateful``.  Each cache input listed in
    *key_value_input_names* gets a Gather along *gather_dim* so that the
    runtime can reorder beams without touching model state directly.
    The new ``beam_idx`` input is appended to *not_kv_inputs*.
    """
    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")

    main_input_name = (
        "input_ids"
        if model_has_input_output_name(ov_model, "input_ids")
        else "inputs_embeds"
    )
    input_batch = ov_model.input(main_input_name).get_partial_shape()[0]
    beam_idx = opset13.parameter(
        name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch])
    )
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])

    # Fuse _reorder_cache: insert Gather for every cache parameter
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(
            parameter_output_port, beam_idx, opset13.constant(gather_dim)
        )
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))

    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int) -> None:
    """Build ShapeOf-based initialization expressions for all ReadValue ops.

    Each ReadValue node gets a Broadcast(0, shape) initializer whose batch
    dimension is derived from the ``input_ids`` (or ``inputs_embeds``) shape
    at runtime, while all other dimensions use the static sizes from the
    partial shape.
    """
    main_input_name = (
        "input_ids"
        if model_has_input_output_name(ov_model, "input_ids")
        else "inputs_embeds"
    )
    input_ids = ov_model.input(main_input_name)
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )

    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [
                opset13.constant(np.array([dim], dtype=np.int64))
                if isinstance(dim, int)
                else dim
                for dim in dims
            ]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(
                opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape
            )
            op.set_arguments([broadcast])

    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List,
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_beams_and_batch: int = None,
) -> None:
    """Hide cache inputs/outputs inside the model as stateful variables.

    Calls OpenVINO's native ``apply_make_stateful_transformation`` to convert
    matching input/output pairs into ReadValue/Assign variable pairs.

    Parameters
    ----------
    ov_model : ov.Model
        The OpenVINO model to transform in-place.
    not_kv_inputs : list
        Input ports that are *not* key/value or SSM cache (e.g. input_ids,
        position_ids, beam_idx).
    key_value_input_names : list[str]
        Tensor names of cache inputs (conv, recurrent, key, value).
    key_value_output_names : list[str]
        Tensor names of corresponding cache outputs.
    batch_dim : int
        Index of the batch dimension in cache tensors.
    num_beams_and_batch : int, optional
        If given, pin the batch dimension to this value for all inputs.
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    if num_beams_and_batch is not None:
        # Pin batch size for non-cache inputs (input_ids, attention_mask, beam_idx)
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_idx
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
            else:
                log.warning(
                    f"Rank of {input.get_any_name()} input of the model is not 2, "
                    "batch size is not set"
                )

    input_output_map = {}
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)

    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful_hybrid_ssm(ov_model: ov.Model) -> None:
    """Main entry point -- convert a Qwen3.5 hybrid model to stateful form.

    Identifies conv / recurrent / key / value cache tensors by their name
    prefixes, inserts beam-reorder Gather ops, then converts all cache
    input/output pairs into OpenVINO stateful variables.

    Expected tensor naming convention (set during ONNX/OV export):

    - ``cache_params.past.conv.*``  /  ``cache_params.present.conv.*``
    - ``cache_params.past.recurrent.*``  /  ``cache_params.present.recurrent.*``
    - ``cache_params.past.key.*``  /  ``cache_params.present.key.*``
    - ``cache_params.past.value.*``  /  ``cache_params.present.value.*``
    """

    def _classify_tensors(ssm_prefixes, kv_prefixes, ov_tensors):
        """Split tensors into KV cache names, SSM state names, and the rest."""
        kv_names = []
        ssm_names = []
        other_tensors = []
        for ov_tensor in ov_tensors:
            ov_tensor_names = ov_tensor.get_names()
            is_kv_or_ssm = False
            for ov_tensor_name in ov_tensor_names:
                if any(prefix in ov_tensor_name for prefix in ssm_prefixes):
                    ssm_names.append(ov_tensor_name)
                    is_kv_or_ssm = True
                    break
                elif any(prefix in ov_tensor_name for prefix in kv_prefixes):
                    kv_names.append(ov_tensor_name)
                    is_kv_or_ssm = True
                    break
            if not is_kv_or_ssm:
                other_tensors.append(ov_tensor)
        return kv_names, ssm_names, other_tensors

    # -- Input prefixes (Qwen3.5: conv + recurrent for linear_attention, key + value for full_attention)
    ssm_prefix_input = [
        "cache_params.past.conv",
        "cache_params.past.recurrent",
    ]
    kv_prefix_input = [
        "cache_params.past.key",
        "cache_params.past.value",
    ]
    kv_input_names, ssm_input_names, not_cache_inputs = _classify_tensors(
        ssm_prefix_input, kv_prefix_input, ov_model.inputs
    )
    cache_inputs = kv_input_names + ssm_input_names

    # -- Output prefixes
    ssm_prefix_output = [
        "cache_params.present.conv",
        "cache_params.present.recurrent",
    ]
    kv_prefix_output = [
        "cache_params.present.key",
        "cache_params.present.value",
    ]
    kv_output_names, ssm_output_names, _ = _classify_tensors(
        ssm_prefix_output, kv_prefix_output, ov_model.outputs
    )
    cache_outputs = kv_output_names + ssm_output_names

    batch_dim = 0
    fuse_cache_reorder(ov_model, not_cache_inputs, cache_inputs, batch_dim)
    make_stateful(ov_model, not_cache_inputs, cache_inputs, cache_outputs, batch_dim)


def patch_stateful_kv_only(ov_model: ov.Model) -> None:
    """Convert only KV cache (key/value) pairs to stateful form.

    Conv and recurrent states remain as explicit Parameters and Results,
    managed in Python at inference time.  This is needed for NPU where
    NPUW_LLM manages KV cache internally but cannot handle the non-KV
    GDN states (different shape semantics: dim 2 in KV cache is
    past_sequence_length, but dim 2 in recurrent state is D_k=128
    which is static and must not be treated as a sequence dimension).
    """
    kv_prefixes = ["cache_params.past.key", "cache_params.past.value"]
    kv_output_prefixes = ["cache_params.present.key", "cache_params.present.value"]

    kv_input_names = []
    not_kv_inputs = []
    for inp in ov_model.inputs:
        names = inp.get_names()
        is_kv = any(
            any(prefix in name for prefix in kv_prefixes)
            for name in names
        )
        if is_kv:
            kv_input_names.append(next(
                name for name in names
                if any(prefix in name for prefix in kv_prefixes)
            ))
        else:
            not_kv_inputs.append(inp)

    kv_output_names = []
    for out in ov_model.outputs:
        names = out.get_names()
        for name in names:
            if any(prefix in name for prefix in kv_output_prefixes):
                kv_output_names.append(name)
                break

    batch_dim = 0
    # Only fuse beam reorder for KV cache inputs (not conv/recurrent)
    fuse_cache_reorder(ov_model, not_kv_inputs, kv_input_names, batch_dim)
    # Only make KV cache stateful (not conv/recurrent)
    make_stateful(ov_model, not_kv_inputs, kv_input_names, kv_output_names, batch_dim)
