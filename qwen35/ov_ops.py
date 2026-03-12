#  Conversion rule for the RecurrentAttentionCellOp operation in a Torch graph.
#
#  The RecurrentAttentionCellOp appears as a result of replacing the
#  RecurrentAttentionCell torch.nn.Module via a registered ModuleExtension
#  in the OpenVINO PyTorch frontend.
#
#  Adapted from optimum-intel PR #1634 (_ov_ops.py).
#  Original code Copyright 2026 The HuggingFace Team, Apache-2.0 license.
#
#  Fix: replaced ScatterUpdate inside Loop body with get_concatenated_slices
#  to avoid GPU FP16 precision loss (OpenVINO issue #34532).

import numpy as np

import openvino as ov
import openvino.opset14 as ops


def convert_recurrent_attention_cell(context):
    """Build an OpenVINO Loop node for the gated-delta-rule recurrence.

    The Loop iterates over the sequence dimension, updating the recurrent
    state (B, H, D_k, D_v) and producing per-timestep attention output
    (B, H, 1, D_v) which gets concatenated via get_concatenated_slices.

    Inputs from context (matching RecurrentAttentionCell.forward args):
        0: query              (B, H, T, D_k)
        1: key                (B, H, T, D_k)
        2: value              (B, H, T, D_v)
        3: g                  (B, H, T)
        4: beta               (B, H, T)
        5: last_recurrent_state (B, H, D_k, D_v)

    Output:
        Flattened concat of [core_attn_out, last_recurrent_state].
    """
    query = context.get_input(0)
    key = context.get_input(1)
    value = context.get_input(2)
    g = context.get_input(3)
    beta = context.get_input(4)
    last_recurrent_state_old = context.get_input(5)

    # Determine seq_len from value shape[2]
    value_shape = ops.shape_of(value)
    const_two_out = ops.constant(2, dtype=np.int32)
    const_zero_out = ops.constant(0, dtype=np.int32)
    seq_len = ops.gather(value_shape, const_two_out, const_zero_out)

    # ---- Loop body parameters ----
    timestep_param = ops.parameter([], np.int32, "timestep")
    q_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "q_t")
    k_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "k_t")
    v_t_param = ops.parameter([-1, -1, 1, -1], np.float32, "v_t")
    g_t_param = ops.parameter([-1, -1, 1], np.float32, "g_t")
    beta_t_param = ops.parameter([-1, -1, 1], np.float32, "beta_t")
    last_recurrent_state_t = ops.parameter([-1, -1, -1, -1], np.float32, "last_recurrent_state_t")

    # ---- Loop body computation ----
    const_two = ops.constant(2, dtype=np.int32)
    q_t = ops.squeeze(q_t_param, const_two)
    k_t = ops.squeeze(k_t_param, const_two)
    v_t = ops.squeeze(v_t_param, const_two)
    const_minus_one = ops.constant(-1, dtype=np.int32)
    g_t = ops.unsqueeze(ops.exp(g_t_param), const_minus_one)
    beta_t = beta_t_param

    # Recurrent update: state = state * g + k^T * delta
    last_recurrent_state_in = ops.multiply(last_recurrent_state_t, g_t)
    const_minus_two = ops.constant(-2, dtype=np.int32)
    kv_mem = ops.multiply(last_recurrent_state_in, ops.unsqueeze(k_t, const_minus_one))
    kv_mem = ops.reduce_sum(kv_mem, const_minus_two, False)
    delta = ops.multiply(ops.subtract(v_t, kv_mem), beta_t)
    last_recurrent_state_delta = ops.multiply(
        ops.unsqueeze(k_t, const_minus_one), ops.unsqueeze(delta, const_minus_two)
    )
    last_recurrent_state_in = ops.add(last_recurrent_state_in, last_recurrent_state_delta)

    # Output: q^T * state -> (B, H, 1, D_v), keep seq dim for concatenation
    core_attn_update = ops.multiply(last_recurrent_state_in, ops.unsqueeze(q_t, const_minus_one))
    core_attn_update = ops.reduce_sum(core_attn_update, const_minus_two, True)

    last_recurrent_state_res = last_recurrent_state_in

    body_cond = ops.constant([True], dtype=bool)

    body_model = ov.Model(
        [body_cond, last_recurrent_state_res, core_attn_update],
        [
            timestep_param,
            q_t_param,
            k_t_param,
            v_t_param,
            g_t_param,
            beta_t_param,
            last_recurrent_state_t,
        ],
        "body_model",
    )

    # ---- Build Loop node ----
    seq_len = ops.convert(seq_len, "i32")
    loop = ops.loop(seq_len, ops.constant(True, dtype="bool"))
    loop.set_function(body_model)

    # Sliced inputs: iterate over seq_len dim (axis=2)
    loop.set_sliced_input(q_t_param, query, 0, 1, 1, -1, 2)
    loop.set_sliced_input(k_t_param, key, 0, 1, 1, -1, 2)
    loop.set_sliced_input(v_t_param, value, 0, 1, 1, -1, 2)
    loop.set_sliced_input(g_t_param, g, 0, 1, 1, -1, 2)
    loop.set_sliced_input(beta_t_param, beta, 0, 1, 1, -1, 2)

    # Merged input: carry recurrent state across iterations
    loop.set_merged_input(last_recurrent_state_t, last_recurrent_state_old, last_recurrent_state_res.output(0))
    loop.set_special_body_ports([0, 0])

    # Concatenate per-timestep outputs along axis 2: (B, H, 1, D_v) * T -> (B, H, T, D_v)
    core_attn_out_new = loop.get_concatenated_slices(core_attn_update.output(0), 0, 1, 1, -1, 2)
    # Get final recurrent state from last iteration
    last_recurrent_state_new = loop.get_iter_value(last_recurrent_state_res.output(0), -1)

    # Flatten and concat (workaround for single-output ModuleExtension)
    flatten_shape = ops.constant([-1], dtype=np.int32)
    core_attn_out_new = ops.reshape(core_attn_out_new, flatten_shape, False)
    last_recurrent_state_new = ops.reshape(last_recurrent_state_new, flatten_shape, False)

    final_output = ops.concat([core_attn_out_new, last_recurrent_state_new], 0)

    return [final_output.output(0)]


def convert_kv_cache_scatter_update(context):
    """Build ScatterUpdate-3 node for KV cache update at cache_position.

    Replaces KVCacheScatterUpdate module with axis-based ScatterUpdate.
    This is different from ScatterElementsUpdate (which torch.scatter produces)
    — ScatterUpdate-3 replaces entire slices along an axis.

    Inputs (from KVCacheScatterUpdate.forward args):
        0: cache          [B, H, MAX_S, D]  — the KV buffer
        1: new_kv         [B, H, S, D]      — new key/value states (S=1 for decode)
        2: cache_position [S]               — write position(s) along seq dim

    Output: updated cache [B, H, MAX_S, D]

    Equivalent to: cache[:, :, cache_position, :] = new_kv
    """
    cache = context.get_input(0)
    new_kv = context.get_input(1)
    cache_position = context.get_input(2)

    # ScatterUpdate-3: scatter_update(data, indices, updates, axis)
    # Updates slices of data along axis at positions given by indices
    axis = ops.constant(np.int64(2))  # seq dim
    result = ops.scatter_update(cache, cache_position, new_kv, axis)
    return result.outputs()


def convert_kv_cache_scatter_nd_update(context):
    """Build ScatterNDUpdate node for KV cache update at cache_position.

    Uses transpose trick: [B,H,S,D] -> [S,B,H,D] so dim 0 is the target,
    then ScatterNDUpdate with indices [[pos]], then transpose back.

    Inputs: same as convert_kv_cache_scatter_update
    Output: updated cache [B, H, MAX_S, D]
    """
    cache = context.get_input(0)
    new_kv = context.get_input(1)
    cache_position = context.get_input(2)

    # Transpose to [MAX_S, B, H, D] so seq dim is first
    perm_fwd = ops.constant(np.array([2, 0, 1, 3], dtype=np.int64))
    cache_t = ops.transpose(cache, perm_fwd)
    new_kv_t = ops.transpose(new_kv, perm_fwd)  # [S, B, H, D]

    # indices: [S] -> [S, 1] for ScatterNDUpdate (last_dim=1, index into dim 0)
    indices = ops.unsqueeze(cache_position, ops.constant(np.int64(1)))  # [S, 1]
    # Convert to i64 (ScatterNDUpdate requires i32 or i64 indices)
    indices = ops.convert(indices, "i64")

    result_t = ops.scatter_nd_update(cache_t, indices, new_kv_t)

    # Transpose back to [B, H, MAX_S, D]
    perm_back = ops.constant(np.array([1, 2, 0, 3], dtype=np.int64))
    result = ops.transpose(result_t, perm_back)

    return result.outputs()
