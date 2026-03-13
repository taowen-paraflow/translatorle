"""Direct single-token GDN recurrence cell for Loop-free S=1 decode.

For S=1 decode, the Loop in RecurrentAttentionCell runs exactly once.
This cell inlines the single iteration as flat ops, eliminating Loop
overhead (3+ host-GPU sync round-trips per Loop node).

With 18 Loop executions per decode token (18 GDN layers across 6 blocks),
that's 54+ unnecessary sync points removed per token.

All operations use standard PyTorch ops that OpenVINO traces directly:
Squeeze, Multiply, ReduceSum, Unsqueeze, Add, Exp, Concat, Reshape.
No Loop node, no ModuleExtension, no Neumann series.
"""

import torch
import torch.nn as nn


class S1RecurrentAttentionCell(nn.Module):
    """Drop-in replacement for RecurrentAttentionCell for S=1 (single token).

    Instead of building a Loop that iterates once, directly computes:
      1. state_decayed = state * exp(g)
      2. kv_mem = sum(state_decayed * k, dim=-2)
      3. delta = (v - kv_mem) * beta
      4. state_new = state_decayed + outer(k, delta)
      5. out = sum(state_new * q, dim=-2)

    Output format matches RecurrentAttentionCell: concat of flattened
    [attention_output, recurrent_state] for splitting in the caller.
    """

    def forward(self, q, k, v, g, beta, last_recurrent_state):
        """
        All inputs have S=1 (single token):
            q: (B, H, 1, D_k)
            k: (B, H, 1, D_k)
            v: (B, H, 1, D_v)
            g: (B, H, 1)
            beta: (B, H, 1)
            last_recurrent_state: (B, H, D_k, D_v)

        Returns: concat of [output_flat, state_flat]
            output: (B, H, 1, D_v) flattened
            state: (B, H, D_k, D_v) flattened
        """
        # Squeeze S=1 dimension for element-wise math
        q_t = q.squeeze(2)    # (B, H, D_k)
        k_t = k.squeeze(2)    # (B, H, D_k)
        v_t = v.squeeze(2)    # (B, H, D_v)

        # Step 1: Gated decay of recurrent state
        decay = torch.exp(g).unsqueeze(-1)  # (B, H, 1, 1)
        state = last_recurrent_state * decay  # (B, H, D_k, D_v)

        # Step 2: Retrieve from state: kv_mem = state @ k
        k_unsq = k_t.unsqueeze(-1)  # (B, H, D_k, 1)
        kv_mem = (state * k_unsq).sum(dim=-2)  # (B, H, D_v)

        # Step 3: Compute delta = (v - kv_mem) * beta
        delta = (v_t - kv_mem) * beta  # (B, H, D_v)

        # Step 4: Update state with rank-1 outer product: k^T * delta
        state = state + k_unsq * delta.unsqueeze(-2)  # (B, H, D_k, D_v)

        # Step 5: Query output: out = state @ q
        q_unsq = q_t.unsqueeze(-1)  # (B, H, D_k, 1)
        out = (state * q_unsq).sum(dim=-2)  # (B, H, D_v)

        # Restore S=1 dimension for shape consistency with caller
        out = out.unsqueeze(2)  # (B, H, 1, D_v)

        # Flatten and concat (matching RecurrentAttentionCell output format)
        output_flat = out.reshape(-1)
        state_flat = state.reshape(-1)
        return torch.cat([output_flat, state_flat], dim=0)


def sequential_s1_reference(q, k, v, g, beta, initial_state):
    """Reference S=1 computation for verification (matches ov_ops.py Loop for T=1)."""
    q_t = q[:, :, 0]    # (B, H, D_k)
    k_t = k[:, :, 0]    # (B, H, D_k)
    v_t = v[:, :, 0]    # (B, H, D_v)
    g_t = g[:, :, 0]    # (B, H)
    beta_t = beta[:, :, 0]  # (B, H)

    # Gated decay
    state = initial_state * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
    # Retrieve
    k_unsq = k_t.unsqueeze(-1)
    kv_mem = (state * k_unsq).sum(dim=-2)
    # Delta
    delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)
    # State update
    state = state + k_unsq * delta.unsqueeze(-2)
    # Query output
    q_unsq = q_t.unsqueeze(-1)
    out = (state * q_unsq).sum(dim=-2)

    return out.unsqueeze(2), state


if __name__ == "__main__":
    import numpy as np

    print("=== S1 GDN Cell Verification ===\n")

    cell = S1RecurrentAttentionCell()
    all_pass = True

    test_configs = [
        # (B, H, D_k, D_v, desc)
        (1, 16, 128, 128, "0.8B dims"),
        (1, 32, 128, 128, "4B dims"),
        (2, 16, 128, 128, "batch=2"),
        (1, 1, 4, 4, "tiny"),
    ]

    for B, H, D_k, D_v, desc in test_configs:
        torch.manual_seed(42)
        q = torch.randn(B, H, 1, D_k)
        k = torch.nn.functional.normalize(torch.randn(B, H, 1, D_k), dim=-1)
        v = torch.randn(B, H, 1, D_v)
        g = torch.randn(B, H, 1) * 0.1
        beta = torch.sigmoid(torch.randn(B, H, 1))
        initial_state = torch.randn(B, H, D_k, D_v) * 0.01

        # Reference
        out_ref, state_ref = sequential_s1_reference(q, k, v, g, beta, initial_state)

        # S1 cell
        with torch.no_grad():
            result = cell(q, k, v, g, beta, initial_state)

        # Split
        output_size = B * H * 1 * D_v
        out_s1 = result[:output_size].reshape(B, H, 1, D_v)
        state_s1 = result[output_size:].reshape(B, H, D_k, D_v)

        out_diff = (out_ref - out_s1).abs().max().item()
        state_diff = (state_ref - state_s1).abs().max().item()
        out_match = torch.allclose(out_ref, out_s1, atol=1e-6, rtol=1e-5)
        state_match = torch.allclose(state_ref, state_s1, atol=1e-6, rtol=1e-5)

        status = "PASS" if (out_match and state_match) else "FAIL"
        print(f"  [{status}] {desc}: B={B}, H={H}, D_k={D_k}, D_v={D_v}, "
              f"out_diff={out_diff:.2e}, state_diff={state_diff:.2e}")
        if not (out_match and state_match):
            all_pass = False

    print()
    if all_pass:
        print("ALL S1 TESTS PASSED!")
    else:
        print("SOME S1 TESTS FAILED!")

    # OpenVINO trace test
    print("\n=== OpenVINO Trace Test ===\n")
    try:
        import openvino as ov

        B, H, D_k, D_v = 1, 16, 128, 128
        torch.manual_seed(123)
        example = (
            torch.randn(B, H, 1, D_k),
            torch.nn.functional.normalize(torch.randn(B, H, 1, D_k), dim=-1),
            torch.randn(B, H, 1, D_v),
            torch.randn(B, H, 1) * 0.1,
            torch.sigmoid(torch.randn(B, H, 1)),
            torch.randn(B, H, D_k, D_v) * 0.01,
        )

        ov_model = ov.convert_model(cell, example_input=example)

        # Check no Loop nodes
        has_loop = any(op.get_type_name() == "Loop" for op in ov_model.get_ordered_ops())
        num_ops = len(list(ov_model.get_ops()))
        print(f"  Ops: {num_ops}, Loop: {has_loop}")

        if has_loop:
            print("  FAIL: Loop nodes found in S1 cell!")
        else:
            print("  PASS: No Loop nodes")

        # Verify OV inference matches PyTorch
        core = ov.Core()
        compiled = core.compile_model(ov_model, "CPU")
        request = compiled.create_infer_request()

        input_dict = {}
        for i, inp in enumerate(ov_model.inputs):
            input_dict[inp.get_any_name()] = example[i].numpy()
        result_ov = request.infer(input_dict)
        output_ov = list(result_ov.values())[0]

        with torch.no_grad():
            output_pt = cell(*example).numpy()

        diff = np.abs(output_ov - output_pt).max()
        print(f"  OV vs PyTorch max diff: {diff:.2e} {'PASS' if diff < 1e-4 else 'FAIL'}")

    except ImportError:
        print("  OpenVINO not available, skipping trace test")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
