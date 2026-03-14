"""Chunkwise parallel GDN recurrence cell for OpenVINO prefill.

Replaces the sequential Loop-based RecurrentAttentionCell with parallel
MatMul operations using the WY representation (Gated DeltaNet, ICLR 2025).

All operations use standard PyTorch ops that OpenVINO can trace directly:
MatMul, cumsum, exp, tril, diag_embed, element-wise multiply/add.
No Loop node, no solve_triangular.

Designed for dynamic shapes in OpenVINO:
- Uses torch.tril instead of torch.arange-based masks
- Uses torch.diag_embed + torch.ones_like for identity matrices
- Fixed Neumann series steps (supports T up to 256)
"""

import torch
import torch.nn as nn

# Fixed number of Neumann doubling steps.
# Supports T up to 2^MAX_NEUMANN_STEPS = 256 (matching MAX_CACHE_LEN).
# Extra steps for small T just multiply zeros — negligible overhead.
MAX_NEUMANN_STEPS = 8


class ChunkwiseRecurrentAttentionCell(nn.Module):
    """Drop-in replacement for RecurrentAttentionCell using chunkwise parallel algorithm.

    Processes all T tokens in parallel using WY representation instead of
    sequential Loop. Supports dynamic T via OV-friendly ops (tril, diag_embed).

    For multi-chunk scenarios, the caller processes chunks sequentially,
    passing the updated state from one chunk to the next.
    """

    def forward(self, q, k, v, g, beta, last_recurrent_state):
        """
        Args:
            q: (B, H, T, D_k) - queries
            k: (B, H, T, D_k) - keys (L2-normalized in Qwen3.5)
            v: (B, H, T, D_v) - values
            g: (B, H, T) - log-space gate (exp(g) = decay factor)
            beta: (B, H, T) - input gate / learning rate
            last_recurrent_state: (B, H, D_k, D_v) - initial recurrent state

        Returns:
            (output_flat, state_flat) concatenated - matching Loop version format
            output: (B, H, T, D_v) flattened
            state: (B, H, D_k, D_v) flattened
        """
        S = last_recurrent_state  # (B, H, D_k, D_v)

        # Step 1: Chunk-local cumulative sum of gates
        g_cumsum = g.cumsum(dim=-1)  # (B, H, T)

        # Step 2: Build A matrix (strictly lower triangular)
        KKT = torch.matmul(k, k.transpose(-1, -2))  # (B, H, T, T)
        g_diff = g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)  # (B, H, T, T)
        A_full = beta.unsqueeze(-1) * torch.exp(g_diff) * KKT  # (B, H, T, T)
        A = torch.tril(A_full, diagonal=-1)  # strictly lower triangular

        # Step 3: Neumann series inverse (I + A)^{-1}
        T_inv = self._neumann_inverse(A)

        # Step 4: Compute W and U
        beta_g = (beta * torch.exp(g_cumsum)).unsqueeze(-1)  # (B, H, T, 1)
        W = torch.matmul(T_inv, beta_g * k)  # (B, H, T, D_k)
        U = torch.matmul(T_inv, beta.unsqueeze(-1) * v)  # (B, H, T, D_v)

        # Step 5: Corrected values using initial state S
        WS = torch.matmul(W, S)  # (B, H, T, D_v)
        v_new = U - WS

        # Step 6: Output computation
        # Inter-chunk: Q @ S * exp(g_cumsum)
        inter = torch.matmul(q, S) * torch.exp(g_cumsum).unsqueeze(-1)

        # Intra-chunk: (Q @ K^T * causal_gate_mask) @ v_new
        QKT = torch.matmul(q, k.transpose(-1, -2))  # (B, H, T, T)
        causal = torch.tril(torch.exp(g_diff), diagonal=0)  # lower tri with gating
        intra = torch.matmul(causal * QKT, v_new)  # (B, H, T, D_v)

        output = inter + intra  # (B, H, T, D_v)

        # Step 7: State update for next chunk
        g_last = g_cumsum[:, :, -1:]  # (B, H, 1)
        S_new = S * torch.exp(g_last).unsqueeze(-1)  # (B, H, D_k, D_v)
        g_scale = torch.exp(g_last.unsqueeze(-1) - g_cumsum.unsqueeze(-1))  # (B, H, T, 1)
        v_new_scaled = v_new * g_scale
        S_new = S_new + torch.matmul(k.transpose(-2, -1), v_new_scaled)  # (B, H, D_k, D_v)

        # Flatten and concatenate (matching Loop version output format)
        output_flat = output.reshape(-1)
        state_flat = S_new.reshape(-1)
        return torch.cat([output_flat, state_flat], dim=0)

    @staticmethod
    def _neumann_inverse(A):
        """Compute (I + A)^{-1} for strictly lower triangular A.

        Uses Neumann series with repeated squaring:
        (I + A)^{-1} = sum_{k=0}^{T-1} (-A)^k

        For strictly lower triangular A of size C x C, A^C = 0 (nilpotent),
        so the Neumann series is finite and exact.

        Uses MAX_NEUMANN_STEPS doubling steps (fixed for OV dynamic shapes).
        Extra steps for small T just multiply zeros — correct but negligible cost.

        All ops are OV-dynamic-shape-friendly:
        - torch.ones_like for shape-inheriting tensor creation
        - torch.tril for identity (tril(ones,0) - tril(ones,-1))

        Args:
            A: (*, T, T) strictly lower triangular matrix

        Returns:
            (*, T, T) inverse matrix
        """
        # Build identity matrix using OV-supported ops (no diag_embed)
        # tril(ones, 0) - tril(ones, -1) = 1 on diagonal, 0 elsewhere
        ones_mat = torch.ones_like(A)
        I_mat = torch.tril(ones_mat, diagonal=0) - torch.tril(ones_mat, diagonal=-1)

        neg_A = -A
        B_power = neg_A  # (-A)^1
        N = I_mat + neg_A  # I + (-A) = first 2 terms of Neumann series

        for _ in range(MAX_NEUMANN_STEPS - 1):
            B_power = torch.matmul(B_power, B_power)  # square: (-A)^{2^i}
            N = N + torch.matmul(B_power, N)  # accumulate next 2^i terms

        return N


class SingleStepRecurrentAttentionCell(nn.Module):
    """Loop-free GDN recurrence for S=1 decode.

    Expresses the single-step GDN recurrence as flat ops (MatMul, Exp, Multiply,
    Add) — no Loop node, no Neumann series. The GPU plugin compiles these into
    an efficient fused graph, eliminating the CPU-orchestrated Loop overhead.

    Drop-in replacement for RecurrentAttentionCell with identical interface.
    Works for any T but is optimized for T=1 (decode).
    """

    def forward(self, q, k, v, g, beta, last_recurrent_state):
        """
        Args:
            q: (B, H, T, D_k) - queries
            k: (B, H, T, D_k) - keys (L2-normalized in Qwen3.5)
            v: (B, H, T, D_v) - values
            g: (B, H, T) - log-space gate (exp(g) = decay factor)
            beta: (B, H, T) - input gate / learning rate
            last_recurrent_state: (B, H, D_k, D_v) - initial recurrent state

        Returns:
            (output_flat, state_flat) concatenated - matching Loop version format
            output: (B, H, T, D_v) flattened
            state: (B, H, D_k, D_v) flattened
        """
        B, H, T, D_k = q.shape
        D_v = v.shape[-1]
        state = last_recurrent_state  # (B, H, D_k, D_v)
        outputs = []

        # Unrolled loop — OV traces each iteration as flat ops (no Loop node).
        # For T=1 decode, this is a single iteration.  For T>1, OV duplicates
        # the subgraph T times (only correct when T matches trace-time T).
        for t in range(T):
            q_t = q[:, :, t]      # (B, H, D_k)
            k_t = k[:, :, t]      # (B, H, D_k)
            v_t = v[:, :, t]      # (B, H, D_v)
            g_t = g[:, :, t]      # (B, H)
            beta_t = beta[:, :, t]  # (B, H)

            # Gated decay: state *= exp(g)
            decay = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            state = state * decay

            # Retrieve: kv_mem = state^T @ k  (memory retrieval)
            k_unsq = k_t.unsqueeze(-1)           # (B, H, D_k, 1)
            kv_mem = (state * k_unsq).sum(dim=-2)  # (B, H, D_v)

            # Delta: (v - kv_mem) * beta
            delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)  # (B, H, D_v)

            # State update: rank-1 outer product
            state = state + k_unsq * delta.unsqueeze(-2)  # (B, H, D_k, D_v)

            # Query output: out = state^T @ q
            q_unsq = q_t.unsqueeze(-1)           # (B, H, D_k, 1)
            out_t = (state * q_unsq).sum(dim=-2)  # (B, H, D_v)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=2)  # (B, H, T, D_v)

        # Flatten and concatenate (matching Loop version output format)
        output_flat = output.reshape(-1)
        state_flat = state.reshape(-1)
        return torch.cat([output_flat, state_flat], dim=0)


def sequential_gdn(q, k, v, g, beta, initial_state):
    """Sequential per-token GDN recurrence (reference matching ov_ops.py Loop)."""
    B, H, T, D_k = q.shape
    D_v = v.shape[-1]
    state = initial_state.clone()  # (B, H, D_k, D_v)
    outputs = []
    for t in range(T):
        q_t = q[:, :, t]  # (B, H, D_k)
        k_t = k[:, :, t]  # (B, H, D_k)
        v_t = v[:, :, t]  # (B, H, D_v)
        g_t = g[:, :, t]  # (B, H)
        beta_t = beta[:, :, t]  # (B, H)

        # Gated decay
        state = state * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
        # Retrieve and compute delta
        k_unsq = k_t.unsqueeze(-1)  # (B, H, D_k, 1)
        kv_mem = (state * k_unsq).sum(dim=-2)  # (B, H, D_v)
        delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)  # (B, H, D_v)
        # State update (rank-1)
        state = state + k_unsq * delta.unsqueeze(-2)  # (B, H, D_k, D_v)
        # Query output
        q_unsq = q_t.unsqueeze(-1)  # (B, H, D_k, 1)
        out_t = (state * q_unsq).sum(dim=-2)  # (B, H, D_v)
        outputs.append(out_t)

    output = torch.stack(outputs, dim=2)  # (B, H, T, D_v)
    return output, state


if __name__ == "__main__":
    import numpy as np

    # ================================================================
    # Test 1: Numerical Verification
    # ================================================================
    print("=== Numerical Verification ===\n")

    cell = ChunkwiseRecurrentAttentionCell()
    all_pass = True

    test_configs = [
        # (B, H, T, D_k, D_v, normalize_keys, dtype, atol, rtol, desc)
        (1, 16, 1, 128, 128, True, torch.float32, 1e-4, 1e-3, "single token"),
        (1, 16, 5, 128, 128, True, torch.float32, 1e-4, 1e-3, "short (5 tok)"),
        (1, 16, 44, 128, 128, True, torch.float32, 1e-4, 1e-3, "typical (44 tok)"),
        (1, 16, 64, 128, 128, True, torch.float32, 1e-4, 1e-3, "exact 64 tok"),
        (1, 1, 3, 2, 2, False, torch.float64, 1e-12, 1e-10, "no key norm tiny f64"),
        (1, 4, 16, 4, 4, False, torch.float32, 1e-4, 1e-3, "no key norm small D"),
    ]

    for B, H, T, D_k, D_v, normalize_keys, dtype, atol, rtol, desc in test_configs:
        torch.manual_seed(42)
        q = torch.randn(B, H, T, D_k, dtype=dtype)
        k = torch.randn(B, H, T, D_k, dtype=dtype)
        v = torch.randn(B, H, T, D_v, dtype=dtype)
        g = torch.randn(B, H, T, dtype=dtype) * 0.1
        beta = torch.sigmoid(torch.randn(B, H, T, dtype=dtype))
        initial_state = torch.randn(B, H, D_k, D_v, dtype=dtype) * 0.01

        if normalize_keys:
            k = torch.nn.functional.normalize(k, dim=-1)

        # Sequential reference
        out_seq, state_seq = sequential_gdn(q, k, v, g, beta, initial_state)

        # Chunkwise parallel
        with torch.no_grad():
            result = cell(q, k, v, g, beta, initial_state)

        # Split concatenated output
        output_size = B * H * T * D_v
        state_size = B * H * D_k * D_v
        out_chunk = result[:output_size].reshape(B, H, T, D_v)
        state_chunk = result[output_size:].reshape(B, H, D_k, D_v)

        out_diff = (out_seq - out_chunk).abs().max().item()
        state_diff = (state_seq - state_chunk).abs().max().item()
        out_match = torch.allclose(out_seq, out_chunk, atol=atol, rtol=rtol)
        state_match = torch.allclose(state_seq, state_chunk, atol=atol, rtol=rtol)

        status = "PASS" if (out_match and state_match) else "FAIL"
        dtype_str = "f64" if dtype == torch.float64 else "f32"
        print(f"  [{status}] {desc}: T={T}, {dtype_str}, "
              f"out_diff={out_diff:.2e}, state_diff={state_diff:.2e}")
        if not (out_match and state_match):
            all_pass = False

    # ================================================================
    # Test 1b: SingleStepRecurrentAttentionCell Verification
    # ================================================================
    print("\n=== SingleStep Cell Verification ===\n")

    ss_cell = SingleStepRecurrentAttentionCell()

    ss_configs = [
        (1, 16, 1, 128, 128, True, torch.float32, 1e-5, 1e-4, "T=1 full dims"),
        (1, 16, 1, 128, 128, False, torch.float32, 1e-5, 1e-4, "T=1 no key norm"),
        (1, 1, 1, 4, 4, False, torch.float64, 1e-12, 1e-10, "T=1 tiny f64"),
    ]

    for B, H, T, D_k, D_v, normalize_keys, dtype, atol, rtol, desc in ss_configs:
        torch.manual_seed(42)
        q = torch.randn(B, H, T, D_k, dtype=dtype)
        k = torch.randn(B, H, T, D_k, dtype=dtype)
        v = torch.randn(B, H, T, D_v, dtype=dtype)
        g = torch.randn(B, H, T, dtype=dtype) * 0.1
        beta = torch.sigmoid(torch.randn(B, H, T, dtype=dtype))
        initial_state = torch.randn(B, H, D_k, D_v, dtype=dtype) * 0.01

        if normalize_keys:
            k = torch.nn.functional.normalize(k, dim=-1)

        out_seq, state_seq = sequential_gdn(q, k, v, g, beta, initial_state)

        with torch.no_grad():
            result = ss_cell(q, k, v, g, beta, initial_state)

        output_size = B * H * T * D_v
        out_ss = result[:output_size].reshape(B, H, T, D_v)
        state_ss = result[output_size:].reshape(B, H, D_k, D_v)

        out_diff = (out_seq - out_ss).abs().max().item()
        state_diff = (state_seq - state_ss).abs().max().item()
        out_match = torch.allclose(out_seq, out_ss, atol=atol, rtol=rtol)
        state_match = torch.allclose(state_seq, state_ss, atol=atol, rtol=rtol)

        status = "PASS" if (out_match and state_match) else "FAIL"
        dtype_str = "f64" if dtype == torch.float64 else "f32"
        print(f"  [{status}] {desc}: {dtype_str}, "
              f"out_diff={out_diff:.2e}, state_diff={state_diff:.2e}")
        if not (out_match and state_match):
            all_pass = False

    print()
    if all_pass:
        print("ALL NUMERICAL TESTS PASSED!")
    else:
        print("SOME NUMERICAL TESTS FAILED!")

    # ================================================================
    # Test 2: OpenVINO Traceability + Dynamic Shapes
    # ================================================================
    print("\n=== OpenVINO Trace Test ===\n")
    try:
        import openvino as ov
        from openvino import Dimension, Symbol

        def make_ov_inputs(B, H, T, D, normalize_keys=True):
            torch.manual_seed(123)
            k_raw = torch.randn(B, H, T, D)
            if normalize_keys:
                k_raw = torch.nn.functional.normalize(k_raw, dim=-1)
            return (
                torch.randn(B, H, T, D),
                k_raw,
                torch.randn(B, H, T, D),
                torch.randn(B, H, T) * 0.1,
                torch.sigmoid(torch.randn(B, H, T)),
                torch.randn(B, H, D, D) * 0.01,
            )

        def test_ov_trace(B, H, T, D, desc=""):
            """Test OV tracing and inference for given dimensions."""
            print(f"  --- {desc} (B={B}, H={H}, T={T}, D={D}) ---")
            cell = ChunkwiseRecurrentAttentionCell()
            example_inputs = make_ov_inputs(B, H, T, D)

            ov_model = ov.convert_model(cell, example_input=example_inputs)

            # Check no Loop nodes
            has_loop = any(op.get_type_name() == "Loop" for op in ov_model.get_ordered_ops())
            matmuls = sum(1 for op in ov_model.get_ordered_ops() if op.get_type_name() == "MatMul")
            print(f"    Ops: {len(list(ov_model.get_ops()))}, MatMul={matmuls}, Loop={has_loop}")

            # Verify OV inference matches PyTorch
            core = ov.Core()
            compiled = core.compile_model(ov_model, "CPU")
            request = compiled.create_infer_request()

            input_dict = {}
            for i, inp in enumerate(ov_model.inputs):
                input_dict[inp.get_any_name()] = example_inputs[i].numpy()
            result_ov = request.infer(input_dict)
            output_ov = list(result_ov.values())[0]

            with torch.no_grad():
                output_pt = cell(*example_inputs).numpy()

            diff = np.abs(output_ov - output_pt).max()
            success = diff < 1e-3
            print(f"    Max diff: {diff:.2e} {'SUCCESS' if success else 'FAIL'}")
            return success, has_loop, ov_model

        # Test with increasing complexity
        ok2, _, _ = test_ov_trace(1, 1, 2, 4, "tiny T=2")
        ok5, _, _ = test_ov_trace(1, 2, 5, 8, "small T=5")
        ok44, _, model44 = test_ov_trace(1, 16, 44, 128, "full T=44")

        # ================================================================
        # Test 3: Dynamic Shapes
        # ================================================================
        print("\n=== Dynamic Shape Test ===\n")
        # Trace with T=2, set dynamic shapes, test with T=5 and T=44
        cell = ChunkwiseRecurrentAttentionCell()
        D = 8
        trace_inputs = make_ov_inputs(1, 2, 2, D)
        ov_model = ov.convert_model(cell, example_input=trace_inputs)

        # Set dynamic T dimension
        seq_sym = Symbol()
        for i, inp in enumerate(ov_model.inputs):
            ps = inp.partial_shape
            if ps.rank.get_length() == 4:  # q, k, v: (B, H, T, D)
                s_dim = Dimension(-1)
                s_dim.set_symbol(seq_sym)
                ps[2] = s_dim
            elif ps.rank.get_length() == 3:  # g, beta: (B, H, T)
                s_dim = Dimension(-1)
                s_dim.set_symbol(seq_sym)
                ps[2] = s_dim
            # state (B, H, D, D) — no T dim, leave static
            inp.get_node().set_partial_shape(ps)
        ov_model.validate_nodes_and_infer_types()

        print("  Model traced with T=2, dynamic shapes set on T dim")
        for i, inp in enumerate(ov_model.inputs):
            print(f"    Input {i}: shape={inp.get_partial_shape()}")

        core = ov.Core()
        compiled = core.compile_model(ov_model, "CPU")

        for test_T in [2, 5, 16, 44]:
            test_inputs = make_ov_inputs(1, 2, test_T, D)
            request = compiled.create_infer_request()
            input_list = [x.numpy() for x in test_inputs]
            try:
                result = request.infer(input_list)
                output_ov = list(result.values())[0]

                with torch.no_grad():
                    output_pt = cell(*test_inputs).numpy()

                diff = np.abs(output_ov - output_pt).max()
                success = diff < 1e-3
                print(f"  T={test_T:3d}: OV shape={output_ov.shape}, diff={diff:.2e} {'OK' if success else 'FAIL'}")
            except Exception as e:
                print(f"  T={test_T:3d}: FAILED - {e}")

    except ImportError:
        print("  OpenVINO not available, skipping trace test")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
