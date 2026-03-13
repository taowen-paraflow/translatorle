"""Test chunkwise parallel GDN vs sequential reference."""
import torch
import time

def sequential_gdn(q, k, v, g, beta, initial_state):
    """Sequential per-token GDN recurrence (reference matching ov_ops.py Loop)."""
    B, H, T, D_k = q.shape
    D_v = v.shape[-1]
    state = initial_state.clone()  # (B, H, D_k, D_v)
    outputs = []
    for t in range(T):
        q_t = q[:, :, t]       # (B, H, D_k)
        k_t = k[:, :, t]       # (B, H, D_k)
        v_t = v[:, :, t]       # (B, H, D_v)
        g_t = g[:, :, t]       # (B, H)
        beta_t = beta[:, :, t] # (B, H)

        # Gated decay
        state = state * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
        # Retrieve and compute delta
        k_unsq = k_t.unsqueeze(-1)    # (B, H, D_k, 1)
        kv_mem = (state * k_unsq).sum(dim=-2)  # (B, H, D_v)
        delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)  # (B, H, D_v)
        # State update (rank-1)
        state = state + k_unsq * delta.unsqueeze(-2)  # (B, H, D_k, D_v)
        # Query output
        q_unsq = q_t.unsqueeze(-1)    # (B, H, D_k, 1)
        out_t = (state * q_unsq).sum(dim=-2)  # (B, H, D_v)
        outputs.append(out_t)

    output = torch.stack(outputs, dim=2)  # (B, H, T, D_v)
    return output, state


def chunkwise_gdn(q, k, v, g, beta, initial_state, chunk_size=64):
    """Chunkwise parallel GDN (WY representation)."""
    B, H, T, D_k = q.shape
    D_v = v.shape[-1]

    # Pad T to multiple of chunk_size
    pad_len = (chunk_size - T % chunk_size) % chunk_size
    if pad_len > 0:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad_len))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
        g = torch.nn.functional.pad(g, (0, pad_len))
        beta = torch.nn.functional.pad(beta, (0, pad_len))

    T_padded = T + pad_len
    NC = T_padded // chunk_size  # number of chunks
    C = chunk_size

    # Reshape to chunks: (B, H, NC, C, ...)
    Q_c = q.reshape(B, H, NC, C, D_k)
    K_c = k.reshape(B, H, NC, C, D_k)
    V_c = v.reshape(B, H, NC, C, D_v)
    G_c = g.reshape(B, H, NC, C)
    Beta_c = beta.reshape(B, H, NC, C)

    # Step 1: Chunk-local cumulative sum of gates
    G_cumsum = G_c.cumsum(dim=-1)  # (B, H, NC, C)

    # Step 2: Build A matrix (strictly lower triangular)
    # A[i,j] = -beta_i * exp(g_cumsum_i - g_cumsum_j) * (k_i @ k_j^T) for i > j
    KKT = torch.einsum('bhnid,bhnjd->bhnij', K_c, K_c)  # (B, H, NC, C, C)
    g_diff = G_cumsum.unsqueeze(-1) - G_cumsum.unsqueeze(-2)  # (B, H, NC, C, C)
    A = Beta_c.unsqueeze(-1) * torch.exp(g_diff) * KKT  # (B, H, NC, C, C)
    # Apply strictly lower triangular mask (diagonal = 0)
    tril_mask = torch.tril(torch.ones(C, C, device=q.device, dtype=q.dtype), diagonal=-1)
    A = A * tril_mask

    # Step 3: Triangular solve - compute (I + A)^{-1}
    # (I + A) is unit lower triangular, so we can use torch.linalg.solve_triangular
    I_mat = torch.eye(C, device=q.device, dtype=q.dtype).expand(B, H, NC, C, C)
    IpA = I_mat + A  # unit lower triangular
    # Solve (I+A) @ X = I => X = (I+A)^{-1}
    T_inv = torch.linalg.solve_triangular(IpA, I_mat, upper=False)

    # Step 4: Compute W and U
    # W = T_inv @ diag(beta * exp(g_cumsum)) @ K
    beta_g = (Beta_c * torch.exp(G_cumsum)).unsqueeze(-1)  # (B, H, NC, C, 1)
    W = torch.einsum('bhnij,bhnjd->bhnid', T_inv, beta_g * K_c)  # (B, H, NC, C, D_k)
    # U = T_inv @ diag(beta) @ V
    beta_u = Beta_c.unsqueeze(-1)  # (B, H, NC, C, 1)
    U = torch.einsum('bhnij,bhnjd->bhnid', T_inv, beta_u * V_c)  # (B, H, NC, C, D_v)

    # Step 5: Inter-chunk state propagation (sequential over chunks)
    S = initial_state.clone()  # (B, H, D_k, D_v)
    states = []
    v_new_list = []
    for c_idx in range(NC):
        states.append(S.clone())
        W_chunk = W[:, :, c_idx]  # (B, H, C, D_k)
        U_chunk = U[:, :, c_idx]  # (B, H, C, D_v)
        K_chunk = K_c[:, :, c_idx]  # (B, H, C, D_k)
        g_last = G_cumsum[:, :, c_idx, -1]  # (B, H) - cumsum at end of chunk
        g_pos = G_cumsum[:, :, c_idx]  # (B, H, C)

        # v_new = U - W @ S
        WS = torch.einsum('bhcd,bhdv->bhcv', W_chunk, S)  # (B, H, C, D_v)
        v_new = U_chunk - WS  # (B, H, C, D_v)
        v_new_list.append(v_new)

        # Decay state by exp(g_last)
        S = S * torch.exp(g_last).unsqueeze(-1).unsqueeze(-1)

        # Scale v_new by exp(g_last - g_pos) for accumulation
        scale = torch.exp(g_last.unsqueeze(-1) - g_pos).unsqueeze(-1)  # (B, H, C, 1)
        v_new_scaled = v_new * scale

        # State update: S += K^T @ v_new_scaled
        S = S + torch.einsum('bhck,bhcv->bhkv', K_chunk, v_new_scaled)

    # Step 6: Output computation
    output_chunks = []
    for c_idx in range(NC):
        Q_chunk = Q_c[:, :, c_idx]  # (B, H, C, D_k)
        K_chunk = K_c[:, :, c_idx]  # (B, H, C, D_k)
        h_c = states[c_idx]         # (B, H, D_k, D_v) - state at chunk start
        v_new_c = v_new_list[c_idx] # (B, H, C, D_v)
        g_c = G_cumsum[:, :, c_idx] # (B, H, C)

        # Inter-chunk: Q @ S * exp(g_pos)
        inter = torch.einsum('bhcd,bhdv->bhcv', Q_chunk, h_c)  # (B, H, C, D_v)
        inter = inter * torch.exp(g_c).unsqueeze(-1)  # apply cumulative gate

        # Intra-chunk: (Q @ K^T * causal_gate_mask) @ v_new
        QKT = torch.einsum('bhid,bhjd->bhij', Q_chunk, K_chunk)  # (B, H, C, C)
        g_diff_c = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)  # (B, H, C, C)
        causal = torch.tril(torch.exp(g_diff_c), diagonal=0)  # lower tri with gating
        intra = torch.einsum('bhij,bhjv->bhiv', causal * QKT, v_new_c)  # (B, H, C, D_v)

        output_chunks.append(inter + intra)

    output = torch.cat(output_chunks, dim=2)  # (B, H, T_padded, D_v)

    # Remove padding
    if pad_len > 0:
        output = output[:, :, :T]

    return output, S


def run_test(B, H, T, D_k, D_v, chunk_size, normalize_keys=True, dtype=torch.float32,
             atol=1e-4, rtol=1e-3, desc=""):
    """Run one comparison test."""
    torch.manual_seed(42)

    q = torch.randn(B, H, T, D_k, dtype=dtype)
    k = torch.randn(B, H, T, D_k, dtype=dtype)
    v = torch.randn(B, H, T, D_v, dtype=dtype)
    g = torch.randn(B, H, T, dtype=dtype) * 0.1  # small gates (log-space)
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=dtype))  # (0, 1)
    initial_state = torch.randn(B, H, D_k, D_v, dtype=dtype) * 0.01

    if normalize_keys:
        k = torch.nn.functional.normalize(k, dim=-1)

    # Sequential reference
    out_seq, state_seq = sequential_gdn(q, k, v, g, beta, initial_state)

    # Chunkwise parallel
    out_chunk, state_chunk = chunkwise_gdn(q, k, v, g, beta, initial_state, chunk_size=chunk_size)

    # Compare
    out_diff = (out_seq - out_chunk).abs().max().item()
    state_diff = (state_seq - state_chunk).abs().max().item()
    out_match = torch.allclose(out_seq, out_chunk, atol=atol, rtol=rtol)
    state_match = torch.allclose(state_seq, state_chunk, atol=atol, rtol=rtol)

    status = "PASS" if (out_match and state_match) else "FAIL"
    dtype_str = "f64" if dtype == torch.float64 else "f32"
    print(f"  [{status}] {desc}: T={T}, C={chunk_size}, {dtype_str}, "
          f"out_diff={out_diff:.2e}, state_diff={state_diff:.2e}")
    return out_match and state_match


def main():
    print("=== Chunkwise Parallel GDN Verification ===\n")

    B, H, D_k, D_v = 1, 16, 128, 128  # Match Qwen3.5-0.8B

    all_pass = True

    # Test 1: T=1 (single token, edge case)
    all_pass &= run_test(B, H, 1, D_k, D_v, chunk_size=64, desc="single token")

    # Test 2: T=5 (short, single chunk)
    all_pass &= run_test(B, H, 5, D_k, D_v, chunk_size=64, desc="short (5 tok)")

    # Test 3: T=44 (typical prompt, single chunk with C=64)
    all_pass &= run_test(B, H, 44, D_k, D_v, chunk_size=64, desc="typical (44 tok, C=64)")

    # Test 4: T=44 with smaller chunk (multiple chunks)
    all_pass &= run_test(B, H, 44, D_k, D_v, chunk_size=16, desc="multi-chunk (44 tok, C=16)")

    # Test 5: T=64 (exact chunk boundary)
    all_pass &= run_test(B, H, 64, D_k, D_v, chunk_size=64, desc="exact boundary (64 tok)")

    # Test 6: T=128 (two chunks)
    all_pass &= run_test(B, H, 128, D_k, D_v, chunk_size=64, desc="two chunks (128 tok)")

    # Test 7: T=44 with C=8 (many chunks)
    all_pass &= run_test(B, H, 44, D_k, D_v, chunk_size=8, desc="many chunks (44 tok, C=8)")

    # Test 8: Without key normalization, small dims (float64 for stability)
    # Unnormalized keys with large D_k make k^T k ~ O(D_k), causing ill-conditioned
    # triangular solve in the WY representation. Qwen3.5 always normalizes keys.
    # We test correctness with small dims and float64 to isolate algorithm bugs
    # from numerical conditioning issues.
    all_pass &= run_test(B, 1, 3, 2, 2, chunk_size=4, normalize_keys=False,
                          dtype=torch.float64, atol=1e-12, rtol=1e-10,
                          desc="no key norm tiny f64 (3 tok)")

    # Test 9: Without key normalization, small D_k (float32, moderate sequence)
    all_pass &= run_test(B, 4, 16, 4, 4, chunk_size=8, normalize_keys=False,
                          desc="no key norm small D (16 tok, D=4)")

    print()
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")

    # Performance comparison
    print("\n=== Performance Comparison ===\n")
    torch.manual_seed(42)
    T = 44
    q = torch.randn(B, H, T, D_k, dtype=torch.float32)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D_k, dtype=torch.float32), dim=-1)
    v = torch.randn(B, H, T, D_v, dtype=torch.float32)
    g = torch.randn(B, H, T, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float32))
    state = torch.randn(B, H, D_k, D_v, dtype=torch.float32) * 0.01

    # Warmup
    for _ in range(3):
        sequential_gdn(q, k, v, g, beta, state)
        chunkwise_gdn(q, k, v, g, beta, state, chunk_size=64)

    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        sequential_gdn(q, k, v, g, beta, state)
    t_seq = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        chunkwise_gdn(q, k, v, g, beta, state, chunk_size=64)
    t_chunk = (time.perf_counter() - t0) / N * 1000

    print(f"  Sequential: {t_seq:.1f}ms (T={T}, 1 GDN layer)")
    print(f"  Chunkwise:  {t_chunk:.1f}ms (T={T}, C=64)")
    print(f"  Ratio:      {t_seq/t_chunk:.2f}x")
    print(f"\n  Note: This is CPU PyTorch timing for 1 GDN layer.")
    print(f"  Real benefit is GPU parallelism of MatMul (not measurable here).")


if __name__ == "__main__":
    main()
