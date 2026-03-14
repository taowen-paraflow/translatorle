// Fused GDN (Gated Delta Network) recurrence kernel for single-step decode.
//
// Adapted from llama.cpp's gated_delta_net.comp (Vulkan -> OpenCL).
// Performs the delta rule recurrence in one kernel launch:
//   state_new = state * exp(g) + k * ((v - state^T @ k * exp(g)) * beta)
//   output = state_new^T @ q
//
// Work organization:
//   - 16 work groups (one per attention head), dispatched as global_x / local_x
//   - 128 work items per group (one per column of 128x128 state matrix)
//   - Each work item holds one column of state in private registers (128 floats)
//
// Memory layout (inputs):
//   q:         [1, 16, 1, 128] contiguous, 2048 floats total
//   k:         [1, 16, 1, 128] contiguous, 2048 floats total
//   v:         [1, 16, 1, 128] contiguous, 2048 floats total
//   g:         [1, 16, 1]      contiguous, 16 floats total (log-space decay)
//   beta:      [1, 16, 1]      contiguous, 16 floats total (sigmoid gate)
//   state_in:  [1, 16, 128, 128] contiguous, 262144 floats (read-only)
//
// Memory layout (single output — replaces original Concat):
//   output:    [264192] flat 1D = [attn_flat(2048) | state_flat(262144)]
//              First 2048 floats: attention output [16 × 128]
//              Next 262144 floats: updated state [16 × 128 × 128]
//
// Register pressure: 128 floats = 512 bytes per work item.
// Let the compiler choose optimal SIMD width for the target GPU.

#define H_DIM 16
#define D_DIM 128
#define STATE_SIZE (D_DIM * D_DIM)  // 128 * 128 = 16384 per head
#define ATTN_TOTAL (H_DIM * D_DIM)  // 16 * 128 = 2048 (attn_flat size)

__attribute__((reqd_work_group_size(D_DIM, 1, 1)))
__kernel void fused_gdn_recurrence(
    const __global float* restrict q,           // input port 0: [1, 16, 1, 128]
    const __global float* restrict k,           // input port 1: [1, 16, 1, 128]
    const __global float* restrict v,           // input port 2: [1, 16, 1, 128]
    const __global float* restrict g,           // input port 3: [1, 16, 1]
    const __global float* restrict beta,        // input port 4: [1, 16, 1]
    const __global float* restrict state_in,    // input port 5: [1, 16, 128, 128]
    __global float* restrict output             // output port 0: [264192] = [attn|state]
) {
    const int head = get_group_id(0);     // 0..15 (work group = head)
    const int col  = get_local_id(0);     // 0..127 (work item = state column)

    // --- Load k and q into shared memory for broadcast ---
    __local float s_k[D_DIM];
    __local float s_q[D_DIM];

    s_k[col] = k[head * D_DIM + col];
    s_q[col] = q[head * D_DIM + col];
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Load head-level scalars ---
    const float decay = exp(g[head]);       // exp(g) = per-head decay factor
    const float beta_val = beta[head];      // sigmoid gate (already applied in projection)

    // --- Load v for this column ---
    const float v_val = v[head * D_DIM + col];

    // --- Load state column into private registers ---
    const int state_base = head * STATE_SIZE;
    float s[D_DIM];
    for (int r = 0; r < D_DIM; r++) {
        s[r] = state_in[state_base + r * D_DIM + col];
    }

    // --- Step 1: kv = state_col^T @ k ---
    // Each work item: dot product of its 128-element state column with k vector
    float kv = 0.0f;
    for (int r = 0; r < D_DIM; r += 4) {
        kv += s[r]   * s_k[r]
            + s[r+1] * s_k[r+1]
            + s[r+2] * s_k[r+2]
            + s[r+3] * s_k[r+3];
    }

    // --- Step 2: Decay + delta rule ---
    // delta = (v[col] - decay * kv) * beta
    float delta_col = (v_val - decay * kv) * beta_val;

    // --- Step 3: Update state and compute output in one pass ---
    // new_state[r, col] = decay * state[r, col] + k[r] * delta
    // attn[col] = sum_r(new_state[r, col] * q[r])
    float attn_col = 0.0f;
    for (int r = 0; r < D_DIM; r += 4) {
        float s0 = decay * s[r]   + s_k[r]   * delta_col;
        float s1 = decay * s[r+1] + s_k[r+1] * delta_col;
        float s2 = decay * s[r+2] + s_k[r+2] * delta_col;
        float s3 = decay * s[r+3] + s_k[r+3] * delta_col;

        s[r]   = s0;
        s[r+1] = s1;
        s[r+2] = s2;
        s[r+3] = s3;

        attn_col += s0 * s_q[r]
                  + s1 * s_q[r+1]
                  + s2 * s_q[r+2]
                  + s3 * s_q[r+3];
    }

    // --- Write to single concatenated output [attn_flat | state_flat] ---
    // Attn: first ATTN_TOTAL floats
    output[head * D_DIM + col] = attn_col;

    // State: after ATTN_TOTAL offset
    for (int r = 0; r < D_DIM; r++) {
        output[ATTN_TOTAL + state_base + r * D_DIM + col] = s[r];
    }
}
