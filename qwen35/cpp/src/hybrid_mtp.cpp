// hybrid_mtp.cpp — MTP speculative decoding: draft, batch verify, snapshot/rollback
//
// Key optimization: batch verification processes all verify tokens in ONE pass
// through GDN prefill blocks + chunked NPU attention, instead of sequential
// S=1 forward passes. This amortizes kernel launch overhead and enables
// parallel GDN processing (chunkwise MatMul, no Loop).
//
// Performance model (mtp_steps=1, ~70% acceptance):
//   Sequential verify: 47ms decode + 11ms draft + 47ms verify = 105ms for 2.7 tokens = 25.7 tok/s
//   Batch verify:      47ms decode + 11ms draft + 30ms batch  = 88ms  for 2.7 tokens = 30.7 tok/s
#include "hybrid_model.h"
#include "utils.h"

#include <algorithm>
#include <cstring>

// ---------------------------------------------------------------------------
// Reset MTP KV cache (called before each draft round)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::reset_mtp_kv() {
    std::memset(mtp_kv_key_.data<float>(), 0, mtp_kv_total_ * sizeof(float));
    std::memset(mtp_kv_value_.data<float>(), 0, mtp_kv_total_ * sizeof(float));
    mtp_past_length_ = 0;
}

// ---------------------------------------------------------------------------
// GDN state snapshot — save all stateful variables for rollback
// ---------------------------------------------------------------------------

void Qwen35HybridModel::save_gdn_snapshot() {
    if (has_gdn_s1_) {
        // S1 explicit I/O: states are in host memory (gdn_prefill_*_states_)
        for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
            for (int j = 0; j < 3; ++j) {
                auto& src_conv = gdn_prefill_conv_states_[blk][j];
                auto& src_rec = gdn_prefill_rec_states_[blk][j];
                auto& dst_conv = gdn_snapshots_[blk].states[j * 2];
                auto& dst_rec = gdn_snapshots_[blk].states[j * 2 + 1];
                std::memcpy(dst_conv.data<float>(), src_conv.data<const float>(),
                            src_conv.get_byte_size());
                std::memcpy(dst_rec.data<float>(), src_rec.data<const float>(),
                            src_rec.get_byte_size());
            }
        }
        return;
    }
    // Stateful path: read from GPU state vars (may crash with internal element types)
    for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
        auto states = gdn_requests_[blk].query_state();
        for (size_t j = 0; j < states.size(); ++j) {
            auto src = states[j].get_state();
            auto shape = src.get_shape();
            size_t num_elements = 1;
            for (auto d : shape) num_elements *= d;
            size_t src_bytes = num_elements * sizeof(float);
            auto& dst = gdn_snapshots_[blk].states[j];
            if (dst.get_byte_size() != src_bytes) {
                dst = ov::Tensor(ov::element::f32, shape);
            }
            std::memcpy(dst.data<void>(), src.data<const void>(), src_bytes);
        }
    }
}

// ---------------------------------------------------------------------------
// GDN state restore — rollback to saved snapshot
// ---------------------------------------------------------------------------

void Qwen35HybridModel::restore_gdn_snapshot() {
    if (has_gdn_s1_) {
        // S1 explicit I/O: restore to host memory buffers
        for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
            for (int j = 0; j < 3; ++j) {
                auto& dst_conv = gdn_prefill_conv_states_[blk][j];
                auto& dst_rec = gdn_prefill_rec_states_[blk][j];
                auto& src_conv = gdn_snapshots_[blk].states[j * 2];
                auto& src_rec = gdn_snapshots_[blk].states[j * 2 + 1];
                std::memcpy(dst_conv.data<float>(), src_conv.data<const float>(),
                            dst_conv.get_byte_size());
                std::memcpy(dst_rec.data<float>(), src_rec.data<const float>(),
                            dst_rec.get_byte_size());
            }
        }
        return;
    }
    // Stateful path
    for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
        auto states = gdn_requests_[blk].query_state();
        for (size_t j = 0; j < states.size(); ++j) {
            states[j].set_state(gdn_snapshots_[blk].states[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer decode GDN states → prefill GDN state buffers
// (Reverse of transfer_prefill_states_to_decode)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::transfer_decode_to_prefill() {
    // S1 explicit I/O: decode and prefill share state buffers, no transfer needed
    if (has_gdn_s1_) return;

    // Stateful path: read from GPU state vars (crashes with internal element types)
    for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
        for (auto& s : gdn_requests_[blk].query_state()) {
            std::string name = s.get_name();
            auto src = s.get_state();
            for (int j = 0; j < 3; ++j) {
                if (name.find("conv" + std::to_string(j)) != std::string::npos) {
                    auto& dst = gdn_prefill_conv_states_[blk][j];
                    std::memcpy(dst.data<void>(), src.data<const void>(), dst.get_byte_size());
                    break;
                }
                if (name.find("rec" + std::to_string(j)) != std::string::npos) {
                    auto& dst = gdn_prefill_rec_states_[blk][j];
                    std::memcpy(dst.data<void>(), src.data<const void>(), dst.get_byte_size());
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU argmax over head output logits
// ---------------------------------------------------------------------------

int Qwen35HybridModel::mtp_cpu_argmax(const float* logits) {
    return static_cast<int>(
        std::max_element(logits, logits + cfg_.vocab_size) - logits);
}

// ---------------------------------------------------------------------------
// CPU lm_head projection + argmax (for MTP draft)
//
// Computes logits = hidden @ embed_tokens.T and returns argmax.
// embed_tokens is [vocab_size, hidden_size] stored row-major.
// For each vocab entry v: logits[v] = dot(hidden, embed_tokens[v, :])
// ---------------------------------------------------------------------------

int Qwen35HybridModel::mtp_cpu_lm_head_argmax(const float* hidden) {
    int V = cfg_.vocab_size;
    int H = cfg_.hidden_size;
    const float* emb = embed_table_.data();

    float best_val = -1e30f;
    int best_idx = 0;

    // Vectorize-friendly: process 4 elements at a time for inner loop.
    // MSVC auto-vectorizes this pattern with /O2 (SSE/AVX).
    int H4 = H & ~3;
    for (int v = 0; v < V; ++v) {
        const float* row = emb + static_cast<int64_t>(v) * H;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int h = 0; h < H4; h += 4) {
            acc0 += hidden[h]     * row[h];
            acc1 += hidden[h + 1] * row[h + 1];
            acc2 += hidden[h + 2] * row[h + 2];
            acc3 += hidden[h + 3] * row[h + 3];
        }
        float dot = acc0 + acc1 + acc2 + acc3;
        for (int h = H4; h < H; ++h) {
            dot += hidden[h] * row[h];
        }
        if (dot > best_val) {
            best_val = dot;
            best_idx = v;
        }
    }
    return best_idx;
}

// ---------------------------------------------------------------------------
// Run one MTP draft step
//
// Input:  token_id — the token just predicted (by main model or previous MTP step)
//         mtp_hidden_buf_ must contain the hidden state from the previous step
//
// Returns: predicted next-next token id
// ---------------------------------------------------------------------------

int Qwen35HybridModel::run_mtp_draft(int token_id) {
    int H = cfg_.hidden_size;
    int P = attn_past_seq_;

    // 1. Embedding lookup
    std::memcpy(mtp_embeds_buf_.data(),
                embed_table_.data() + static_cast<int64_t>(token_id) * H,
                H * sizeof(float));

    // 2. Position IDs: use ACTUAL sequence position for RoPE, not MTP cache pos.
    //    RoPE encodes absolute position; using MTP cache pos (starting at 0)
    //    gives completely wrong rotation angles.
    int actual_pos = past_length_ + mtp_past_length_;
    for (int c = 0; c < 3; ++c)
        mtp_pos_buf_[c] = actual_pos;

    // 3. Cache position
    mtp_cache_pos_buf_[0] = mtp_past_length_;

    // 4. Attention mask
    int valid = std::min(mtp_past_length_ + 1, P);
    std::memset(mtp_mask_buf_.data(), 0, valid * sizeof(float));
    std::fill(mtp_mask_buf_.data() + valid, mtp_mask_buf_.data() + P, -65504.0f);

    // 5. MTP inference (inputs bound in load_mtp_block)
    mtp_request_.infer();

    // 6. Copy output KV cache (explicit I/O)
    std::memcpy(mtp_kv_key_.data<float>(),
                mtp_request_.get_output_tensor(1).data<const float>(),
                mtp_kv_total_ * sizeof(float));
    std::memcpy(mtp_kv_value_.data<float>(),
                mtp_request_.get_output_tensor(2).data<const float>(),
                mtp_kv_total_ * sizeof(float));

    // 7. Copy output hidden (for next MTP step)
    std::memcpy(mtp_hidden_buf_.data(),
                mtp_request_.get_output_tensor(0).data<const float>(),
                H * sizeof(float));
    mtp_past_length_++;

    // 8. Compute next token prediction
    //
    // MTP output has mtp.norm applied. Head block applies model.norm + lm_head.
    // Two strategies:
    //   (a) With norm correction: multiply by (model.norm.w / mtp.norm.w) to cancel
    //       mtp.norm and replace with model.norm, then use GPU head block (~3ms).
    //   (b) Without: CPU lm_head projection = hidden @ embed_tokens.T (~130ms).
    if (!mtp_norm_correction_.empty()) {
        // Apply norm correction: hidden[h] *= correction[h]
        for (int h = 0; h < H; ++h) {
            mtp_hidden_buf_[h] *= mtp_norm_correction_[h];
        }
        // Copy to hidden_buf_ for head block (s1_hidden_ wraps hidden_buf_)
        std::memcpy(hidden_buf_.data(), mtp_hidden_buf_.data(), H * sizeof(float));
        head_request_.set_input_tensor(0, s1_hidden_);
        head_request_.infer();
        return mtp_cpu_argmax(head_request_.get_output_tensor(0).data<const float>());
    } else {
        return mtp_cpu_lm_head_argmax(mtp_hidden_buf_.data());
    }
}

// ---------------------------------------------------------------------------
// Batch verify: process all verify tokens in one pass
//
// Uses GDN prefill blocks (chunkwise parallel, any S) + chunked NPU attention.
// Much cheaper than sequential S=1 forward passes for 2+ tokens.
//
// verify_tokens: [main_next, D[0], ..., D[N-1]]  length = verify_len
// draft_tokens:  [D[0], ..., D[N-1]]             length = draft_len
// verify_len = 1 + draft_len
//
// Returns: (accepted_count, logits_ptr)
//   - accepted_count: number of D[i] accepted by main model
//   - logits_ptr: logits for bonus token (from last checked position)
//   - Model state set correctly for continued decode
// ---------------------------------------------------------------------------

std::pair<int, const float*> Qwen35HybridModel::batch_verify_draft(
    const int64_t* verify_tokens, int verify_len,
    const int64_t* draft_tokens, int draft_len)
{
    int H = cfg_.hidden_size;
    int V = cfg_.vocab_size;
    int verify_start = past_length_;

    // 0. Save GDN snapshot for rollback on partial accept.
    // With S1 explicit I/O: decode and prefill share state buffers, so
    // batch verify will overwrite decode states. Need snapshot for rollback.
    save_gdn_snapshot();

    // 1. Copy decode GDN states → prefill buffers (no-op for S1 shared buffers)
    transfer_decode_to_prefill();

    // 2. Embedding lookup for all verify tokens
    std::vector<float> verify_hidden(verify_len * H);
    for (int i = 0; i < verify_len; ++i) {
        std::memcpy(verify_hidden.data() + i * H,
                    embed_table_.data() + verify_tokens[i] * H,
                    H * sizeof(float));
    }

    // 3. Tensor wrappers for GDN prefill
    std::vector<int64_t> verify_mask(verify_len, 1LL);
    ov::Tensor vh_tensor(ov::element::f32,
        {1, static_cast<size_t>(verify_len), static_cast<size_t>(H)},
        verify_hidden.data());
    ov::Tensor vm_tensor(ov::element::i64,
        {1, static_cast<size_t>(verify_len)},
        verify_mask.data());

    // 4. Compute attention chunks (decompose verify_len into powers of 2)
    struct Chunk { int start; int len; };
    std::vector<Chunk> chunks;
    {
        int pos = 0;
        int C = prefill_chunk_size_;
        while (pos < verify_len) {
            int remaining = verify_len - pos;
            int cs = C;
            while (cs > remaining) cs /= 2;
            if (cs < 1) cs = 1;
            chunks.push_back({pos, cs});
            pos += cs;
        }
    }

    // 5. Layer-major: GDN prefill + chunked attention (same as prefill path)
    for (int layer = 0; layer < cfg_.num_blocks; ++layer) {
        // GDN prefill: process all verify tokens at once (chunkwise parallel)
        run_gdn_prefill_block(layer, vh_tensor, vm_tensor);

        // Attention: chunked through NPU
        for (auto& chunk : chunks) {
            int cs = chunk.len;
            past_length_ = verify_start + chunk.start;

            std::memcpy(hidden_buf_.data(),
                        verify_hidden.data() + chunk.start * H,
                        cs * H * sizeof(float));

            bool use_prefill = (attn_prefill_requests_.count(cs) > 0);
            run_attn_block(layer, cs, use_prefill);

            std::memcpy(verify_hidden.data() + chunk.start * H,
                        hidden_buf_.data(),
                        cs * H * sizeof(float));
        }
    }
    past_length_ = verify_start + verify_len;

    // 6. Check each position: run head, compare prediction with next draft token
    int accepted = 0;
    const float* bonus_logits = nullptr;

    for (int i = 0; i < verify_len; ++i) {
        // Copy position i's hidden to hidden_buf_ for head
        std::memcpy(hidden_buf_.data(),
                    verify_hidden.data() + i * H,
                    H * sizeof(float));
        head_request_.set_input_tensor(0, s1_hidden_);
        head_request_.infer();
        decode_head_ms_ += 0;  // not counting verify head time in decode stats
        bonus_logits = head_request_.get_output_tensor(0).data<const float>();

        if (i < draft_len) {
            int pred = static_cast<int>(
                std::max_element(bonus_logits, bonus_logits + V) - bonus_logits);
            if (pred == static_cast<int>(draft_tokens[i])) {
                accepted++;
            } else {
                // Divergence at position i — bonus_logits has the correct prediction
                break;
            }
        }
        // i == verify_len-1: last position, bonus_logits has bonus token prediction
    }

    // 7. State management
    if (accepted == draft_len) {
        // Full accept: GDN states already correct (batch verify processed all tokens).
        // For S1 explicit I/O: shared buffers already updated.
        // For stateful: transfer prefill states → decode GPU state.
        transfer_prefill_states_to_decode();
        bind_decode_tensors();
    } else {
        // Partial/no accept: restore GDN states to pre-verify snapshot,
        // then re-process only the accepted tokens to update GDN + KV cache.
        restore_gdn_snapshot();
        past_length_ = verify_start;

        // Re-process [main_next, D[0], ..., D[accepted-1]] sequentially.
        // This sets correct GDN/KV state. The last forward gives bonus logits.
        int reprocess_len = 1 + accepted;
        for (int r = 0; r < reprocess_len; ++r) {
            int64_t tid = verify_tokens[r];
            bool last = (r == reprocess_len - 1);
            bonus_logits = forward(&tid, 1, /*run_head=*/last);
        }
    }

    return {accepted, bonus_logits};
}
