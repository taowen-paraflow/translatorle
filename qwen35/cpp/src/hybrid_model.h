#ifndef HYBRID_MODEL_H
#define HYBRID_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <openvino/openvino.hpp>

class OVTokenizer;

struct ModelConfig {
    int hidden_size;
    int num_kv_heads;
    int head_dim;
    int num_v_heads;
    int k_head_dim;
    int v_head_dim;
    int conv_dim;        // linear_num_key_heads * linear_key_head_dim * 2 + linear_num_value_heads * linear_value_head_dim
    int conv_kernel;     // linear_conv_kernel_dim
    int vocab_size;
    int num_blocks;      // number of GDN+Attn block pairs (detected from files)
};

/// Hybrid GPU+NPU inference for Qwen3.5 using up to 19 subgraph IRs.
/// GDN decode blocks run on GPU (stateful Loop), GDN prefill blocks run on
/// GPU (explicit I/O, chunkwise parallel MatMul, no Loop), Attention blocks
/// on NPU (explicit I/O with fixed-size ScatterUpdate-3 KV cache), Head on GPU.
///
/// Performance optimizations over naive approach:
///   - CACHE_DIR for NPU/GPU model caching (60s→<1s startup)
///   - Fixed-size KV cache (no per-step padding/compacting)
///   - Pre-allocated tensor wrappers (zero allocation in hot loop)
///   - Chunked prefill (S=16 NPU models, skip Head for intermediate chunks)
///   - Chunkwise parallel GDN prefill (no Loop, ~2x prefill speedup)
///   - Return logits pointer (no 1MB copy per decode step)
class Qwen35HybridModel {
public:
    /// @param model_dir           Directory containing the 13 hybrid IR files
    /// @param attn_past_seq       Static KV cache size for NPU attention (default 256)
    /// @param prefill_chunk_size  Tokens per prefill chunk (default 16, 1=token-by-token)
    /// @param tokenizers_lib      Path to openvino_tokenizers shared library (.dll/.so)
    /// @param use_latency_hint    Whether to use PERFORMANCE_HINT: LATENCY (default false)
    Qwen35HybridModel(const std::string& model_dir,
                       int attn_past_seq = 256,
                       int prefill_chunk_size = 16,
                       const std::string& tokenizers_lib = "",
                       bool use_latency_hint = false,
                       bool no_gdn_prefill = false,
                       int mtp_steps = 0);

    ~Qwen35HybridModel();

    /// Generate text from a prompt using greedy decoding
    std::string generate(const std::string& prompt, int max_new_tokens = 100);

private:
    ModelConfig load_config(const std::string& model_dir);
    void load_gdn_blocks(const std::string& model_dir);
    void load_gdn_prefill_blocks(const std::string& model_dir);
    void load_attn_blocks(const std::string& model_dir);
    void load_head(const std::string& model_dir);
    void load_embeddings(const std::string& model_dir);
    void load_mtp_block(const std::string& model_dir);
    int run_mtp_draft(int token_id);
    int mtp_cpu_argmax(const float* logits);
    int mtp_cpu_lm_head_argmax(const float* hidden);
    void save_gdn_snapshot();
    void restore_gdn_snapshot();
    void reset_mtp_kv();
    void transfer_decode_to_prefill();
    /// Batch-verify draft tokens: GDN prefill + chunked attention in one pass.
    /// Returns (accepted_count, logits_ptr). Sets model state correctly.
    std::pair<int, const float*> batch_verify_draft(
        const int64_t* verify_tokens, int verify_len,
        const int64_t* draft_tokens, int draft_len);

    void init_gdn_states();
    void init_gdn_prefill_states();
    void init_kv_caches();
    void alloc_buffers();
    void reset();

    void run_gdn_prefill_block(int block_idx, ov::Tensor& hidden_tensor, ov::Tensor& mask_tensor);
    void transfer_prefill_states_to_decode();

    /// Run one forward pass. Returns pointer to last token's logits (vocab_size floats).
    /// Pointer valid until next forward() call. If run_head=false, returns nullptr.
    const float* forward(const int64_t* token_ids, int seq_len, bool run_head = true);

    /// Layer-major prefill: full-batch GDN (GPU) + chunked attention (NPU).
    /// Returns pointer to last token's logits. More efficient than calling
    /// forward() per chunk: fewer GDN dispatches, head runs once, Loop amortized.
    const float* prefill(const int64_t* token_ids, int prompt_len);

    void run_gdn_block(int block_idx, int seq_len);
    void run_attn_block(int block_idx, int seq_len, bool use_prefill);
    void fill_attn_mask(int seq_len);

    static std::shared_ptr<ov::Model> add_f32_output_conversion(std::shared_ptr<ov::Model> model);

    void bind_decode_tensors();

    // Config
    ModelConfig cfg_;
    int attn_past_seq_;
    int prefill_chunk_size_;
    int past_length_;
    int kv_total_;  // num_kv_heads * attn_past_seq * head_dim (cached for memcpy)
    bool use_latency_hint_;
    bool no_gdn_prefill_;

    // MTP speculative decoding
    int mtp_steps_ = 0;  // 0 = disabled
    ov::CompiledModel mtp_model_;
    ov::InferRequest mtp_request_;
    bool has_mtp_ = false;

    // MTP KV cache — explicit I/O (not stateful), 2 KV heads, head_dim=256, max_cache=256
    ov::Tensor mtp_kv_key_;    // [1, 2, 256, 256]
    ov::Tensor mtp_kv_value_;  // [1, 2, 256, 256]
    int mtp_past_length_ = 0;
    int mtp_kv_total_ = 0;    // num_kv_heads * max_cache * head_dim

    // MTP pre-allocated buffers
    std::vector<float> mtp_hidden_buf_;     // [hidden_size] MTP output
    std::vector<float> mtp_saved_hidden_;   // [hidden_size] saved from main model decode
    ov::Tensor mtp_hidden_tensor_;   // [1, 1, hidden_size] wrapping mtp_hidden_buf_
    ov::Tensor mtp_embeds_tensor_;   // [1, 1, hidden_size]
    ov::Tensor mtp_pos_tensor_;      // [3, 1, 1]
    ov::Tensor mtp_cache_pos_tensor_; // [1]
    ov::Tensor mtp_mask_tensor_;     // [1, 1, 1, 256]
    std::vector<float> mtp_embeds_buf_;     // [hidden_size]
    std::vector<int64_t> mtp_pos_buf_;      // [3]
    std::vector<int64_t> mtp_cache_pos_buf_; // [1]
    std::vector<float> mtp_mask_buf_;       // [256]
    std::vector<float> mtp_norm_correction_; // [hidden_size] = model.norm.w / mtp.norm.w

    // GDN state snapshots for speculative decode rollback
    struct GdnSnapshot {
        std::vector<ov::Tensor> states;  // all states for one GDN block
    };
    std::vector<GdnSnapshot> gdn_snapshots_;

    // Speculative decode statistics
    int spec_accepted_ = 0;
    int spec_rejected_ = 0;
    double spec_draft_ms_ = 0;
    double spec_verify_ms_ = 0;
    double spec_rollback_ms_ = 0;

    // Decode timing accumulators (filled during generate, printed at end)
    std::vector<double> decode_gdn_ms_;   // per-block GDN time
    std::vector<double> decode_attn_ms_;  // per-block Attn time
    double decode_head_ms_ = 0;
    int decode_steps_ = 0;

    // OpenVINO
    ov::Core core_;

    // GDN blocks — GPU, stateful (conv/recurrent state persists in GPU memory)
    std::vector<ov::CompiledModel> gdn_models_;
    std::vector<ov::InferRequest> gdn_requests_;

    // GDN prefill blocks — GPU, explicit I/O (chunkwise parallel, no Loop)
    std::vector<ov::CompiledModel> gdn_prefill_models_;
    std::vector<ov::InferRequest> gdn_prefill_requests_;
    bool has_gdn_prefill_ = false;
    bool has_gdn_s1_ = false;  // S1 no-Loop GDN decode blocks loaded

    // Explicit GDN state buffers for prefill (conv + rec per layer per block)
    // gdn_prefill_conv_states_[block][layer] — [1, conv_dim, conv_kernel]
    // gdn_prefill_rec_states_[block][layer]  — [1, num_v_heads, k_head_dim, v_head_dim]
    std::vector<std::vector<ov::Tensor>> gdn_prefill_conv_states_;
    std::vector<std::vector<ov::Tensor>> gdn_prefill_rec_states_;

    // Attention blocks — NPU, explicit I/O, S=1 (decode)
    std::vector<ov::CompiledModel> attn_models_;
    std::vector<ov::InferRequest> attn_requests_;

    // Attention blocks — NPU, explicit I/O, S=2/4/8/16 (prefill, descending powers of 2)
    std::map<int, std::vector<ov::CompiledModel>> attn_prefill_models_;
    std::map<int, std::vector<ov::InferRequest>> attn_prefill_requests_;

    // Head block — GPU, stateless
    ov::CompiledModel head_model_;
    ov::InferRequest head_request_;

    // Embedding table [vocab_size * hidden_size] stored flat, float32
    std::vector<float> embed_table_;

    // Fixed-size KV caches — ov::Tensor [1, H, P, D] per block (owns memory)
    std::vector<ov::Tensor> kv_key_tensors_;
    std::vector<ov::Tensor> kv_value_tensors_;

    // Pre-allocated buffers (reused every forward call, never reallocated)
    std::vector<float> hidden_buf_;         // max_seq * hidden_size
    std::vector<int64_t> gdn_mask_buf_;     // max_seq (all 1s)
    std::vector<int64_t> pos_buf_;          // 3 * max_seq
    std::vector<int64_t> cache_pos_buf_;    // max_seq
    std::vector<float> attn_mask_buf_;      // max_seq * attn_past_seq

    // Pre-allocated tensor wrappers — decode (S=1), no allocation in hot loop
    ov::Tensor s1_hidden_;      // [1, 1, hidden_size] wrapping hidden_buf_
    ov::Tensor s1_gdn_mask_;    // [1, 1] wrapping gdn_mask_buf_
    ov::Tensor s1_pos_;         // [3, 1, 1] wrapping pos_buf_
    ov::Tensor s1_cache_pos_;   // [1] wrapping cache_pos_buf_
    ov::Tensor s1_attn_mask_;   // [1, 1, 1, P] wrapping attn_mask_buf_

    // Pre-allocated tensor wrappers — prefill (S=2/4/8/16, one set per chunk size)
    std::map<int, ov::Tensor> sc_hidden_;      // [1, C, hidden_size] wrapping hidden_buf_
    std::map<int, ov::Tensor> sc_gdn_mask_;    // [1, C] wrapping gdn_mask_buf_
    std::map<int, ov::Tensor> sc_pos_;         // [3, 1, C] wrapping pos_buf_
    std::map<int, ov::Tensor> sc_cache_pos_;   // [C] wrapping cache_pos_buf_
    std::map<int, ov::Tensor> sc_attn_mask_;   // [1, 1, C, P] wrapping attn_mask_buf_

    // Tokenizer
    std::unique_ptr<OVTokenizer> tokenizer_;
};

#endif // HYBRID_MODEL_H
