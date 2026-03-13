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

/// Hybrid GPU+NPU inference for Qwen3.5 using 13 subgraph IRs.
/// GDN blocks run on GPU (stateful), Attention blocks on NPU (explicit I/O
/// with fixed-size ScatterUpdate-3 KV cache), Head on GPU.
///
/// Performance optimizations over naive approach:
///   - CACHE_DIR for NPU/GPU model caching (60s→<1s startup)
///   - Fixed-size KV cache (no per-step padding/compacting)
///   - Pre-allocated tensor wrappers (zero allocation in hot loop)
///   - Chunked prefill (S=16 NPU models, skip Head for intermediate chunks)
///   - Return logits pointer (no 1MB copy per decode step)
class Qwen35HybridModel {
public:
    /// @param model_dir           Directory containing the 13 hybrid IR files
    /// @param attn_past_seq       Static KV cache size for NPU attention (default 256)
    /// @param prefill_chunk_size  Tokens per prefill chunk (default 16, 1=token-by-token)
    /// @param tokenizers_lib      Path to openvino_tokenizers shared library (.dll/.so)
    Qwen35HybridModel(const std::string& model_dir,
                       int attn_past_seq = 256,
                       int prefill_chunk_size = 16,
                       const std::string& tokenizers_lib = "");

    ~Qwen35HybridModel();

    /// Generate text from a prompt using greedy decoding
    std::string generate(const std::string& prompt, int max_new_tokens = 100);

private:
    ModelConfig load_config(const std::string& model_dir);
    void load_gdn_blocks(const std::string& model_dir);
    void load_attn_blocks(const std::string& model_dir);
    void load_head(const std::string& model_dir);
    void load_embeddings(const std::string& model_dir);

    void init_gdn_states();
    void init_kv_caches();
    void alloc_buffers();
    void reset();

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

    // Config
    ModelConfig cfg_;
    int attn_past_seq_;
    int prefill_chunk_size_;
    int past_length_;
    int kv_total_;  // num_kv_heads * attn_past_seq * head_dim (cached for memcpy)

    // OpenVINO
    ov::Core core_;

    // GDN blocks — GPU, stateful (conv/recurrent state persists in GPU memory)
    std::vector<ov::CompiledModel> gdn_models_;
    std::vector<ov::InferRequest> gdn_requests_;

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
