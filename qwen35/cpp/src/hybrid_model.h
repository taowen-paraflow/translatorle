#ifndef HYBRID_MODEL_H
#define HYBRID_MODEL_H

#include <string>
#include <vector>
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
/// GDN blocks run on GPU (stateful), Attention blocks on NPU (explicit I/O),
/// Head on GPU.
class Qwen35HybridModel {
public:
    /// @param model_dir      Directory containing the 13 hybrid IR files
    /// @param attn_past_seq  Static KV cache size for NPU attention blocks (default 256)
    /// @param tokenizers_lib Path to openvino_tokenizers shared library (.dll/.so)
    Qwen35HybridModel(const std::string& model_dir,
                       int attn_past_seq = 256,
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
    void reset();

    std::vector<float> forward(const std::vector<int64_t>& token_ids);

    void run_gdn_block(int block_idx, std::vector<float>& hidden, int seq_len);
    void run_attn_block(int block_idx, std::vector<float>& hidden, int seq_len);

    std::vector<float> build_attn_mask(int actual_cache_len, int padded_cache_len, int seq_len);
    std::vector<float> pad_kv_cache(const std::vector<float>& cache, int num_heads,
                                     int current_len, int target_len, int head_dim, bool is_key);

    static std::shared_ptr<ov::Model> add_f32_output_conversion(std::shared_ptr<ov::Model> model);

    // Config
    ModelConfig cfg_;
    int attn_past_seq_;
    int past_length_;

    // OpenVINO
    ov::Core core_;

    // GDN blocks — GPU, stateful (conv/recurrent state persists in GPU memory)
    std::vector<ov::CompiledModel> gdn_models_;
    std::vector<ov::InferRequest> gdn_requests_;

    // Attention blocks — NPU, explicit I/O (KV cache padded to static size)
    std::vector<ov::CompiledModel> attn_models_;
    std::vector<ov::InferRequest> attn_requests_;

    // Head block — GPU, stateless
    ov::CompiledModel head_model_;
    ov::InferRequest head_request_;

    // Embedding table [vocab_size * hidden_size] stored flat, float32
    std::vector<float> embed_table_;

    // KV caches for NPU explicit mode [1, num_kv_heads, len, head_dim] flat
    struct KVCache {
        std::vector<float> key;
        std::vector<float> value;
        int len;  // current sequence length in cache (starts at 1 for dummy)
    };
    std::vector<KVCache> kv_caches_;

    // Tokenizer
    std::unique_ptr<OVTokenizer> tokenizer_;
};

#endif // HYBRID_MODEL_H
