// hybrid_model.cpp — Construction, config loading, model compilation, state init
#include "hybrid_model.h"
#include "tokenizer.h"
#include "npy_reader.h"
#include "utils.h"

#include <fstream>
#include <cstring>
#include <filesystem>

#include <nlohmann/json.hpp>
#include <openvino/pass/make_stateful.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

Qwen35HybridModel::Qwen35HybridModel(
    const std::string& model_dir,
    int attn_past_seq,
    int prefill_chunk_size,
    const std::string& tokenizers_lib,
    bool use_latency_hint,
    bool no_gdn_prefill,
    int mtp_steps)
    : attn_past_seq_(attn_past_seq),
      prefill_chunk_size_(prefill_chunk_size),
      past_length_(0),
      kv_total_(0),
      use_latency_hint_(use_latency_hint),
      no_gdn_prefill_(no_gdn_prefill),
      mtp_steps_(mtp_steps)
{
    cfg_ = load_config(model_dir);
    log("Config: hidden=" + std::to_string(cfg_.hidden_size) +
        " kv_heads=" + std::to_string(cfg_.num_kv_heads) +
        " head_dim=" + std::to_string(cfg_.head_dim) +
        " blocks=" + std::to_string(cfg_.num_blocks) +
        " prefill_chunk=" + std::to_string(prefill_chunk_size));

    // Enable model caching — NPU compilation is ~10s/block (60s total).
    // After first run, cached blobs make subsequent starts <1s.
    std::string cache_dir = model_dir + "/cache";
    core_.set_property(ov::cache_dir(cache_dir));
    log("Model cache: " + cache_dir);

    // Load order matters for thermal management on Lunar Lake iGPU:
    // NPU models first (doesn't heat GPU), then GPU models.
    // This gives the GPU cooling time between compilation and inference.
    load_attn_blocks(model_dir);         // NPU — no GPU heat
    load_gdn_blocks(model_dir);          // GPU — decode critical
    load_head(model_dir);                // GPU — decode critical
    if (!no_gdn_prefill) {
        load_gdn_prefill_blocks(model_dir);  // GPU — prefill only, loaded last
    } else {
        log("Skipping GDN prefill blocks (--no-gdn-prefill)");
    }
    load_embeddings(model_dir);
    if (mtp_steps_ > 0) {
        load_mtp_block(model_dir);
    }
    alloc_buffers();
    bind_decode_tensors();

    // Initialize tokenizer
    std::string tok_xml = model_dir + "/openvino_tokenizer.xml";
    if (fs::exists(tok_xml)) {
        if (!tokenizers_lib.empty()) {
            tokenizer_ = std::make_unique<OVTokenizer>(core_, model_dir, tokenizers_lib);
            log("Tokenizer loaded from OV IR");
        } else {
            log("WARNING: Tokenizer IR found but no --tokenizers-lib specified. "
                "Run with --tokenizers-lib path/to/openvino_tokenizers.dll");
        }
    } else {
        log("WARNING: No openvino_tokenizer.xml found. Run convert_tokenizer.py first.");
    }
}

Qwen35HybridModel::~Qwen35HybridModel() = default;

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

ModelConfig Qwen35HybridModel::load_config(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + model_dir + "/config.json");

    json root = json::parse(f);
    json cfg = root.contains("text_config") ? root["text_config"] : root;

    ModelConfig c{};
    c.hidden_size = cfg["hidden_size"].get<int>();
    c.num_kv_heads = cfg["num_key_value_heads"].get<int>();
    int num_attn_heads = cfg["num_attention_heads"].get<int>();
    c.head_dim = cfg.value("head_dim", c.hidden_size / num_attn_heads);
    c.num_v_heads = cfg["linear_num_value_heads"].get<int>();
    c.k_head_dim = cfg["linear_key_head_dim"].get<int>();
    c.v_head_dim = cfg["linear_value_head_dim"].get<int>();
    int num_k_heads = cfg["linear_num_key_heads"].get<int>();
    c.conv_dim = num_k_heads * c.k_head_dim * 2 + c.num_v_heads * c.v_head_dim;
    c.conv_kernel = cfg["linear_conv_kernel_dim"].get<int>();
    c.vocab_size = cfg.value("vocab_size", 248320);

    c.num_blocks = 0;
    while (fs::exists(model_dir + "/gdn_block_" + std::to_string(c.num_blocks) + ".xml"))
        c.num_blocks++;

    return c;
}

// ---------------------------------------------------------------------------
// FP32 output conversion (GPU/NPU output FP16 by default)
// ---------------------------------------------------------------------------

std::shared_ptr<ov::Model> Qwen35HybridModel::add_f32_output_conversion(
    std::shared_ptr<ov::Model> model)
{
    ov::preprocess::PrePostProcessor ppp(model);
    for (size_t i = 0; i < model->outputs().size(); ++i) {
        ppp.output(i).tensor().set_element_type(ov::element::f32);
    }
    return ppp.build();
}

// ---------------------------------------------------------------------------
// GDN block loading (GPU, stateful)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_gdn_blocks(const std::string& model_dir) {
    // Prefer S1 (no-Loop) blocks for decode if available.
    // S1 blocks only support S=1, so they require prefill blocks for the
    // prefill phase (S>1). If prefill blocks won't be loaded (--no-gdn-prefill),
    // the Loop-based blocks are used as fallback since they support any S.
    bool s1_exists = fs::exists(model_dir + "/gdn_s1_block_0.xml");
    bool prefill_exists = fs::exists(model_dir + "/gdn_prefill_block_0.xml");
    bool use_s1 = s1_exists && prefill_exists && !no_gdn_prefill_;

    if (s1_exists && !use_s1) {
        log("S1 GDN blocks found but not using: prefill blocks required for S>1 fallback");
    }

    std::string block_type = use_s1 ? "S1 no-Loop" : "Loop-based";
    log("Compiling " + std::to_string(cfg_.num_blocks) + " GDN " + block_type +
        " blocks on GPU (stateful)...");
    auto t0 = std::chrono::steady_clock::now();

    // 3 layers per block × (conv + rec) = 6 state pairs
    std::map<std::string, std::string> state_map;
    for (int j = 0; j < 3; ++j) {
        state_map["in_conv" + std::to_string(j)] = "out_conv" + std::to_string(j);
        state_map["in_rec" + std::to_string(j)] = "out_rec" + std::to_string(j);
    }

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        std::string xml = use_s1
            ? model_dir + "/gdn_s1_block_" + std::to_string(i) + ".xml"
            : model_dir + "/gdn_block_" + std::to_string(i) + ".xml";
        auto model = core_.read_model(xml);
        bool make_stateful = !use_s1 || (mtp_steps_ == 0);
        if (make_stateful) {
            // Stateful: state lives in GPU VRAM (faster ~15%, but can't read state back)
            ov::pass::MakeStateful(state_map).run_on_model(model);
        }
        // S1 + MTP: keep explicit I/O (state in host memory, enables batch verify)
        model = add_f32_output_conversion(model);
        ov::AnyMap gpu_config = {
            {ov::hint::num_requests.name(), uint32_t(1)}
        };
        if (use_latency_hint_) {
            gpu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        auto compiled = core_.compile_model(model, "GPU", gpu_config);
        gdn_models_.push_back(compiled);
        gdn_requests_.push_back(compiled.create_infer_request());
    }

    // has_gdn_s1_ = true means S1 blocks with explicit I/O (state in host memory).
    // Only enable when MTP is active (needed for batch verify).
    // When MTP is off, S1 blocks are stateful (faster, ~15% less GDN overhead).
    has_gdn_s1_ = use_s1 && (mtp_steps_ > 0);
    if (has_gdn_s1_) {
        // S1 explicit I/O: allocate shared state buffers (used by both decode and prefill)
        init_gdn_prefill_states();
    } else {
        init_gdn_states();
    }
    log("  GDN compilation: " + std::to_string(elapsed_ms(t0)) + " ms" +
        (use_s1 ? " (S1 no-Loop)" : " (Loop-based)"));
}

// ---------------------------------------------------------------------------
// GDN prefill block loading (GPU, explicit I/O, chunkwise parallel)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_gdn_prefill_blocks(const std::string& model_dir) {
    std::string test_xml = model_dir + "/gdn_prefill_block_0.xml";
    if (!fs::exists(test_xml)) {
        log("No chunkwise GDN prefill blocks found, using Loop-based prefill");
        return;
    }

    log("Compiling " + std::to_string(cfg_.num_blocks) +
        " GDN prefill blocks on GPU (explicit I/O, FP32)...");
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        std::string xml = model_dir + "/gdn_prefill_block_" + std::to_string(i) + ".xml";
        auto model = core_.read_model(xml);
        model = add_f32_output_conversion(model);

        // Force FP32 inference: the Neumann series (7 matrix squarings)
        // accumulates catastrophic precision loss in FP16.
        ov::AnyMap gpu_config = {
            {ov::hint::num_requests.name(), uint32_t(1)},
            {"INFERENCE_PRECISION_HINT", "f32"}
        };
        if (use_latency_hint_) {
            gpu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        auto compiled = core_.compile_model(model, "GPU", gpu_config);
        gdn_prefill_models_.push_back(compiled);
        gdn_prefill_requests_.push_back(compiled.create_infer_request());
    }

    has_gdn_prefill_ = true;
    init_gdn_prefill_states();
    log("  GDN prefill compilation: " + std::to_string(elapsed_ms(t0)) + " ms");
}

// ---------------------------------------------------------------------------
// Attention block loading (NPU, explicit I/O with ScatterUpdate-3 fixed KV)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_attn_blocks(const std::string& model_dir) {
    int N = cfg_.num_blocks;
    int H = cfg_.num_kv_heads;
    int P = attn_past_seq_;
    int D = cfg_.head_dim;
    int C = prefill_chunk_size_;
    int hidden = cfg_.hidden_size;

    // --- Decode models (S=1) ---
    log("Compiling " + std::to_string(N) + " Attn blocks on NPU (S=1, KV=" +
        std::to_string(P) + ")...");
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < N; ++i) {
        std::string xml = model_dir + "/attn_block_" + std::to_string(i) + ".xml";
        auto model = core_.read_model(xml);

        // Reshape to static shapes for NPU — 6 inputs (ScatterUpdate-3 IR):
        //   0: hidden [1,1,hidden_size]
        //   1: position_ids [3,1,1]
        //   2: key_cache [1,H,P,D]
        //   3: value_cache [1,H,P,D]
        //   4: cache_position [1]
        //   5: attention_mask [1,1,1,P]
        std::map<std::string, ov::PartialShape> shapes;
        auto inputs = model->inputs();
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            std::string name = inputs[idx].get_any_name();
            switch (idx) {
                case 0: shapes[name] = {1, 1, hidden}; break;
                case 1: shapes[name] = {3, 1, 1}; break;
                case 2: case 3: shapes[name] = {1, H, P, D}; break;
                case 4: shapes[name] = {1}; break;
                case 5: shapes[name] = {1, 1, 1, P}; break;
            }
        }
        model->reshape(shapes);
        model = add_f32_output_conversion(model);

        ov::AnyMap npu_config = {
            {"NPU_COMPILER_TYPE", "PREFER_PLUGIN"},
            {ov::hint::num_requests.name(), uint32_t(1)}
        };
        if (use_latency_hint_) {
            npu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        auto compiled = core_.compile_model(model, "NPU", npu_config);
        attn_models_.push_back(compiled);
        attn_requests_.push_back(compiled.create_infer_request());
    }
    log("  Attn decode compilation: " + std::to_string(elapsed_ms(t0)) + " ms");

    // --- Prefill models (descending powers of 2: 16, 8, 4, 2) ---
    if (C > 1) {
        // Collect chunk sizes: e.g. 16 -> {16, 8, 4, 2}
        std::vector<int> chunk_sizes;
        for (int cs = C; cs >= 2; cs /= 2)
            chunk_sizes.push_back(cs);

        std::string sizes_str;
        for (size_t si = 0; si < chunk_sizes.size(); ++si) {
            if (si > 0) sizes_str += ",";
            sizes_str += std::to_string(chunk_sizes[si]);
        }
        log("Compiling " + std::to_string(N) + " prefill Attn blocks on NPU for S=[" +
            sizes_str + "]...");
        t0 = std::chrono::steady_clock::now();

        for (int cs : chunk_sizes) {
            std::vector<ov::CompiledModel> models;
            std::vector<ov::InferRequest> requests;
            for (int i = 0; i < N; ++i) {
                std::string xml = model_dir + "/attn_block_" + std::to_string(i) + ".xml";
                auto model = core_.read_model(xml);

                std::map<std::string, ov::PartialShape> shapes;
                auto inputs = model->inputs();
                for (size_t idx = 0; idx < inputs.size(); ++idx) {
                    std::string name = inputs[idx].get_any_name();
                    switch (idx) {
                        case 0: shapes[name] = {1, cs, hidden}; break;
                        case 1: shapes[name] = {3, 1, cs}; break;
                        case 2: case 3: shapes[name] = {1, H, P, D}; break;
                        case 4: shapes[name] = {cs}; break;
                        case 5: shapes[name] = {1, 1, cs, P}; break;
                    }
                }
                model->reshape(shapes);
                model = add_f32_output_conversion(model);

                ov::AnyMap npu_config = {
                    {"NPU_COMPILER_TYPE", "PREFER_PLUGIN"},
                    {ov::hint::num_requests.name(), uint32_t(1)}
                };
                if (use_latency_hint_) {
                    npu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
                }
                auto compiled = core_.compile_model(model, "NPU", npu_config);
                models.push_back(compiled);
                requests.push_back(compiled.create_infer_request());
            }
            attn_prefill_models_[cs] = std::move(models);
            attn_prefill_requests_[cs] = std::move(requests);
        }
        log("  Attn prefill compilation (" + std::to_string(chunk_sizes.size()) +
            " sizes): " + std::to_string(elapsed_ms(t0)) + " ms");
    }

    init_kv_caches();
}

// ---------------------------------------------------------------------------
// Head block loading (GPU, stateless)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_head(const std::string& model_dir) {
    log("Compiling Head on GPU...");
    auto t0 = std::chrono::steady_clock::now();

    auto model = core_.read_model(model_dir + "/head.xml");
    model = add_f32_output_conversion(model);
    ov::AnyMap gpu_config = {
        {ov::hint::num_requests.name(), uint32_t(1)}
    };
    if (use_latency_hint_) {
        gpu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    }
    head_model_ = core_.compile_model(model, "GPU", gpu_config);
    head_request_ = head_model_.create_infer_request();

    log("  Head compilation: " + std::to_string(elapsed_ms(t0)) + " ms");
}

// ---------------------------------------------------------------------------
// Embedding loading
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_embeddings(const std::string& model_dir) {
    log("Loading embed_tokens.npy...");
    auto t0 = std::chrono::steady_clock::now();

    std::vector<size_t> shape;
    embed_table_ = load_npy_fp16_as_fp32(model_dir + "/embed_tokens.npy", shape);

    log("  Embeddings: [" + std::to_string(shape[0]) + ", " + std::to_string(shape[1]) +
        "] loaded in " + std::to_string(elapsed_ms(t0)) + " ms");
}

// ---------------------------------------------------------------------------
// KV cache initialization (fixed-size, zero-filled)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::init_kv_caches() {
    int N = cfg_.num_blocks;
    int H = cfg_.num_kv_heads;
    int P = attn_past_seq_;
    int D = cfg_.head_dim;
    kv_total_ = H * P * D;

    kv_key_tensors_.clear();
    kv_value_tensors_.clear();
    for (int i = 0; i < N; ++i) {
        ov::Tensor key(ov::element::f32,
                       {1, static_cast<size_t>(H),
                        static_cast<size_t>(P),
                        static_cast<size_t>(D)});
        ov::Tensor val(ov::element::f32,
                       {1, static_cast<size_t>(H),
                        static_cast<size_t>(P),
                        static_cast<size_t>(D)});
        std::memset(key.data<float>(), 0, kv_total_ * sizeof(float));
        std::memset(val.data<float>(), 0, kv_total_ * sizeof(float));
        kv_key_tensors_.push_back(std::move(key));
        kv_value_tensors_.push_back(std::move(val));
    }
}

// ---------------------------------------------------------------------------
// MTP block loading (GPU, explicit I/O KV cache)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::load_mtp_block(const std::string& model_dir) {
    std::string mtp_xml = model_dir + "/mtp_block.xml";
    if (!fs::exists(mtp_xml)) {
        log("WARNING: mtp_block.xml not found, MTP disabled");
        has_mtp_ = false;
        mtp_steps_ = 0;
        return;
    }

    log("Compiling MTP block on GPU...");
    auto t0 = std::chrono::steady_clock::now();

    auto model = core_.read_model(mtp_xml);
    model = add_f32_output_conversion(model);
    ov::AnyMap gpu_config = {
        {ov::hint::num_requests.name(), uint32_t(1)}
    };
    if (use_latency_hint_) {
        gpu_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    }
    mtp_model_ = core_.compile_model(model, "GPU", gpu_config);
    mtp_request_ = mtp_model_.create_infer_request();

    // MTP KV cache: [1, num_kv_heads, max_cache_len, head_dim]
    // For Qwen3.5-0.8B: num_kv_heads=2, head_dim=256, max_cache_len=attn_past_seq_
    int mtp_kv_heads = cfg_.num_kv_heads;  // 2
    int mtp_head_dim = cfg_.head_dim;      // 256
    int mtp_cache_len = attn_past_seq_;    // 256
    mtp_kv_total_ = mtp_kv_heads * mtp_cache_len * mtp_head_dim;

    mtp_kv_key_ = ov::Tensor(ov::element::f32,
        {1, static_cast<size_t>(mtp_kv_heads),
         static_cast<size_t>(mtp_cache_len),
         static_cast<size_t>(mtp_head_dim)});
    mtp_kv_value_ = ov::Tensor(ov::element::f32,
        {1, static_cast<size_t>(mtp_kv_heads),
         static_cast<size_t>(mtp_cache_len),
         static_cast<size_t>(mtp_head_dim)});
    std::memset(mtp_kv_key_.data<float>(), 0, mtp_kv_total_ * sizeof(float));
    std::memset(mtp_kv_value_.data<float>(), 0, mtp_kv_total_ * sizeof(float));
    mtp_past_length_ = 0;

    // Allocate MTP buffers
    int H = cfg_.hidden_size;
    mtp_hidden_buf_.resize(H, 0.0f);
    mtp_saved_hidden_.resize(H, 0.0f);
    mtp_embeds_buf_.resize(H, 0.0f);
    mtp_pos_buf_.resize(3, 0LL);
    mtp_cache_pos_buf_.resize(1, 0LL);
    mtp_mask_buf_.resize(mtp_cache_len, -65504.0f);

    // Create tensor wrappers
    mtp_hidden_tensor_ = ov::Tensor(ov::element::f32,
        {1, 1, static_cast<size_t>(H)}, mtp_hidden_buf_.data());
    mtp_embeds_tensor_ = ov::Tensor(ov::element::f32,
        {1, 1, static_cast<size_t>(H)}, mtp_embeds_buf_.data());
    mtp_pos_tensor_ = ov::Tensor(ov::element::i64, {3, 1, 1}, mtp_pos_buf_.data());
    mtp_cache_pos_tensor_ = ov::Tensor(ov::element::i64, {1}, mtp_cache_pos_buf_.data());
    mtp_mask_tensor_ = ov::Tensor(ov::element::f32,
        {1, 1, 1, static_cast<size_t>(mtp_cache_len)}, mtp_mask_buf_.data());

    // Bind MTP tensors
    mtp_request_.set_input_tensor(0, mtp_hidden_tensor_);   // in_hidden
    mtp_request_.set_input_tensor(1, mtp_embeds_tensor_);   // in_embeds
    mtp_request_.set_input_tensor(2, mtp_pos_tensor_);      // in_position_ids
    mtp_request_.set_input_tensor(3, mtp_kv_key_);          // in_mtp_key_cache
    mtp_request_.set_input_tensor(4, mtp_kv_value_);        // in_mtp_value_cache
    mtp_request_.set_input_tensor(5, mtp_cache_pos_tensor_); // in_cache_position
    mtp_request_.set_input_tensor(6, mtp_mask_tensor_);     // in_attention_mask

    // Allocate GDN state snapshots for rollback
    gdn_snapshots_.resize(cfg_.num_blocks);
    if (has_gdn_s1_) {
        // S1 explicit I/O: 6 states per block (conv0, rec0, conv1, rec1, conv2, rec2)
        for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
            gdn_snapshots_[blk].states.resize(6);
            for (int j = 0; j < 3; ++j) {
                gdn_snapshots_[blk].states[j * 2] = ov::Tensor(ov::element::f32,
                    gdn_prefill_conv_states_[blk][j].get_shape());
                gdn_snapshots_[blk].states[j * 2 + 1] = ov::Tensor(ov::element::f32,
                    gdn_prefill_rec_states_[blk][j].get_shape());
            }
        }
    } else {
        // Stateful: query GPU state vars for shapes
        for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
            auto states = gdn_requests_[blk].query_state();
            gdn_snapshots_[blk].states.resize(states.size());
            for (size_t j = 0; j < states.size(); ++j) {
                auto st = states[j].get_state();
                gdn_snapshots_[blk].states[j] = ov::Tensor(st.get_element_type(), st.get_shape());
            }
        }
    }

    // Load norm correction vector: model.norm.weight / mtp.norm.weight
    // This corrects the MTP output (which has mtp.norm) before feeding to
    // the head block (which applies model.norm + lm_head).
    std::string correction_path = model_dir + "/mtp_norm_correction.npy";
    if (fs::exists(correction_path)) {
        std::vector<size_t> shape;
        mtp_norm_correction_ = load_npy_fp32(correction_path, shape);
        log("  Loaded norm correction: " + std::to_string(mtp_norm_correction_.size()) + " elements");
    } else {
        log("  WARNING: mtp_norm_correction.npy not found, using CPU lm_head fallback");
    }

    has_mtp_ = true;
    log("  MTP compilation: " + std::to_string(elapsed_ms(t0)) + " ms");
    log("  MTP KV cache: [1, " + std::to_string(mtp_kv_heads) + ", " +
        std::to_string(mtp_cache_len) + ", " + std::to_string(mtp_head_dim) + "]");
    log("  GDN snapshots: " + std::to_string(cfg_.num_blocks) + " blocks, " +
        std::to_string(gdn_snapshots_[0].states.size()) + " states/block");
}

// ---------------------------------------------------------------------------
// Pre-allocate buffers and tensor wrappers
// ---------------------------------------------------------------------------

void Qwen35HybridModel::alloc_buffers() {
    int max_seq = std::max(1, prefill_chunk_size_);
    int H = cfg_.hidden_size;
    int P = attn_past_seq_;
    int C = prefill_chunk_size_;

    // Allocate raw buffers
    hidden_buf_.resize(max_seq * H, 0.0f);
    gdn_mask_buf_.assign(max_seq, 1LL);    // all ones, never changes
    pos_buf_.resize(3 * max_seq, 0LL);
    cache_pos_buf_.resize(max_seq, 0LL);
    attn_mask_buf_.resize(max_seq * P, 0.0f);

    // Create tensor wrappers for decode (S=1) — wraps raw buffers, no copy
    s1_hidden_ = ov::Tensor(ov::element::f32, {1, 1, static_cast<size_t>(H)}, hidden_buf_.data());
    s1_gdn_mask_ = ov::Tensor(ov::element::i64, {1, 1}, gdn_mask_buf_.data());
    s1_pos_ = ov::Tensor(ov::element::i64, {3, 1, 1}, pos_buf_.data());
    s1_cache_pos_ = ov::Tensor(ov::element::i64, {1}, cache_pos_buf_.data());
    s1_attn_mask_ = ov::Tensor(ov::element::f32,
                                {1, 1, 1, static_cast<size_t>(P)},
                                attn_mask_buf_.data());

    // Create tensor wrappers for prefill — one set per chunk size (16, 8, 4, 2)
    // All share the same underlying buffers (hidden_buf_, etc.) since only one
    // chunk size is active at a time during prefill.
    for (int cs = C; cs >= 2; cs /= 2) {
        size_t S = static_cast<size_t>(cs);
        sc_hidden_[cs] = ov::Tensor(ov::element::f32,
                                     {1, S, static_cast<size_t>(H)},
                                     hidden_buf_.data());
        sc_gdn_mask_[cs] = ov::Tensor(ov::element::i64,
                                        {1, S},
                                        gdn_mask_buf_.data());
        sc_pos_[cs] = ov::Tensor(ov::element::i64,
                                   {3, 1, S},
                                   pos_buf_.data());
        sc_cache_pos_[cs] = ov::Tensor(ov::element::i64,
                                         {S},
                                         cache_pos_buf_.data());
        sc_attn_mask_[cs] = ov::Tensor(ov::element::f32,
                                         {1, 1, S, static_cast<size_t>(P)},
                                         attn_mask_buf_.data());
    }
}

// ---------------------------------------------------------------------------
// GDN state initialization
// ---------------------------------------------------------------------------

void Qwen35HybridModel::init_gdn_states() {
    // S1 explicit I/O: states are in gdn_prefill_*_states_ buffers, no GPU state vars
    if (has_gdn_s1_) return;

    int conv_size = cfg_.conv_dim * cfg_.conv_kernel;
    int rec_size = cfg_.num_v_heads * cfg_.k_head_dim * cfg_.v_head_dim;

    for (auto& req : gdn_requests_) {
        for (auto& s : req.query_state()) {
            std::string name = s.get_name();
            if (name.find("conv") != std::string::npos) {
                ov::Tensor t(ov::element::f32,
                             {1, static_cast<size_t>(cfg_.conv_dim),
                              static_cast<size_t>(cfg_.conv_kernel)});
                std::memset(t.data<float>(), 0, conv_size * sizeof(float));
                s.set_state(t);
            } else if (name.find("rec") != std::string::npos) {
                ov::Tensor t(ov::element::f32,
                             {1, static_cast<size_t>(cfg_.num_v_heads),
                              static_cast<size_t>(cfg_.k_head_dim),
                              static_cast<size_t>(cfg_.v_head_dim)});
                std::memset(t.data<float>(), 0, rec_size * sizeof(float));
                s.set_state(t);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GDN prefill state initialization (explicit buffers, zero-filled)
// ---------------------------------------------------------------------------

void Qwen35HybridModel::init_gdn_prefill_states() {
    // Guard: only allocate once (shared by decode S1 + prefill)
    if (!gdn_prefill_conv_states_.empty()) return;

    int conv_dim = cfg_.conv_dim;
    int conv_kernel = cfg_.conv_kernel;
    int num_v = cfg_.num_v_heads;
    int kd = cfg_.k_head_dim;
    int vd = cfg_.v_head_dim;

    gdn_prefill_conv_states_.resize(cfg_.num_blocks);
    gdn_prefill_rec_states_.resize(cfg_.num_blocks);

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        gdn_prefill_conv_states_[i].resize(3);
        gdn_prefill_rec_states_[i].resize(3);
        for (int j = 0; j < 3; ++j) {
            gdn_prefill_conv_states_[i][j] = ov::Tensor(ov::element::f32,
                {1, static_cast<size_t>(conv_dim), static_cast<size_t>(conv_kernel)});
            std::memset(gdn_prefill_conv_states_[i][j].data<float>(), 0,
                        conv_dim * conv_kernel * sizeof(float));

            gdn_prefill_rec_states_[i][j] = ov::Tensor(ov::element::f32,
                {1, static_cast<size_t>(num_v), static_cast<size_t>(kd),
                 static_cast<size_t>(vd)});
            std::memset(gdn_prefill_rec_states_[i][j].data<float>(), 0,
                        num_v * kd * vd * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer chunkwise prefill states to stateful decode GDN blocks
// ---------------------------------------------------------------------------

void Qwen35HybridModel::transfer_prefill_states_to_decode() {
    // S1 explicit I/O: decode and prefill share the same state buffers, no transfer needed
    if (has_gdn_s1_) return;

    for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
        for (auto& s : gdn_requests_[blk].query_state()) {
            std::string name = s.get_name();
            for (int j = 0; j < 3; ++j) {
                if (name.find("conv" + std::to_string(j)) != std::string::npos) {
                    s.set_state(gdn_prefill_conv_states_[blk][j]);
                    break;
                }
                if (name.find("rec" + std::to_string(j)) != std::string::npos) {
                    s.set_state(gdn_prefill_rec_states_[blk][j]);
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-bind decode tensors (S=1) — call once after alloc_buffers()
// Avoids 55+ set_input/output_tensor calls per decode step.
// Only position_ids, cache_position, and attn_mask content change per step.
// ---------------------------------------------------------------------------

void Qwen35HybridModel::bind_decode_tensors() {
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        auto& gdn_req = gdn_requests_[i];
        gdn_req.set_input_tensor(0, s1_hidden_);
        gdn_req.set_input_tensor(1, s1_gdn_mask_);
        gdn_req.set_output_tensor(0, s1_hidden_);

        if (has_gdn_s1_) {
            // S1 explicit I/O: bind state tensors (shared with prefill).
            // Output states point to same buffer as input: GPU copies in before
            // processing, writes out after. Same pattern as hidden zero-copy.
            for (int j = 0; j < 3; ++j) {
                gdn_req.set_input_tensor(2 + j * 2, gdn_prefill_conv_states_[i][j]);
                gdn_req.set_input_tensor(3 + j * 2, gdn_prefill_rec_states_[i][j]);
                gdn_req.set_output_tensor(1 + j * 2, gdn_prefill_conv_states_[i][j]);
                gdn_req.set_output_tensor(2 + j * 2, gdn_prefill_rec_states_[i][j]);
            }
        }

        auto& attn_req = attn_requests_[i];
        attn_req.set_input_tensor(0, s1_hidden_);
        attn_req.set_input_tensor(1, s1_pos_);
        attn_req.set_input_tensor(2, kv_key_tensors_[i]);
        attn_req.set_input_tensor(3, kv_value_tensors_[i]);
        attn_req.set_input_tensor(4, s1_cache_pos_);
        attn_req.set_input_tensor(5, s1_attn_mask_);
    }
    head_request_.set_input_tensor(0, s1_hidden_);
}

// ---------------------------------------------------------------------------
// Reset all states for a new generation
// ---------------------------------------------------------------------------

void Qwen35HybridModel::reset() {
    if (has_gdn_s1_) {
        // S1 explicit I/O: zero shared state buffers (used by both decode and prefill)
        for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
            for (int j = 0; j < 3; ++j) {
                std::memset(gdn_prefill_conv_states_[blk][j].data<float>(), 0,
                            gdn_prefill_conv_states_[blk][j].get_byte_size());
                std::memset(gdn_prefill_rec_states_[blk][j].data<float>(), 0,
                            gdn_prefill_rec_states_[blk][j].get_byte_size());
            }
        }
    } else {
        init_gdn_states();
        if (has_gdn_prefill_) {
            // Prefill blocks have separate state buffers — zero them
            for (int blk = 0; blk < cfg_.num_blocks; ++blk) {
                for (int j = 0; j < 3; ++j) {
                    std::memset(gdn_prefill_conv_states_[blk][j].data<float>(), 0,
                                gdn_prefill_conv_states_[blk][j].get_byte_size());
                    std::memset(gdn_prefill_rec_states_[blk][j].data<float>(), 0,
                                gdn_prefill_rec_states_[blk][j].get_byte_size());
                }
            }
        }
    }
    // Reset KV caches to zeros (fixed-size, stays allocated)
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        std::memset(kv_key_tensors_[i].data<float>(), 0, kv_total_ * sizeof(float));
        std::memset(kv_value_tensors_[i].data<float>(), 0, kv_total_ * sizeof(float));
    }
    past_length_ = 0;

    // Reset MTP state
    if (has_mtp_) {
        reset_mtp_kv();
        spec_accepted_ = 0;
        spec_rejected_ = 0;
        spec_draft_ms_ = 0;
        spec_verify_ms_ = 0;
        spec_rollback_ms_ = 0;
    }
}
