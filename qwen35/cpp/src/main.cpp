#include "hybrid_model.h"
#include <iostream>
#include <string>
#include <chrono>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model-dir PATH       Model directory (default: models/qwen35/Qwen3.5-0.8B-hybrid)\n"
              << "  --prompt TEXT           Input prompt (default: \"The capital of France is\")\n"
              << "  --max-tokens N          Maximum new tokens to generate (default: 50)\n"
              << "  --attn-past-seq N       NPU attention KV cache static size (default: 256)\n"
              << "  --prefill-chunk-size N  Prefill chunk size (default: 16, 1=token-by-token)\n"
              << "  --tokenizers-lib PATH   Path to openvino_tokenizers shared library\n"
              << "  --latency               Use PERFORMANCE_HINT: LATENCY (default: off)\n"
              << "  --no-gdn-prefill        Skip chunkwise GDN prefill (reduces GPU memory)\n"
              << "  --mtp-steps N           MTP speculative decode draft steps (default: 0 = disabled)\n"
              << "  --help                  Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string model_dir = "models/qwen35/Qwen3.5-0.8B-hybrid";
    std::string prompt = "The capital of France is";
    int max_tokens = 50;
    int attn_past_seq = 256;
    int prefill_chunk_size = 16;
    std::string tokenizers_lib;
    bool use_latency_hint = false;
    bool no_gdn_prefill = false;
    int mtp_steps = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--attn-past-seq" && i + 1 < argc) {
            attn_past_seq = std::stoi(argv[++i]);
        } else if (arg == "--prefill-chunk-size" && i + 1 < argc) {
            prefill_chunk_size = std::stoi(argv[++i]);
        } else if (arg == "--tokenizers-lib" && i + 1 < argc) {
            tokenizers_lib = argv[++i];
        } else if (arg == "--latency") {
            use_latency_hint = true;
        } else if (arg == "--no-gdn-prefill") {
            no_gdn_prefill = true;
        } else if (arg == "--mtp-steps" && i + 1 < argc) {
            mtp_steps = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== Qwen3.5 Hybrid GPU+NPU Inference (C++) ===\n"
              << "Model dir:           " << model_dir << "\n"
              << "Prompt:              \"" << prompt << "\"\n"
              << "Max new tokens:      " << max_tokens << "\n"
              << "Attn past seq:       " << attn_past_seq << "\n"
              << "Prefill chunk size:  " << prefill_chunk_size << "\n";
    if (!tokenizers_lib.empty()) {
        std::cout << "Tokenizers lib:      " << tokenizers_lib << "\n";
    }
    std::cout << "Latency hint:        " << (use_latency_hint ? "ON" : "OFF") << "\n";
    std::cout << "MTP steps:           " << mtp_steps << (mtp_steps > 0 ? " (speculative)" : " (disabled)") << "\n";
    std::cout << "===============================================\n\n";

    try {
        auto t_start = std::chrono::steady_clock::now();

        Qwen35HybridModel model(model_dir, attn_past_seq, prefill_chunk_size, tokenizers_lib, use_latency_hint, no_gdn_prefill, mtp_steps);

        auto t_loaded = std::chrono::steady_clock::now();
        double load_sec = std::chrono::duration<double>(t_loaded - t_start).count();
        std::cout << "Model loaded in " << load_sec << " s\n\n";

        std::string output = model.generate(prompt, max_tokens);

        auto t_done = std::chrono::steady_clock::now();
        double gen_sec = std::chrono::duration<double>(t_done - t_loaded).count();
        double total_sec = std::chrono::duration<double>(t_done - t_start).count();

        std::cout << "\n--- Generated output ---\n" << output << "\n";
        std::cout << "--- Timing ---\n"
                  << "Load:     " << load_sec << " s\n"
                  << "Generate: " << gen_sec << " s\n"
                  << "Total:    " << total_sec << " s\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
