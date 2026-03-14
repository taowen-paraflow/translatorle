#include "gpu_model.h"
#include <iostream>
#include <string>
#include <chrono>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model-dir PATH       Model directory (default: models/qwen35/Qwen3.5-0.8B-paro-ov-int4sym)\n"
              << "  --prompt TEXT           Input prompt (default: \"The capital of France is\")\n"
              << "  --max-tokens N          Maximum new tokens (default: 50)\n"
              << "  --tokenizers-lib PATH   Path to openvino_tokenizers shared library\n"
              << "  --timing                Enable timing output\n"
              << "  --help                  Show this help\n";
}

int main(int argc, char* argv[]) {
    std::string model_dir = "models/qwen35/Qwen3.5-0.8B-paro-ov-int4sym";
    std::string prompt = "The capital of France is";
    int max_tokens = 50;
    std::string tokenizers_lib;
    bool timing = false;

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
        } else if (arg == "--tokenizers-lib" && i + 1 < argc) {
            tokenizers_lib = argv[++i];
        } else if (arg == "--timing") {
            timing = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== Qwen3.5 GPU-Only Inference (C++, Single IR, PARO) ===\n"
              << "Model dir:      " << model_dir << "\n"
              << "Prompt:         \"" << prompt << "\"\n"
              << "Max new tokens: " << max_tokens << "\n"
              << "=====================================================\n\n";

    try {
        auto t_start = std::chrono::steady_clock::now();

        Qwen35GPUModel model(model_dir, tokenizers_lib, timing);

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
