#include "gpu_model.h"
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>  // CommandLineToArgvW
#endif

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --model-dir PATH       Model directory (default: auto-detect)\n"
              << "  --prompt TEXT           Input prompt (default: \"The capital of France is\")\n"
              << "  --max-tokens N          Maximum new tokens (default: 50)\n"
              << "  --tokenizers-lib PATH   Path to openvino_tokenizers shared library\n"
              << "  --timing                Enable timing output\n"
              << "  --help                  Show this help\n";
}

static std::string detect_model_dir() {
    // Packaged layout: model/ next to the exe
    if (std::filesystem::exists("model/config.json"))
        return "model";
    // Development layout
    if (std::filesystem::exists("models/qwen35/Qwen3.5-0.8B-paro-ov-int4sym/config.json"))
        return "models/qwen35/Qwen3.5-0.8B-paro-ov-int4sym";
    return "model";
}

#ifdef _WIN32
// Convert wide string to UTF-8
static std::string wstr_to_utf8(const std::wstring& ws) {
    if (ws.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.data(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
    std::string s(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), (int)ws.size(), s.data(), len, nullptr, nullptr);
    return s;
}

// Get argv as UTF-8 strings via Windows wide-char API
static std::vector<std::string> get_utf8_args() {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    std::vector<std::string> args;
    if (wargv) {
        for (int i = 0; i < wargc; ++i)
            args.push_back(wstr_to_utf8(wargv[i]));
        LocalFree(wargv);
    }
    return args;
}
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // Use wide-char API to get proper UTF-8 arguments (argv uses ANSI code page)
    auto utf8_args = get_utf8_args();
    std::vector<const char*> utf8_ptrs;
    for (auto& a : utf8_args) utf8_ptrs.push_back(a.c_str());
    argc = (int)utf8_ptrs.size();
    argv = const_cast<char**>(utf8_ptrs.data());
#endif

    std::string model_dir;
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

    if (model_dir.empty())
        model_dir = detect_model_dir();

    // Auto-detect tokenizers lib next to exe (packaged layout)
    if (tokenizers_lib.empty() && std::filesystem::exists("openvino_tokenizers.dll"))
        tokenizers_lib = "openvino_tokenizers.dll";

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
