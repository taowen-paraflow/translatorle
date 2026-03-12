#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <iostream>
#include <chrono>

inline void log(const std::string& msg) {
    std::cerr << "[INFO] " << msg << "\n";
}

inline double elapsed_ms(std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

#endif // UTILS_H
