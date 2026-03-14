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

/// Format a double as "123.4" (1 decimal place) — replaces verbose substr/find patterns
inline std::string fmt_ms(double val) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", val);
    return buf;
}

/// Format a percentage as "82.5" (1 decimal place)
inline std::string fmt_pct(double val) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", val);
    return buf;
}

#endif // UTILS_H
