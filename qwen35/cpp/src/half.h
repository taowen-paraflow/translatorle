#ifndef HALF_H
#define HALF_H

#include <cstdint>
#include <cstring>

// Minimal FP16 (IEEE 754 binary16) <-> FP32 conversion utilities.
// Header-only, no external dependencies.
//
// Layout of binary16:  1 sign | 5 exponent | 10 mantissa
// Layout of binary32:  1 sign | 8 exponent | 23 mantissa

inline float half_to_float(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 0x1;
    uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1F;
    uint32_t mant = static_cast<uint32_t>(h) & 0x3FF;

    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            // +/- zero
            f = sign << 31;
        } else {
            // Denormal: renormalize into a normal FP32 value.
            // The value is (-1)^sign * 2^(-14) * (mant / 1024).
            // Shift mantissa up until the implicit leading 1 appears.
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp++;
            }
            mant &= 0x3FF; // remove the leading 1
            // FP16 exponent bias = 15, FP32 exponent bias = 127.
            // Biased FP32 exponent = (1 - exp) + (127 - 15) + 1 = 113 - exp + 1
            f = (sign << 31) | ((127 - 15 + 1 - exp + 1) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // Inf or NaN. Preserve mantissa bits for NaN payload.
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        // Normal number. Re-bias the exponent from FP16 (bias 15) to FP32 (bias 127).
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

inline uint16_t float_to_half(float value) {
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));

    uint32_t sign = (f >> 31) & 0x1;
    int32_t  exp  = static_cast<int32_t>((f >> 23) & 0xFF) - 127; // unbiased
    uint32_t mant = f & 0x7FFFFF;

    uint16_t h;

    if (exp > 15) {
        // Overflow -> Inf (or NaN if original was NaN).
        if (exp == 128 && mant != 0) {
            // NaN: keep some mantissa bits so it stays NaN.
            h = static_cast<uint16_t>((sign << 15) | (0x1F << 10) | (mant >> 13));
            if ((h & 0x3FF) == 0) {
                h |= 1; // ensure mantissa is nonzero for NaN
            }
        } else {
            h = static_cast<uint16_t>((sign << 15) | (0x1F << 10));
        }
    } else if (exp < -14) {
        // Underflow: encode as denormal or flush to zero.
        // FP16 denormal: value = (-1)^s * 2^(-14) * (mant / 1024)
        // shift = how many positions to shift the (implicit-1 + mantissa) right
        int shift = -14 - exp; // >= 1
        // Full mantissa with implicit leading 1 (24 bits -> need to fit into 10 bits)
        uint32_t full_mant = mant | 0x800000;
        // Shift right by (13 + shift): 13 to go from 23-bit to 10-bit, plus extra for denorm
        int total_shift = 13 + shift;
        if (total_shift < 32) {
            // Round-to-nearest-even
            uint32_t half_bit = 1u << (total_shift - 1);
            uint32_t remainder = full_mant & ((1u << total_shift) - 1);
            uint32_t result_mant = full_mant >> total_shift;
            if (remainder > half_bit || (remainder == half_bit && (result_mant & 1))) {
                result_mant++;
            }
            h = static_cast<uint16_t>((sign << 15) | result_mant);
        } else {
            h = static_cast<uint16_t>(sign << 15); // flush to zero
        }
    } else if (exp == -128) {
        // FP32 zero (exp == -127 unbiased for zero/denorm, but stored exponent = 0)
        // For simplicity, treat as zero. FP32 denormals are tiny, flush to FP16 zero.
        h = static_cast<uint16_t>(sign << 15);
    } else {
        // Normal range for FP16.
        uint32_t biased_exp = static_cast<uint32_t>(exp + 15);
        // Round-to-nearest-even on the 13 truncated mantissa bits.
        uint32_t half_bit = 1u << 12;
        uint32_t remainder = mant & 0x1FFF;
        uint32_t result_mant = mant >> 13;
        if (remainder > half_bit || (remainder == half_bit && (result_mant & 1))) {
            result_mant++;
            if (result_mant >= 0x400) {
                // Mantissa overflow: carry into exponent.
                result_mant = 0;
                biased_exp++;
                if (biased_exp >= 0x1F) {
                    // Overflow to infinity.
                    h = static_cast<uint16_t>((sign << 15) | (0x1F << 10));
                    return h;
                }
            }
        }
        h = static_cast<uint16_t>((sign << 15) | (biased_exp << 10) | result_mant);
    }

    return h;
}

#endif // HALF_H
