// Float8_e4m3fn / Float8_e5m2 -- 8-bit floating point types.
//
// Conversion uses round-to-nearest-even on the way in; e4m3fn has no Inf --
// overflow saturates to max finite, NaN maps to 0x7f; e5m2 follows IEEE half
// rules.
#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#include "Half.h"

#if defined(__CUDACC__)
#define TP_F8_HOST_DEVICE __host__ __device__
#else
#define TP_F8_HOST_DEVICE
#endif

namespace tensorplay {

inline TP_F8_HOST_DEVICE uint32_t fp8_fp32_to_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

inline TP_F8_HOST_DEVICE float fp8_bits_to_fp32(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// e4m3fn -> fp32
// ---------------------------------------------------------------------------
inline TP_F8_HOST_DEVICE float fp8e4m3fn_to_fp32_value(uint8_t input) {
    const uint32_t w = (uint32_t)input << 24;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
#if defined(__CUDA_ARCH__)
    uint32_t renorm_shift = __clz(nonsign);
#elif defined(_MSC_VER) && !defined(__clang__)
    unsigned long nonsign_bsr;
    _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
    uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31u;
#else
    uint32_t renorm_shift =
        nonsign != 0 ? (uint32_t)__builtin_clz(nonsign) : sizeof(uint32_t) * 8;
#endif
    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
    const int32_t inf_nan_mask =
        ((int32_t)(nonsign + UINT32_C(0x01000000)) >> 8) & INT32_C(0x7F800000);
    const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
    const uint32_t result =
        sign | ((((nonsign << renorm_shift >> 4) +
                  ((UINT32_C(0x78) - renorm_shift) << 23)) |
                 (uint32_t)inf_nan_mask) &
                ~(uint32_t)zero_mask);
    return fp8_bits_to_fp32(result);
}

// fp32 -> e4m3fn
inline TP_F8_HOST_DEVICE uint8_t fp8e4m3fn_from_fp32_value(float f) {
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
    uint32_t f_bits = fp8_fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
        if (f_bits > UINT32_C(0x7F800000)) {
            result = 0x7f;  // NaN -> NaN
        } else {
            result = 0x7e;  // finite overflow / inf saturates (no Inf in fn)
        }
    } else if (f_bits < (UINT32_C(121) << 23)) {
        f_bits = fp8_fp32_to_bits(
            fp8_bits_to_fp32(f_bits) + fp8_bits_to_fp32(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
        const uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        result = static_cast<uint8_t>(f_bits >> 20);
        if (result == 0x7f) result = 0x7e;  // rounding carry: saturate
    }
    return result | (uint8_t)(sign >> 24);
}

// ---------------------------------------------------------------------------
// e5m2: same bit layout as IEEE half shifted left by 8.
// ---------------------------------------------------------------------------
inline TP_F8_HOST_DEVICE float fp8e5m2_to_fp32_value(uint8_t input) {
    uint16_t half_repr = (uint16_t)input << 8;
    return detail::half_to_float_bits(half_repr);
}

inline TP_F8_HOST_DEVICE uint8_t fp8e5m2_from_fp32_value(float f) {
    constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
    constexpr uint32_t fp8_max = UINT32_C(143) << 23;
    constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
    uint32_t f_bits = fp8_fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
        // e5m2 has real infinities.
        result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
    } else if (f_bits < (UINT32_C(113) << 23)) {
        f_bits = fp8_fp32_to_bits(
            fp8_bits_to_fp32(f_bits) + fp8_bits_to_fp32(denorm_mask));
        result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
        const uint32_t mant_odd = (f_bits >> 21) & 1;
        f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
        f_bits += mant_odd;
        result = static_cast<uint8_t>(f_bits >> 21);
    }
    return result | (uint8_t)(sign >> 24);
}

// ---------------------------------------------------------------------------
// Wrapper classes
// ---------------------------------------------------------------------------

struct Float8_e4m3fn {
    uint8_t x_;

    TP_F8_HOST_DEVICE Float8_e4m3fn() = default;
    static constexpr TP_F8_HOST_DEVICE Float8_e4m3fn from_bits(uint8_t bits) {
        Float8_e4m3fn r; r.x_ = bits; return r;
    }
    /* implicit */ TP_F8_HOST_DEVICE Float8_e4m3fn(float v)
        : x_(fp8e4m3fn_from_fp32_value(v)) {}
    TP_F8_HOST_DEVICE operator float() const {
        return fp8e4m3fn_to_fp32_value(x_);
    }
};

struct Float8_e5m2 {
    uint8_t x_;

    TP_F8_HOST_DEVICE Float8_e5m2() = default;
    static constexpr TP_F8_HOST_DEVICE Float8_e5m2 from_bits(uint8_t bits) {
        Float8_e5m2 r; r.x_ = bits; return r;
    }
    /* implicit */ TP_F8_HOST_DEVICE Float8_e5m2(float v)
        : x_(fp8e5m2_from_fp32_value(v)) {}
    TP_F8_HOST_DEVICE operator float() const {
        return fp8e5m2_to_fp32_value(x_);
    }
};
// ---------------------------------------------------------------------------
// Arithmetic/comparison via float promotion. Constrained to
// Float8 operands: without the constraint these templates hijack every other
// type combination (complex, etc.) and hard-fail on the float() casts.
// ---------------------------------------------------------------------------
namespace detail {
template <typename T> struct is_float8 : std::false_type {};
template <> struct is_float8<Float8_e4m3fn> : std::true_type {};
template <> struct is_float8<Float8_e5m2> : std::true_type {};
}
#define TP_F8_BINARY_OP(OP)                                                    \
    template <typename A, typename B,                                          \
              typename = std::enable_if_t<                                     \
                  detail::is_float8<std::decay_t<A>>::value ||                 \
                  detail::is_float8<std::decay_t<B>>::value>>                  \
    inline TP_F8_HOST_DEVICE auto operator OP(const A& a, const B& b)          \
        ->decltype(float(a) OP float(b)) {                                     \
        return float(a) OP float(b);                                           \
    }
TP_F8_BINARY_OP(+)
TP_F8_BINARY_OP(-)
TP_F8_BINARY_OP(*)
TP_F8_BINARY_OP(/)
#undef TP_F8_BINARY_OP

#define TP_F8_CMP_OP(OP)                                                       \
    template <typename A, typename B>                                          \
    inline TP_F8_HOST_DEVICE bool operator OP(const A& a, const B& b)          \
      requires std::is_same_v<A, tensorplay::Float8_e4m3fn> || std::is_same_v<A, tensorplay::Float8_e5m2> \
                           || std::is_same_v<B, tensorplay::Float8_e4m3fn>                 \
                           || std::is_same_v<B, tensorplay::Float8_e5m2> {                 \
        return float(a) OP float(b);                                           \
    }
TP_F8_CMP_OP(==)
TP_F8_CMP_OP(!=)
TP_F8_CMP_OP(<)
TP_F8_CMP_OP(<=)
TP_F8_CMP_OP(>)
TP_F8_CMP_OP(>=)
#undef TP_F8_CMP_OP

// compound assignment through float promotion
inline TP_F8_HOST_DEVICE Float8_e4m3fn& operator+=(Float8_e4m3fn& a, const Float8_e4m3fn& b){ a = Float8_e4m3fn(float(a)+float(b)); return a; }
inline TP_F8_HOST_DEVICE Float8_e4m3fn& operator-=(Float8_e4m3fn& a, const Float8_e4m3fn& b){ a = Float8_e4m3fn(float(a)-float(b)); return a; }
inline TP_F8_HOST_DEVICE Float8_e4m3fn& operator*=(Float8_e4m3fn& a, const Float8_e4m3fn& b){ a = Float8_e4m3fn(float(a)*float(b)); return a; }
inline TP_F8_HOST_DEVICE Float8_e4m3fn& operator/=(Float8_e4m3fn& a, const Float8_e4m3fn& b){ a = Float8_e4m3fn(float(a)/float(b)); return a; }
inline TP_F8_HOST_DEVICE Float8_e5m2& operator+=(Float8_e5m2& a, const Float8_e5m2& b){ a = Float8_e5m2(float(a)+float(b)); return a; }
inline TP_F8_HOST_DEVICE Float8_e5m2& operator-=(Float8_e5m2& a, const Float8_e5m2& b){ a = Float8_e5m2(float(a)-float(b)); return a; }
inline TP_F8_HOST_DEVICE Float8_e5m2& operator*=(Float8_e5m2& a, const Float8_e5m2& b){ a = Float8_e5m2(float(a)*float(b)); return a; }
inline TP_F8_HOST_DEVICE Float8_e5m2& operator/=(Float8_e5m2& a, const Float8_e5m2& b){ a = Float8_e5m2(float(a)/float(b)); return a; }
}  // namespace tensorplay
