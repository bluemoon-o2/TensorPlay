#pragma once

#include <variant>
#include <stdexcept>
#include <iostream>
#include <string>
#include <type_traits>

#include "tensorplay/core/DType.h"

namespace tensorplay {

// Scalar class aligned with Tensor DType system
class Scalar {
public:
    Scalar() : type_(DType::Undefined) {}

    // Strict constructors for core types
    // We allow implicit conversions for usability
    
    Scalar(int32_t v) : val_(v), type_(DType::Int32) {}
    Scalar(int64_t v) : val_(v), type_(DType::Int64) {}
    Scalar(float v) : val_(v), type_(DType::Float32) {}
    Scalar(double v) : val_(v), type_(DType::Float64) {}
    Scalar(bool v) : val_(v), type_(DType::Bool) {}

    // Copy/Move
    Scalar(const Scalar&) = default;
    Scalar(Scalar&&) = default;
    Scalar& operator=(const Scalar&) = default;
    Scalar& operator=(Scalar&&) = default;

    // Accessors
    template<typename T>
    T to() const {
        // Strict safe conversion or same-precision conversion
        if (type_ == DType::Float64) {
            if constexpr (std::is_integral_v<T>) throw std::runtime_error("Cannot safely convert Float64 Scalar to Integral type");
            return static_cast<T>(std::get<double>(val_));
        } else if (type_ == DType::Float32) {
            if constexpr (std::is_integral_v<T>) throw std::runtime_error("Cannot safely convert Float32 Scalar to Integral type");
            return static_cast<T>(std::get<float>(val_));
        } else if (type_ == DType::Int64) {
            int64_t v = std::get<int64_t>(val_);
            // Safe narrowing check for int32
            if constexpr (std::is_same_v<T, int32_t>) {
                if (v > INT32_MAX || v < INT32_MIN) throw std::runtime_error("Scalar value overflow for Int32");
            }
            return static_cast<T>(v);
        } else if (type_ == DType::Int32) {
            return static_cast<T>(std::get<int32_t>(val_));
        } else if (type_ == DType::Bool) {
            if constexpr (!std::is_same_v<T, bool>) throw std::runtime_error("Cannot convert Bool Scalar to non-Bool type implicitly");
            return static_cast<T>(std::get<bool>(val_));
        }
        throw std::runtime_error("Scalar is undefined");
    }

    // Type checking
    DType dtype() const { return type_; }
    
    bool isFloatingPoint() const {
        return type_ == DType::Float64 || type_ == DType::Float32;
    }

    bool isIntegral(bool includeBool = false) const {
        return type_ == DType::Int64 || type_ == DType::Int32 || (includeBool && type_ == DType::Bool);
    }
    
    bool isBoolean() const {
        return type_ == DType::Bool;
    }
    
    bool is_dtype(DType dt) const {
        return type_ == dt;
    }

    // String representation
    std::string toString() const {
        if (type_ == DType::Undefined) return "Scalar(Undefined)";
        std::string s = "Scalar(";
        if (type_ == DType::Float64) s += std::to_string(std::get<double>(val_));
        else if (type_ == DType::Float32) s += std::to_string(std::get<float>(val_));
        else if (type_ == DType::Int64) s += std::to_string(std::get<int64_t>(val_));
        else if (type_ == DType::Int32) s += std::to_string(std::get<int32_t>(val_));
        else if (type_ == DType::Bool) s += (std::get<bool>(val_) ? "true" : "false");
        s += ", dtype=";
        s += ::tensorplay::toString(type_);
        s += ")";
        return s;
    }

    // Operators
    Scalar operator+(const Scalar& other) const;
    Scalar operator-(const Scalar& other) const;
    Scalar operator*(const Scalar& other) const;
    Scalar operator/(const Scalar& other) const;
    
    bool operator==(const Scalar& other) const;
    bool operator!=(const Scalar& other) const;
    bool operator>(const Scalar& other) const;
    bool operator<(const Scalar& other) const;

private:
    std::variant<int32_t, int64_t, float, double, bool> val_;
    DType type_;
    
    // Helper for promotion
    static DType promote_types(DType a, DType b);
};

// Stream operator
inline std::ostream& operator<<(std::ostream& os, const Scalar& s) {
    os << s.toString();
    return os;
}

} // namespace tensorplay
