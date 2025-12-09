#pragma once
#include <Python.h>
#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Exception.h"
#include <nanobind/nanobind.h>

namespace tensorplay {
namespace python {

// Recursively parse Python list shape, verify regularity 
// (e.g., [[1,2],[3]] is irregular, throw error)
inline void parse_shape(PyObject* list, std::vector<int64_t>& shape) {
    if (!PyList_Check(list)) {
        TP_THROW(TypeError, "Input must be a list");
    }
    int64_t len = PyList_Size(list);
    shape.push_back(len);
    if (len == 0) return;

    // Ensure all sublists have the same length and recursively check shape
    PyObject* first = PyList_GetItem(list, 0);
    if (PyList_Check(first)) {
        std::vector<int64_t> sub_shape;
        parse_shape(first, sub_shape);
        // Verify all sublists have the same shape
        for (int64_t i = 1; i < len; ++i) {
            PyObject* sublist = PyList_GetItem(list, i);
            if (!PyList_Check(sublist)) {
                TP_THROW(ValueError, "Irregular list (mixed types)");
            }
            std::vector<int64_t> cur_sub_shape;
            parse_shape(sublist, cur_sub_shape);
            if (cur_sub_shape != sub_shape) {
                TP_THROW(ValueError, "Irregular list (sublists have different lengths)");
            }
        }
        // Merge sub-shape (e.g., [2] + [3] -> [2,3])
        shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());
    }
}

// Infer data type of list elements 
// (uniform to highest precision, e.g., int and float mixed -> float64)
inline DType infer_dtype(PyObject* list) {
    if (!PyList_Check(list)) {
        TP_THROW(TypeError, "Can not transform " + std::string(Py_TYPE(list)->tp_name) + " to tensor");
    }
    int64_t len = PyList_Size(list);
    if (len == 0) return DType::Float32;  // Empty list default to float32

    DType dtype = DType::Int64;  // Default to int64
    bool has_float = false;
    
    for (int64_t i = 0; i < len; ++i) {
        PyObject* item = PyList_GetItem(list, i);
        if (PyList_Check(item)) {
            // Recursively infer dtype of sublists
            DType sub_dtype = infer_dtype(item);
            // Track float presence
            if (sub_dtype == DType::Float32 || sub_dtype == DType::Float64) {
                has_float = true;
            }
        } else {
            // Basic type inference
            if (PyFloat_Check(item)) {
                has_float = true;  // Mark that we have a float
            } else if (PyLong_Check(item)) {
                // Int is fine
            } else if (PyBool_Check(item)) {
                 // Bool is fine
            } else {
                TP_THROW(TypeError, "Unsupported element type (only int/float/bool supported)");
            }
        }
    }
    
    // Determine final dtype based on what we found
    if (has_float) {
        dtype = DType::Float32;  // Default to float32 for any float
    } else {
        dtype = DType::Int64;    // Default to Int64 for ints
    }
    
    return dtype;
}

// Recursively copy list data to Tensor memory (row-major order)
template <typename T>
void copy_data(PyObject* list, T* data, size_t& index, const std::vector<int64_t>& shape, int dim) {
    int64_t len = PyList_Size(list);
    for (int64_t i = 0; i < len; ++i) {
        PyObject* item = PyList_GetItem(list, i);
        if (PyList_Check(item)) {
             if (dim >= shape.size() - 1) {
                 TP_THROW(RuntimeError, "Unexpected nesting level in list");
             }
             copy_data(item, data, index, shape, dim + 1);
        } else {
             // Basic types
             if (dim != shape.size() - 1) {
                 TP_THROW(RuntimeError, "Unexpected non-list element (ragged nested list?)");
             }
             
             T val;
             if (PyFloat_Check(item)) {
                 val = static_cast<T>(PyFloat_AsDouble(item));
             } else if (PyBool_Check(item)) {
                 val = static_cast<T>(item == Py_True);
             } else if (PyLong_Check(item)) {
                 val = static_cast<T>(PyLong_AsLongLong(item));
             } else {
                 TP_THROW(TypeError, "Unsupported element type");
             }
             
             data[index++] = val;
        }
    }
}

// Helper dispatch macro for local use
#define TP_DISPATCH_CASE(enum_type, type, ...) \
  case enum_type: { \
    using scalar_t = type; \
    __VA_ARGS__(); \
    break; \
  }

#define TP_DISPATCH_ALL_TYPES(dtype, NAME, ...) \
  switch (dtype) { \
    TP_DISPATCH_CASE(DType::UInt8, uint8_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Int8, int8_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Int16, int16_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Int32, int32_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Int64, int64_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::UInt16, uint16_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::UInt32, uint32_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::UInt64, uint64_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Float32, float, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Float64, double, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Bool, bool, __VA_ARGS__) \
    default: \
      TP_THROW(NotImplementedError, std::string(NAME) + " not implemented for this dtype"); \
  }

inline Tensor list_to_tensor(PyObject* list) {
    std::vector<int64_t> shape;
    parse_shape(list, shape);
    
    DType dtype = infer_dtype(list);
    
    // Create CPU tensor
    Tensor t(shape, dtype, Device(DeviceType::CPU));
    
    // Dispatch copy_data based on dtype
    size_t index = 0;
    
    TP_DISPATCH_ALL_TYPES(dtype, "list_to_tensor", [&] {
        using T = scalar_t;
        T* data_ptr = t.data_ptr<T>();
        copy_data(list, data_ptr, index, shape, 0);
    });
    
    return t;
}

inline void convert_tensor_data(const Tensor& src, Tensor& dst) {
    dst.copy_(src);
}

} // namespace python
} // namespace tensorplay
