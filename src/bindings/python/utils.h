#pragma once
#include <Python.h>
#include "tensorplay/core/Tensor.h"

namespace tensorplay {
namespace python {

// Recursively parse Python list shape, verify regularity 
// (e.g., [[1,2],[3]] is irregular, throw error)
void parse_shape(PyObject* list, std::vector<int64_t>& shape) {
    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a list");
        throw std::runtime_error("Non-list type");
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
                PyErr_SetString(PyExc_ValueError, "Irregular list (mixed types)");
                throw std::runtime_error("Irregular list");
            }
            std::vector<int64_t> cur_sub_shape;
            parse_shape(sublist, cur_sub_shape);
            if (cur_sub_shape != sub_shape) {
                PyErr_SetString(PyExc_ValueError, "Irregular list (sublists have different lengths)");
                throw std::runtime_error("Irregular list");
            }
        }
        // Merge sub-shape (e.g., [2] + [3] -> [2,3])
        shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());
    }
}

// Infer data type of list elements 
// (uniform to highest precision, e.g., int and float mixed -> float64)
DType infer_dtype(PyObject* list) {
    if (!PyList_Check(list)) {
        PyErr_SetString(
            PyExc_TypeError,
            ("Can not transform " + std::string(Py_TYPE(list)->tp_name) + " to tensor").c_str()
        );
        throw std::runtime_error("Non-list type");
    }
    int64_t len = PyList_Size(list);
    if (len == 0) return DType::Float32;  // Empty list default to float32

    DType dtype = DType::Int64;  // Default to int64
    bool has_float = false;
    bool has_int64 = false;
    
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
                 // Bool is fine, compatible with Int/Float
            } else {
                PyErr_SetString(PyExc_TypeError, "Unsupported element type (only int/float/bool supported)");
                throw std::runtime_error("Unsupported element type");
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
                 throw std::runtime_error("Unexpected nesting level in list");
             }
             copy_data(item, data, index, shape, dim + 1);
        } else {
             // Basic types
             if (PyFloat_Check(item)) {
                 data[index++] = static_cast<T>(PyFloat_AsDouble(item));
             } else if (PyLong_Check(item)) {
                 data[index++] = static_cast<T>(PyLong_AsLongLong(item));
             } else if (PyBool_Check(item)) {
                 data[index++] = static_cast<T>(PyObject_IsTrue(item));
             } else {
                 throw std::runtime_error("Unsupported element type during copy");
             }
        }
    }
}

// Python list -> C++ Tensor
Tensor list_to_tensor(PyObject* py_list) {
    // 1. Parse shape
    std::vector<int64_t> shape;
    parse_shape(py_list, shape);

    // 2. Infer data type
    DType dtype = infer_dtype(py_list);

    // 3. Create Tensor (pre-allocate memory)
    Tensor tensor(shape, dtype);

    // 4. Copy data based on dtype
    // Use reinterpret_cast to avoid template type checking issues
    size_t index = 0;
    switch (dtype) {
        case DType::Float32:
            copy_data<float>(py_list, reinterpret_cast<float*>(tensor.data_ptr()), index, shape, 0);
            break;
        case DType::Float64:
            copy_data<double>(py_list, reinterpret_cast<double*>(tensor.data_ptr()), index, shape, 0);
            break;
        case DType::Int32:
            copy_data<int32_t>(py_list, reinterpret_cast<int32_t*>(tensor.data_ptr()), index, shape, 0);
            break;
        case DType::Int64:
            copy_data<int64_t>(py_list, reinterpret_cast<int64_t*>(tensor.data_ptr()), index, shape, 0);
            break;
        case DType::UInt32:
            copy_data<uint32_t>(py_list, reinterpret_cast<uint32_t*>(tensor.data_ptr()), index, shape, 0);
            break;
        case DType::UInt64:
            copy_data<uint64_t>(py_list, reinterpret_cast<uint64_t*>(tensor.data_ptr()), index, shape, 0);
            break;
        case DType::Bool:
            copy_data<bool>(py_list, reinterpret_cast<bool*>(tensor.data_ptr()), index, shape, 0);
            break;
    }

    return tensor;
}

// Convert tensor data from one dtype to another
void convert_tensor_data(const Tensor& src, Tensor& dst) {
    // Get total number of elements in the tensor
    size_t total_elements = src.numel();
    
    // Handle different type conversions
    if (src.dtype() == DType::Float32 && dst.dtype() == DType::Float64) {
        const float* src_data = src.data_ptr<float>();
        double* dst_data = dst.data_ptr<double>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<double>(src_data[i]);
        }
    } else if (src.dtype() == DType::Float64 && dst.dtype() == DType::Float32) {
        const double* src_data = src.data_ptr<double>();
        float* dst_data = dst.data_ptr<float>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
    } else if (src.dtype() == DType::Int32 && dst.dtype() == DType::Int64) {
        const int32_t* src_data = src.data_ptr<int32_t>();
        int64_t* dst_data = dst.data_ptr<int64_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int64_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::Int64 && dst.dtype() == DType::Int32) {
        const int64_t* src_data = src.data_ptr<int64_t>();
        int32_t* dst_data = dst.data_ptr<int32_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int32_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::Float32 && dst.dtype() == DType::Int32) {
        const float* src_data = src.data_ptr<float>();
        int32_t* dst_data = dst.data_ptr<int32_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int32_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::Float32 && dst.dtype() == DType::Int64) {
        const float* src_data = src.data_ptr<float>();
        int64_t* dst_data = dst.data_ptr<int64_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int64_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::Float64 && dst.dtype() == DType::Int32) {
        const double* src_data = src.data_ptr<double>();
        int32_t* dst_data = dst.data_ptr<int32_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int32_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::Float64 && dst.dtype() == DType::Int64) {
        const double* src_data = src.data_ptr<double>();
        int64_t* dst_data = dst.data_ptr<int64_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int64_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::Int32 && dst.dtype() == DType::Float32) {
        const int32_t* src_data = src.data_ptr<int32_t>();
        float* dst_data = dst.data_ptr<float>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
    } else if (src.dtype() == DType::Int32 && dst.dtype() == DType::Float64) {
        const int32_t* src_data = src.data_ptr<int32_t>();
        double* dst_data = dst.data_ptr<double>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<double>(src_data[i]);
        }
    } else if (src.dtype() == DType::Int64 && dst.dtype() == DType::Float32) {
        const int64_t* src_data = src.data_ptr<int64_t>();
        float* dst_data = dst.data_ptr<float>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
    } else if (src.dtype() == DType::Int64 && dst.dtype() == DType::Float64) {
        const int64_t* src_data = src.data_ptr<int64_t>();
        double* dst_data = dst.data_ptr<double>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<double>(src_data[i]);
        }
    } else if (src.dtype() == dst.dtype()) {
        // Same dtype, just copy
        memcpy(dst.data_ptr(), src.data_ptr(), total_elements * src.itemsize());
    } else if (src.dtype() == DType::UInt32 && dst.dtype() == DType::Float32) {
        const uint32_t* src_data = src.data_ptr<uint32_t>();
        float* dst_data = dst.data_ptr<float>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
    } else if (src.dtype() == DType::UInt32 && dst.dtype() == DType::Float64) {
        const uint32_t* src_data = src.data_ptr<uint32_t>();
        double* dst_data = dst.data_ptr<double>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<double>(src_data[i]);
        }
    } else if (src.dtype() == DType::UInt32 && dst.dtype() == DType::Int32) {
        const uint32_t* src_data = src.data_ptr<uint32_t>();
        int32_t* dst_data = dst.data_ptr<int32_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int32_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::UInt32 && dst.dtype() == DType::Int64) {
        const uint32_t* src_data = src.data_ptr<uint32_t>();
        int64_t* dst_data = dst.data_ptr<int64_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int64_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::UInt64 && dst.dtype() == DType::Float32) {
        const uint64_t* src_data = src.data_ptr<uint64_t>();
        float* dst_data = dst.data_ptr<float>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
    } else if (src.dtype() == DType::UInt64 && dst.dtype() == DType::Float64) {
        const uint64_t* src_data = src.data_ptr<uint64_t>();
        double* dst_data = dst.data_ptr<double>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<double>(src_data[i]);
        }
    } else if (src.dtype() == DType::UInt64 && dst.dtype() == DType::Int32) {
        const uint64_t* src_data = src.data_ptr<uint64_t>();
        int32_t* dst_data = dst.data_ptr<int32_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int32_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::UInt64 && dst.dtype() == DType::Int64) {
        const uint64_t* src_data = src.data_ptr<uint64_t>();
        int64_t* dst_data = dst.data_ptr<int64_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int64_t>(src_data[i]);
        }
    } else if (src.dtype() == DType::Bool && dst.dtype() == DType::Float32) {
        const bool* src_data = src.data_ptr<bool>();
        float* dst_data = dst.data_ptr<float>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<float>(src_data[i]);
        }
    } else if (src.dtype() == DType::Bool && dst.dtype() == DType::Float64) {
        const bool* src_data = src.data_ptr<bool>();
        double* dst_data = dst.data_ptr<double>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<double>(src_data[i]);
        }
    } else if (src.dtype() == DType::Bool && dst.dtype() == DType::Int32) {
        const bool* src_data = src.data_ptr<bool>();
        int32_t* dst_data = dst.data_ptr<int32_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            dst_data[i] = static_cast<int32_t>(src_data[i]);
        }
    }
}

}} // namespace tensorplay::python
