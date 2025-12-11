#include "python_bindings.h"
#include "tensorplay/ops/TensorBindingsGenerated.h"
#include "utils.h"
#include "dlpack_types.h"
#include "TensorImpl.h" // For unsafeGetTensorImpl
#include "TPXTensor.h" // TPX Tensor
#include "Storage.h"
#include "DataPtr.h"
#include "Node.h" // For grad_fn
#include <mutex>

using namespace tensorplay::python;
// Using TPX Tensor as the main Tensor exposed to Python
using Tensor = tensorplay::tpx::Tensor; 
using P10Tensor = tensorplay::Tensor; // Core Tensor

// --- DLPack Helpers ---

static DLDataType to_dlpack_dtype(DType dtype) {
    DLDataType dt;
    dt.lanes = 1;
    switch (dtype) {
        case DType::Float32: dt.code = kDLFloat; dt.bits = 32; break;
        case DType::Float64: dt.code = kDLFloat; dt.bits = 64; break;
        case DType::Int32:   dt.code = kDLInt;   dt.bits = 32; break;
        case DType::Int64:   dt.code = kDLInt;   dt.bits = 64; break;
        case DType::Int8:    dt.code = kDLInt;   dt.bits = 8;  break;
        case DType::Int16:   dt.code = kDLInt;   dt.bits = 16; break;
        case DType::UInt8:   dt.code = kDLUInt;  dt.bits = 8;  break;
        case DType::UInt16:  dt.code = kDLUInt;  dt.bits = 16; break;
        case DType::UInt32:  dt.code = kDLUInt;  dt.bits = 32; break;
        case DType::UInt64:  dt.code = kDLUInt;  dt.bits = 64; break;
        case DType::Bool:    dt.code = kDLBool;  dt.bits = 8;  break;
        default: TP_THROW(RuntimeError, "Unsupported DType for DLPack");
    }
    return dt;
}

static DType from_dlpack_dtype(DLDataType dt) {
    if (dt.lanes != 1) TP_THROW(RuntimeError, "DLPack: Unsupported lanes != 1");
    if (dt.code == kDLFloat) {
        if (dt.bits == 32) return DType::Float32;
        if (dt.bits == 64) return DType::Float64;
    } else if (dt.code == kDLInt) {
        if (dt.bits == 32) return DType::Int32;
        if (dt.bits == 64) return DType::Int64;
        if (dt.bits == 8)  return DType::Int8;
        if (dt.bits == 16) return DType::Int16;
    } else if (dt.code == kDLUInt) {
        if (dt.bits == 8)  return DType::UInt8;
        if (dt.bits == 16) return DType::UInt16;
        if (dt.bits == 32) return DType::UInt32;
        if (dt.bits == 64) return DType::UInt64;
    } else if (dt.code == kDLBool) {
        if (dt.bits == 8) return DType::Bool;
    }
    TP_THROW(RuntimeError, "Unsupported DLPack dtype");
}

static DLDevice to_dlpack_device(Device device) {
    DLDevice d;
    d.device_id = device.index();
    switch (device.type()) {
        case DeviceType::CPU: d.device_type = kDLCPU; break;
        case DeviceType::CUDA: d.device_type = kDLCUDA; break;
        default: TP_THROW(RuntimeError, "Unsupported Device for DLPack");
    }
    return d;
}

static Device from_dlpack_device(DLDevice d) {
    DeviceType type;
    switch (d.device_type) {
        case kDLCPU: type = DeviceType::CPU; break;
        case kDLCUDA: type = DeviceType::CUDA; break;
        case kDLCUDAHost: type = DeviceType::CPU; break; // Treat CUDA Host as CPU
        default: TP_THROW(RuntimeError, "Unsupported DLPack device type");
    }
    return Device(type, d.device_id);
}

// Simple thread-safe object pool for DLManagedTensor
struct DLManagedTensorPool {
    std::vector<DLManagedTensor*> pool;
    std::mutex mutex;
    
    ~DLManagedTensorPool() {
        for (auto* p : pool) delete p;
    }
    
    DLManagedTensor* allocate() {
        std::lock_guard<std::mutex> lock(mutex);
        if (pool.empty()) {
            return new DLManagedTensor();
        }
        DLManagedTensor* p = pool.back();
        pool.pop_back();
        return p;
    }
    
    void deallocate(DLManagedTensor* p) {
        std::lock_guard<std::mutex> lock(mutex);
        pool.push_back(p);
    }
};

static DLManagedTensorPool global_dlpack_pool;

// DLPack Deleter (C-compatible)
static void dlpack_deleter(DLManagedTensor* tensor) {
    if (tensor->manager_ctx) {
        // Decrement refcount of the tensorplay Tensor
        delete static_cast<Tensor*>(tensor->manager_ctx);
    }
    // Return to pool instead of delete
    global_dlpack_pool.deallocate(tensor);
}

// Optimized deleter for PyObject-managed DLPack
static void dlpack_pyobject_deleter(DLManagedTensor* managed) {
    if (managed->manager_ctx) {
        nb::gil_scoped_acquire gil;
        Py_DECREF(static_cast<PyObject*>(managed->manager_ctx));
    }
    // Return to pool instead of delete
    managed->manager_ctx = nullptr; // Clear ctx
    global_dlpack_pool.deallocate(managed);
}

// Capsule Destructor
static void dlpack_capsule_destructor(PyObject* cap) {
    // If the capsule is still named "dltensor", it means it wasn't consumed.
    // We must clean up the DLManagedTensor.
    const char* name = PyCapsule_GetName(cap);
    if (name && strcmp(name, "dltensor") == 0) {
        DLManagedTensor* managed = (DLManagedTensor*)PyCapsule_GetPointer(cap, "dltensor");
        if (managed) {
            managed->deleter(managed);
        }
    }
}

static nb::capsule to_dlpack(nb::object self_obj, std::optional<int64_t> stream = std::nullopt) {
    const Tensor& self = nb::cast<const Tensor&>(self_obj);
    // Use pool
    DLManagedTensor* managed = global_dlpack_pool.allocate();
    
    // Optimization: Keep the Python object alive instead of copying the C++ Tensor.
    // This avoids one heap allocation (new Tensor) and one copy constructor.
    PyObject* ptr = self_obj.ptr();
    Py_INCREF(ptr);
    managed->manager_ctx = ptr;
    managed->deleter = dlpack_pyobject_deleter;
    
    DLTensor& dl = managed->dl_tensor;
    dl.data = self.data_ptr();
    dl.byte_offset = 0;
    dl.ndim = static_cast<int>(self.dim());
    
    // We need persistent pointers for shape and strides.
    // The Python object owns the C++ Tensor, which owns the TensorImpl, which owns the vectors.
    // So as long as self_obj is alive, these pointers are valid.
    auto impl = self.core().unsafeGetTensorImpl();
    
    dl.shape = const_cast<int64_t*>(impl->sizes().data());
    dl.strides = const_cast<int64_t*>(impl->strides().data());
    
    dl.dtype = to_dlpack_dtype(self.dtype());
    dl.device = to_dlpack_device(self.device());
    
    PyObject* cap = PyCapsule_New(managed, "dltensor", dlpack_capsule_destructor);
    return nb::steal<nb::capsule>(cap);
}

static Tensor from_dlpack(nb::object o) {
    PyObject* cap_ptr;
    bool is_capsule = PyCapsule_CheckExact(o.ptr());
    nb::capsule cap_holder; // Holds reference if we created a new capsule
    
    if (is_capsule) {
        cap_ptr = o.ptr();
    } else {
        // Optimization: Use C-API to call __dlpack__ directly.
        // faster than nb::hasattr + o.attr()()
        static PyObject* dlpack_str = PyUnicode_InternFromString("__dlpack__");
        PyObject* res = PyObject_CallMethodObjArgs(o.ptr(), dlpack_str, nullptr);
        if (!res) {
             PyErr_Clear(); // Clear AttributeError
             TP_THROW(TypeError, "Object is not a DLPack capsule and does not have __dlpack__ method");
        }
        cap_holder = nb::steal<nb::capsule>(res);
        cap_ptr = res;
    }
    
    // Check name
    const char* name = PyCapsule_GetName(cap_ptr);
    if (strcmp(name, "dltensor") != 0) {
        TP_THROW(ValueError, "DLPack capsule is invalid or already consumed");
    }
    
    DLManagedTensor* managed = (DLManagedTensor*)PyCapsule_GetPointer(cap_ptr, "dltensor");
    if (!managed) TP_THROW(ValueError, "Invalid DLPack capsule pointer");
    
    // Rename to mark as consumed
    PyCapsule_SetName(cap_ptr, "used_dltensor");
    
    // Extract metadata
    DLTensor& dl = managed->dl_tensor;
    DType dtype = from_dlpack_dtype(dl.dtype);
    Device device = from_dlpack_device(dl.device);
    
    std::vector<int64_t> shape(dl.shape, dl.shape + dl.ndim);
    std::vector<int64_t> strides;
    if (dl.strides) {
        strides.assign(dl.strides, dl.strides + dl.ndim);
    } else {
        // Assume contiguous
        strides.resize(dl.ndim);
        int64_t stride = 1;
        for (int i = dl.ndim - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
    
    // Create DataPtr with custom deleter
    // The deleter must call managed->deleter(managed)
    auto deleter = [managed](void*) {
        if (managed->deleter) {
            managed->deleter(managed);
        }
    };
    
    // Calculate total bytes roughly for storage info (optional but good for tracking)
    size_t nbytes = 1;
    for(auto s : shape) nbytes *= s;
    nbytes *= (dl.dtype.bits / 8);
    
    // Data pointer
    void* data_ptr_raw = static_cast<char*>(dl.data) + dl.byte_offset;
    
    tensorplay::DataPtr ptr(data_ptr_raw, deleter, device);
    tensorplay::Storage storage(std::move(ptr), nbytes); // Using wrapper constructor
    
    // Create Tensor directly with strides (Optimization: Avoid as_strided overhead)
    auto impl = std::make_shared<tensorplay::TensorImpl>(storage, shape, strides, dtype);
    return Tensor(P10Tensor(impl));
}

// Helper function implementation
Tensor create_tensor(nb::object data, std::optional<DType> dtype, std::optional<Device> device) {
    Tensor t;
    
    // 0. Fast Path: Check for NumPy array directly using type comparison
    // This is faster than strcmp and avoids string operations
    static PyTypeObject* numpy_array_type = []() -> PyTypeObject* {
        PyObject* np = PyImport_ImportModule("numpy");
        if (!np) { PyErr_Clear(); return nullptr; }
        PyObject* type = PyObject_GetAttrString(np, "ndarray");
        Py_DECREF(np);
        return (PyTypeObject*)type; // Leak reference intentionally to keep type alive
    }();

    if (numpy_array_type && Py_TYPE(data.ptr()) == numpy_array_type) {
         // Use nb::numpy tag to ensure we get correct strides for NumPy arrays
         nb::ndarray<nb::numpy> array = nb::cast<nb::ndarray<nb::numpy>>(data);
         
         // Optimization: Use loop for shape construction (safer with nanobind wrapper)
         size_t ndim = array.ndim();
         std::vector<int64_t> shape(ndim);
         for (size_t i = 0; i < ndim; ++i) {
             shape[i] = array.shape(i);
         }
         
         nb::dlpack::dtype dt = array.dtype();
         DType inferred_dtype = DType::Undefined;
         
         // Map dlpack dtype to TensorPlay DType
         uint8_t code = dt.code;
         uint8_t bits = dt.bits;
         
         if (code == 2 && bits == 32) inferred_dtype = DType::Float32;
         else if (code == 2 && bits == 64) inferred_dtype = DType::Float64;
         else if (code == 0 && bits == 32) inferred_dtype = DType::Int32;
         else if (code == 0 && bits == 64) inferred_dtype = DType::Int64;
         else if (code == 1 && bits == 32) inferred_dtype = DType::UInt32;
         else if (code == 1 && bits == 64) inferred_dtype = DType::UInt64;
         else if (code == 6 || (code == 1 && bits == 8)) inferred_dtype = DType::Bool; 
         else {
             TP_THROW(TypeError, "Unsupported NumPy/DLPack dtype: code=" + std::to_string(code) + ", bits=" + std::to_string(bits));
         }
         
         DType final_dtype = dtype.value_or(inferred_dtype);
         
         // Calculate element-wise strides for TensorPlay
         size_t itemsize = bits / 8;
         std::vector<int64_t> strides;
         
         // Fallback to Python API for strides to avoid 0-stride issue with nanobind ndarray wrapper
         try {
             nb::tuple py_strides = data.attr("strides");
             for (size_t i = 0; i < ndim; ++i) {
                 int64_t s = nb::cast<int64_t>(py_strides[i]);
                 strides.push_back(s / itemsize);
             }
         } catch (...) {
             // Should not happen for numpy array, but if it does, try nanobind API
             for (size_t i = 0; i < array.ndim(); ++i) {
                 strides.push_back(array.stride(i) / itemsize);
             }
         }
         
         // Zero-Copy Path conditions:
         // 1. dtypes match
         // 2. target device is CPU (since numpy is on CPU)
         // 3. no explicit device move requested to non-CPU
         bool is_cpu = !device.has_value() || device->type() == DeviceType::CPU;
         
         if (final_dtype == inferred_dtype && is_cpu) {
             // ZERO-COPY IMPLEMENTATION
             // We use the raw pointer from numpy and keep the numpy object alive via deleter
             
             PyObject* py_obj = data.ptr();
             // Increment refcount to keep numpy array alive
             Py_INCREF(py_obj);
             
             auto deleter = [py_obj](void*) {
                 // Acquire GIL because we are touching Python objects
                 nb::gil_scoped_acquire gil;
                 Py_DECREF(py_obj);
             };
             
             // Calculate total size for info
             size_t numel = 1;
             for(auto s : shape) numel *= s;
             size_t nbytes = numel * itemsize;
             
             tensorplay::DataPtr ptr(array.data(), deleter, Device(DeviceType::CPU));
             tensorplay::Storage storage(std::move(ptr), nbytes);
             
             // Create Tensor with specific strides directly (Optimization: Avoid as_strided overhead)
            auto impl = std::make_shared<tensorplay::TensorImpl>(storage, shape, strides, final_dtype);
            t = Tensor(P10Tensor(impl));
            
        } else {
             // Copy Path (Casting or Device Move)
             // Use P10Tensor directly for intermediate
             P10Tensor p10_t(shape, final_dtype, device.value_or(Device(DeviceType::CPU)));
             t = Tensor(p10_t);
             
             size_t numel = 1;
             for(auto s : shape) numel *= s;
             size_t total_bytes = numel * itemsize; 
             
             if (final_dtype == inferred_dtype) {
                 std::memcpy(t.core().data_ptr(), array.data(), total_bytes);
             } else {
                 P10Tensor src(shape, inferred_dtype, Device(DeviceType::CPU));
                 std::memcpy(src.data_ptr(), array.data(), total_bytes);
                 t.core().copy_(src);
             }
         }
    }
    // 1. Check for DLPack support (fast path for interop other than NumPy)
    // Use C-API with interned string for maximum speed
    else if ([](PyObject* ptr) {
        static PyObject* dlpack_attr_name = PyUnicode_InternFromString("__dlpack__");
        return PyObject_HasAttr(ptr, dlpack_attr_name);
    }(data.ptr())) {
        t = from_dlpack(data);
    } else if (nb::isinstance<nb::list>(data) || nb::isinstance<nb::tuple>(data)) {
        t = list_to_tensor(data.ptr());
    } else if (nb::isinstance<nb::int_>(data) || nb::isinstance<nb::float_>(data) || nb::isinstance<nb::bool_>(data)) {
          nb::list l;
          l.append(data);
         t = list_to_tensor(l.ptr());
         t = t.reshape({});
    } else {
         if (data.is_none()) {
            // Return undefined tensor (default constructor)
            return Tensor();
         }
         try {
             nb::list l(data);
             t = list_to_tensor(l.ptr());
         } catch (...) {
             TP_THROW(TypeError, "Unsupported data type for Tensor creation");
         }
    }
    
    // Handle dtype conversion if needed
    if (dtype.has_value() && t.dtype() != *dtype) {
        P10Tensor new_t(static_cast<std::vector<int64_t>>(t.shape()), *dtype, device.value_or(Device(DeviceType::CPU)));
        Tensor new_t_wrapper(new_t);
        convert_tensor_data(t, new_t_wrapper);
        t = new_t_wrapper;
    }
    
    // Handle device movement if needed
    // Note: list_to_tensor returns CPU tensor.
    if (device.has_value() && t.device() != *device) {
        P10Tensor new_t(static_cast<std::vector<int64_t>>(t.shape()), t.dtype(), *device);
        new_t.copy_(t.core());
        t = Tensor(new_t);
    }

    return t;
}

void init_tensor(nb::module_& m) {
    // Expose from_dlpack as a module function
    m.def("from_dlpack", &from_dlpack, "obj"_a);
    m.def("to_dlpack", &to_dlpack, "obj"_a, "stream"_a = nb::none());

    nb::class_<Tensor> tensor(m, "TensorBase");
    tensor.attr("__module__") = "tensorplay._C";
    
    tensor
        .def(nb::init<>())
        // Constructor from data (torch.tensor equivalent)
        .def("__init__", [](Tensor* self, nb::object data, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
            Tensor t = create_tensor(data, dtype, device);
            t.set_requires_grad(requires_grad);
            new (self) Tensor(std::move(t));
        }, "data"_a, "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false)
        
        // Properties
        .def_prop_ro("shape", &Tensor::shape)
        .def_prop_ro("dtype", &Tensor::dtype)
        .def_prop_ro("device", &Tensor::device)
        .def_prop_ro("ndim", &Tensor::dim)
        .def("dim", &Tensor::dim)
        .def("numel", &Tensor::numel)
        .def("itemsize", &Tensor::itemsize)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def_prop_ro("is_sparse", &Tensor::is_sparse)
        .def_prop_ro("strides", [](const Tensor& self) {
            return nb::tuple(nb::cast(self.strides()));
        })
        .def("stride", [](const Tensor& self) {
            return nb::tuple(nb::cast(self.strides()));
        })
        .def("stride", [](const Tensor& self, int64_t dim) {
            return self.stride(dim);
        })
        .def_prop_rw("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def_prop_ro("is_leaf", &Tensor::is_leaf)
        .def_prop_ro("grad_fn", [](const Tensor& self) { return self.grad_fn(); })
        .def_prop_ro("is_cuda", [](const Tensor& self) { return self.device().type() == DeviceType::CUDA; })
        .def_prop_rw("grad", 
            [](const Tensor& self) -> std::optional<Tensor> {
                Tensor g = self.grad();
                if (g.core().defined()) return g;
                return std::nullopt;
            },
            [](Tensor& self, const Tensor* grad) {
                if (grad) {
                    self.set_grad(*grad);
                } else {
                    self.set_grad(Tensor());
                }
            },
            nb::arg("grad").none()
        )
        .def("retain_grad", &Tensor::retain_grad)
        .def("backward", [](Tensor& self, std::optional<Tensor> gradient, bool retain_graph, bool create_graph) {
             if (gradient) {
                 tensorplay::tpx::backward(self, *gradient, retain_graph, create_graph);
             } else {
                 tensorplay::tpx::backward(self, Tensor(), retain_graph, create_graph);
             } 
        }, "gradient"_a = nb::none(), "retain_graph"_a = false, "create_graph"_a = false)
        .def_prop_rw("data", 
            [](const Tensor& self) { return self.detach(); },
            [](Tensor& self, const Tensor& other) {
                if (!self.core().defined() || !other.core().defined()) {
                    self = other;
                    return;
                }
                // Update underlying TensorImpl data/metadata in-place
                // This ensures other references (like p.grad) see the change
                self.core().unsafeGetTensorImpl()->copy_metadata_from(*other.core().unsafeGetTensorImpl());
            }
        )
        .def("detach", &Tensor::detach)
        .def("detach_", [](nb::object self_obj) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.set_requires_grad(false);
            if (self.core().unsafeGetTensorImpl()) self.set_grad_fn(nullptr);
            return self_obj;
        })
        .def("clone", &Tensor::clone)
        .def("requires_grad_", [](nb::object self_obj, bool requires_grad) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.set_requires_grad(requires_grad);
            return self_obj;
        }, "requires_grad"_a = true)
        
        // Methods
        .def("size", [](const Tensor& self) {
            return self.shape();
        })
        .def("size", [](const Tensor& self, int64_t dim) {
            return self.size(dim);
        })
        .def("view", [](const Tensor& self, nb::args args) {
            std::vector<int64_t> shape;
            if (args.size() == 1 && (nb::isinstance<nb::list>(args[0]) || nb::isinstance<nb::tuple>(args[0]) || nb::isinstance<Size>(args[0]))) {
                nb::object obj = args[0];
                for (auto item : obj) {
                    shape.push_back(nb::cast<int64_t>(item));
                }
            } else {
                for (auto item : args) {
                    shape.push_back(nb::cast<int64_t>(item));
                }
            }
            return self.view(shape);
        })
        .def("reshape", [](const Tensor& self, nb::args args) {
            std::vector<int64_t> shape;
            if (args.size() == 1 && (nb::isinstance<nb::list>(args[0]) || nb::isinstance<nb::tuple>(args[0]) || nb::isinstance<Size>(args[0]))) {
                nb::object obj = args[0];
                for (auto item : obj) {
                    shape.push_back(nb::cast<int64_t>(item));
                }
            } else {
                for (auto item : args) {
                    shape.push_back(nb::cast<int64_t>(item));
                }
            }
            return self.reshape(shape);
        })
        .def("expand", &Tensor::expand)
        .def("as_strided", &Tensor::as_strided, "size"_a, "stride"_a, "storage_offset"_a = nb::none())
        .def("select", &Tensor::select, "dim"_a, "index"_a)
        .def("slice", &Tensor::slice, "dim"_a, "start"_a, "end"_a, "step"_a = 1)
        .def("copy_", [](nb::object self_obj, const Tensor& src) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.copy_(src);
            return self_obj;
        }, "src"_a)
        .def("fill_", [](nb::object self_obj, Scalar value) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.fill_(value);
            return self_obj;
        }, "value"_a)
        .def("zero_", [](nb::object self_obj) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.zero_();
            return self_obj;
        })
        // In-place random sampling
        .def("bernoulli_", [](nb::object self_obj) {
             nb::cast<Tensor&>(self_obj).bernoulli_();
             return self_obj;
        })
        .def("cauchy_", [](nb::object self_obj, double median, double sigma) {
             nb::cast<Tensor&>(self_obj).cauchy_(median, sigma);
             return self_obj;
        }, "median"_a = 0.0, "sigma"_a = 1.0)
        .def("exponential_", [](nb::object self_obj, double lambd) {
             nb::cast<Tensor&>(self_obj).exponential_(lambd);
             return self_obj;
        }, "lambd"_a = 1.0)
        .def("geometric_", [](nb::object self_obj, double p) {
             nb::cast<Tensor&>(self_obj).geometric_(p);
             return self_obj;
        }, "p"_a)
        .def("log_normal_", [](nb::object self_obj, double mean, double std) {
             nb::cast<Tensor&>(self_obj).log_normal_(mean, std);
             return self_obj;
        }, "mean"_a = 1.0, "std"_a = 2.0)
        .def("normal_", [](nb::object self_obj, double mean, double std) {
             nb::cast<Tensor&>(self_obj).normal_(mean, std);
             return self_obj;
        }, "mean"_a = 0.0, "std"_a = 1.0)
        .def("random_", [](nb::object self_obj, int64_t low, int64_t high) {
             nb::cast<Tensor&>(self_obj).random_(low, high);
             return self_obj;
        }, "low"_a = 0, "high"_a = 0)
        .def("uniform_", [](nb::object self_obj, double from, double to) {
             nb::cast<Tensor&>(self_obj).uniform_(from, to);
             return self_obj;
        }, "from"_a = 0.0, "to"_a = 1.0);
        
        // Bind generated methods
        bind_generated_tensor_methods(tensor);
        
        tensor.def("sum", [](const Tensor& self, std::optional<DType> dtype) {
            return self.sum(dtype.value_or(DType::Undefined));
        }, nb::kw_only(), "dtype"_a = nb::none())
        .def("sum", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype) {
            return self.sum(dim, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none())
        .def("sum", [](const Tensor& self, int64_t dim, bool keepdim, std::optional<DType> dtype) {
            return self.sum({dim}, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none())

        .def("mean", [](const Tensor& self, std::optional<DType> dtype) {
            return self.mean(dtype.value_or(DType::Undefined));
        }, nb::kw_only(), "dtype"_a = nb::none())
        .def("mean", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype) {
            return self.mean(dim, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none())
        .def("mean", [](const Tensor& self, int64_t dim, bool keepdim, std::optional<DType> dtype) {
            return self.mean({dim}, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none())

        .def("prod", [](const Tensor& self, std::optional<DType> dtype) {
            return self.prod(dtype.value_or(DType::Undefined));
        }, nb::kw_only(), "dtype"_a = nb::none())
        .def("prod", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype) {
            return self.prod(dim, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none())
        .def("prod", [](const Tensor& self, int64_t dim, bool keepdim, std::optional<DType> dtype) {
            return self.prod({dim}, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none())

        .def("all", [](const Tensor& self) { return self.all(); })
        .def("all", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.all(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("all", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.all({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("any", [](const Tensor& self) { return self.any(); })
        .def("any", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.any(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("any", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.any({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("argmax", [](const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
            return self.argmax(dim, keepdim);
        }, "dim"_a = nb::none(), "keepdim"_a = false)
        .def("argmin", [](const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
            return self.argmin(dim, keepdim);
        }, "dim"_a = nb::none(), "keepdim"_a = false)

        .def("var", [](const Tensor& self, int64_t correction) {
            return self.var(correction);
        }, "correction"_a = 1)
        .def("var", [](const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
            return self.var(dim, correction, keepdim);
        }, "dim"_a, "correction"_a = 1, "keepdim"_a = false)
        .def("var", [](const Tensor& self, int64_t dim, int64_t correction, bool keepdim) {
            return self.var({dim}, correction, keepdim);
        }, "dim"_a, "correction"_a = 1, "keepdim"_a = false)

        .def("std", [](const Tensor& self, int64_t correction) {
            return self.std(correction);
        }, "correction"_a = 1)
        .def("std", [](const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
            return self.std(dim, correction, keepdim);
        }, "dim"_a, "correction"_a = 1, "keepdim"_a = false)
        .def("std", [](const Tensor& self, int64_t dim, int64_t correction, bool keepdim) {
            return self.std({dim}, correction, keepdim);
        }, "dim"_a, "correction"_a = 1, "keepdim"_a = false)

        .def("norm", [](const Tensor& self, double p) {
            return self.norm(p);
        }, "p"_a = 2.0)
        .def("norm", [](const Tensor& self, const std::vector<int64_t>& dim, double p, bool keepdim) {
            return self.norm(dim, p, keepdim);
        }, "dim"_a, "p"_a = 2.0, "keepdim"_a = false)
        .def("norm", [](const Tensor& self, int64_t dim, double p, bool keepdim) {
            return self.norm({dim}, p, keepdim);
        }, "dim"_a, "p"_a = 2.0, "keepdim"_a = false)

        .def("max", [](const Tensor& self) { return self.max(); })
        .def("max", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.max(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("max", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.max({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("min", [](const Tensor& self) { return self.min(); })
        .def("min", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.min(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("min", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.min({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("pow", [](const Tensor& self, Scalar exponent) {
            return self.pow(exponent);
        }, "exponent"_a)
        .def("sqrt", [](const Tensor& self) { return self.sqrt(); })
        .def("abs", [](const Tensor& self) { return self.abs(); })

        .def("max", [](const Tensor& self) {
            return self.max();
        })
        .def("max", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.max(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("max", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.max({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("min", [](const Tensor& self) {
            return self.min();
        })
        .def("min", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.min(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("min", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.min({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        // Manual overloads using lambdas
        .def("to", [](const Tensor& self, DType dtype, bool non_blocking, bool copy) {
            return self.to(dtype, non_blocking, copy);
        }, "dtype"_a, "non_blocking"_a = false, "copy"_a = false)
        .def("to", [](const Tensor& self, Device device, bool non_blocking, bool copy) {
            return self.to(device, non_blocking, copy);
        }, "device"_a, "non_blocking"_a = false, "copy"_a = false)
        .def("to", [](const Tensor& self, Device device, DType dtype, bool non_blocking, bool copy) {
            return self.to(device, dtype, non_blocking, copy);
        }, "device"_a, "dtype"_a, "non_blocking"_a = false, "copy"_a = false)

        .def("__array__", [](nb::object self_obj, nb::object dtype) {
            try {
                nb::module_ np = nb::module_::import_("numpy");
                // Delegate to from_dlpack which is zero-copy and efficient
                nb::object arr = np.attr("from_dlpack")(self_obj);
                if (!dtype.is_none()) {
                    return arr.attr("astype")(dtype, "copy"_a = false);
                }
                return arr;
            } catch (const std::exception&) {
                TP_THROW(RuntimeError, "numpy is not installed or cannot be imported.");
            }
        }, "dtype"_a = nb::none())

        .def("numpy", [](nb::object self_obj) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            if (self.requires_grad()) {
                TP_THROW(RuntimeError, "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.");
            }
            if (self.device().type() != DeviceType::CPU) {
                TP_THROW(RuntimeError, "Can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.");
            }
            // Optimization: Directly create nb::ndarray using DLPack type codes
            // This avoids Python-side np.asarray and dlpack capsule overhead completely.
            
            DType dtype = self.dtype();
            nb::dlpack::dtype dt;
            dt.lanes = 1;
            
            switch (dtype) {
                case DType::Float32: dt.code = 2; dt.bits = 32; break; // kDLFloat
                case DType::Float64: dt.code = 2; dt.bits = 64; break;
                case DType::Int32:   dt.code = 0; dt.bits = 32; break; // kDLInt
                case DType::Int64:   dt.code = 0; dt.bits = 64; break;
                case DType::Int8:    dt.code = 0; dt.bits = 8;  break;
                case DType::Int16:   dt.code = 0; dt.bits = 16; break;
                case DType::UInt8:   dt.code = 1; dt.bits = 8;  break; // kDLUInt
                case DType::UInt16:  dt.code = 1; dt.bits = 16; break;
                case DType::UInt32:  dt.code = 1; dt.bits = 32; break;
                case DType::UInt64:  dt.code = 1; dt.bits = 64; break;
                case DType::Bool:    dt.code = 6; dt.bits = 8;  break; // kDLBool (or UInt8=1)
                default: TP_THROW(RuntimeError, "Unsupported DType for NumPy conversion");
            }

            auto sizes = self.shape();
            std::vector<size_t> shape(sizes.begin(), sizes.end());
            std::vector<int64_t> strides_int64 = self.strides();
            std::vector<int64_t> strides_bytes;
            strides_bytes.reserve(strides_int64.size());
            
            // NumPy strides are in bytes, TensorPlay strides are in elements
            // size_t itemsize = dt.bits / 8;
            // for (auto s : strides_int64) {
            //     strides_bytes.push_back(s * itemsize);
            // }

            // Nanobind seems to expect strides in elements for nb::ndarray ?? 
            // Or maybe it expects bytes but we are getting double multiplication?
            // Let's try passing element strides directly.
            // Update: nanobind ndarray constructor for numpy backend might be taking element strides if it constructs via array interface or similar?
            // Actually, let's just stick to element strides if bytes resulted in 4x.

            // Create nb::ndarray
            // owner = self_obj to keep the tensor alive
            return nb::ndarray<nb::numpy>(
                self.data_ptr(),
                shape.size(),
                shape.data(),
                self_obj,
                strides_int64.data(),
                dt
            );
        })
        
        .def("item", [](const Tensor& self) -> nb::object {
            switch (self.dtype()) {
                case DType::Float32: return nb::float_(self.item().to<float>());
                case DType::Float64: return nb::float_(self.item().to<double>());
                case DType::Int32: return nb::int_(self.item().to<int32_t>());
                case DType::Int64: return nb::int_(self.item().to<int64_t>());
                case DType::Bool: return nb::bool_(self.item().to<bool>());
                default: TP_THROW(NotImplementedError, "item() not implemented for this dtype");
            }
        })
        
        // Indexing
        .def("tolist", [](const Tensor& self) -> nb::object {
            if (self.device().type() != DeviceType::CPU) {
                 TP_THROW(RuntimeError, "tolist() is only supported on CPU tensors");
            }

            auto get_dtype_size = [](DType dtype) -> size_t {
                switch (dtype) {
                    case DType::Float32: return 4;
                    case DType::Float64: return 8;
                    case DType::Int32:   return 4;
                    case DType::Int64:   return 8;
                    case DType::Int8:    return 1;
                    case DType::Int16:   return 2;
                    case DType::UInt8:   return 1;
                    case DType::UInt16:  return 2;
                    case DType::UInt32:  return 4;
                    case DType::UInt64:  return 8;
                    case DType::Bool:    return 1;
                    default: return 1;
                }
            };
            
            if (self.dim() == 0) {
                // Reuse item() logic
                switch (self.dtype()) {
                    case DType::Float32: return nb::float_(self.item().to<float>());
                    case DType::Float64: return nb::float_(self.item().to<double>());
                    case DType::Int32: return nb::int_(self.item().to<int32_t>());
                    case DType::Int64: return nb::int_(self.item().to<int64_t>());
                    case DType::Bool: return nb::bool_(self.item().to<bool>());
                    case DType::Int8: return nb::int_(self.item().to<int8_t>());
                    case DType::Int16: return nb::int_(self.item().to<int16_t>());
                    case DType::UInt8: return nb::int_(self.item().to<uint8_t>());
                    case DType::UInt16: return nb::int_(self.item().to<uint16_t>());
                    case DType::UInt32: return nb::int_(self.item().to<uint32_t>());
                    case DType::UInt64: return nb::int_(self.item().to<uint64_t>());
                    default: TP_THROW(NotImplementedError, "tolist() not implemented for this dtype");
                }
            }

            // Recursive helper lambda
            auto recurse = [&](auto&& self_recurse, const void* data, int64_t ndim, const int64_t* sizes, const int64_t* strides, DType dtype) -> nb::object {
                int64_t size = sizes[0];
                int64_t stride = strides[0];
                nb::list result;
                size_t itemsize = get_dtype_size(dtype);

                if (ndim == 1) {
                    for (int64_t i = 0; i < size; ++i) {
                        const char* ptr = static_cast<const char*>(data) + i * stride * itemsize;
                        switch (dtype) {
                            case DType::Float32: result.append(*reinterpret_cast<const float*>(ptr)); break;
                            case DType::Float64: result.append(*reinterpret_cast<const double*>(ptr)); break;
                            case DType::Int32:   result.append(*reinterpret_cast<const int32_t*>(ptr)); break;
                            case DType::Int64:   result.append(*reinterpret_cast<const int64_t*>(ptr)); break;
                            case DType::Bool:    result.append(*reinterpret_cast<const bool*>(ptr)); break;
                            case DType::Int8:    result.append(*reinterpret_cast<const int8_t*>(ptr)); break;
                            case DType::Int16:   result.append(*reinterpret_cast<const int16_t*>(ptr)); break;
                            case DType::UInt8:   result.append(*reinterpret_cast<const uint8_t*>(ptr)); break;
                            case DType::UInt16:  result.append(*reinterpret_cast<const uint16_t*>(ptr)); break;
                            case DType::UInt32:  result.append(*reinterpret_cast<const uint32_t*>(ptr)); break;
                            case DType::UInt64:  result.append(*reinterpret_cast<const uint64_t*>(ptr)); break;
                            default: TP_THROW(NotImplementedError, "tolist() not implemented for this dtype");
                        }
                    }
                } else {
                    for (int64_t i = 0; i < size; ++i) {
                        const char* ptr = static_cast<const char*>(data) + i * stride * itemsize;
                        result.append(self_recurse(self_recurse, ptr, ndim - 1, sizes + 1, strides + 1, dtype));
                    }
                }
                return result;
            };

            std::vector<int64_t> shape_vec = static_cast<std::vector<int64_t>>(self.shape());
            std::vector<int64_t> strides_vec = self.strides();
            
            return recurse(recurse, self.data_ptr(), self.dim(), shape_vec.data(), strides_vec.data(), self.dtype());
        })

        .def("__getitem__", [](const Tensor& self, nb::object index) -> Tensor {
            if (nb::isinstance<Tensor>(index)) {
            Tensor idx = nb::cast<Tensor>(index);
            if (idx.dtype() == DType::Bool) {
                return self.masked_select(idx);
            }
        }
            if (nb::isinstance<nb::tuple>(index)) {
                 nb::tuple indices = nb::cast<nb::tuple>(index);
                 Tensor result = self;
                 int64_t target_dim = 0;
                 for (size_t i = 0; i < indices.size(); ++i) {
                     nb::object idx = indices[i];
                     if (nb::isinstance<nb::int_>(idx)) {
                         int64_t val = nb::cast<int64_t>(idx);
                         result = result.select(target_dim, val);
                     } else if (nb::isinstance<nb::slice>(idx)) {
                         nb::slice s = nb::cast<nb::slice>(idx);
                         auto [start, stop, step, slicelength] = s.compute(result.size(target_dim));
                         result = result.slice(target_dim, start, stop, step);
                         target_dim++;
                     } else {
                         TP_THROW(TypeError, "Unsupported index type in tuple");
                     }
                 }
                 return result;
            } else if (nb::isinstance<nb::int_>(index)) {
                return self.select(0, nb::cast<int64_t>(index));
            } else if (nb::isinstance<nb::slice>(index)) {
                nb::slice s = nb::cast<nb::slice>(index);
                auto [start, stop, step, slicelength] = s.compute(self.size(0));
                return self.slice(0, start, stop, step);
            }
            TP_THROW(TypeError, "Unsupported index type");
        })
        .def("__setitem__", [](Tensor& self, nb::object index, nb::object value) {
            Tensor target;
            if (nb::isinstance<nb::tuple>(index)) {
                 nb::tuple indices = nb::cast<nb::tuple>(index);
                 target = self;
                 int64_t target_dim = 0;
                 for (size_t i = 0; i < indices.size(); ++i) {
                     nb::object idx = indices[i];
                     if (nb::isinstance<nb::int_>(idx)) {
                         int64_t val = nb::cast<int64_t>(idx);
                         target = target.select(target_dim, val);
                     } else if (nb::isinstance<nb::slice>(idx)) {
                         nb::slice s = nb::cast<nb::slice>(idx);
                         auto [start, stop, step, slicelength] = s.compute(target.size(target_dim));
                         target = target.slice(target_dim, start, stop, step);
                         target_dim++;
                     } else {
                         TP_THROW(TypeError, "Unsupported index type in tuple");
                     }
                 }
            } else if (nb::isinstance<nb::int_>(index)) {
                target = self.select(0, nb::cast<int64_t>(index));
            } else if (nb::isinstance<nb::slice>(index)) {
                nb::slice s = nb::cast<nb::slice>(index);
                auto [start, stop, step, slicelength] = s.compute(self.size(0));
                target = self.slice(0, start, stop, step);
            } else {
                TP_THROW(TypeError, "Unsupported index type");
            }

            if (nb::isinstance<Tensor>(value)) {
                target.copy_(nb::cast<Tensor>(value));
            } else {
                try {
                    // Try to cast to scalar (float/int/bool)
                    if (nb::isinstance<nb::float_>(value) || nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
                         double v = nb::cast<double>(value);
                         target.fill_(Scalar(v));
                    } else {
                         TP_THROW(TypeError, "Unsupported value type for setitem");
                    }
                } catch (...) {
                    TP_THROW(TypeError, "Unsupported value type for setitem");
                }
            }
        })
        
        // Operators
        .def(-nb::self)
        .def(nb::self + nb::self)
        .def(nb::self + float())
        .def(nb::self - nb::self)
        .def(nb::self - float())
        .def(nb::self * nb::self)
        .def(nb::self * float())
        .def(nb::self / nb::self)
        .def(nb::self / float())
        .def(nb::self += nb::self)
        .def(nb::self += float())
        .def(nb::self -= nb::self)
        .def(nb::self -= float())
        .def(nb::self *= nb::self)
        .def(nb::self *= float())
        .def(nb::self /= nb::self)
        .def(nb::self /= float())
        
        .def("__add__", [](const Tensor& t, double s) { return t + Scalar(s); })
        .def("__sub__", [](const Tensor& t, double s) { return t - Scalar(s); })
        .def("__mul__", [](const Tensor& t, double s) { return t * Scalar(s); })
        .def("__truediv__", [](const Tensor& t, double s) { return t / Scalar(s); })
        .def("__radd__", [](const Tensor& t, double s) { return Scalar(s) + t; })
        .def("__rsub__", [](const Tensor& t, double s) { return Scalar(s) - t; })
        .def("__rmul__", [](const Tensor& t, double s) { return Scalar(s) * t; })
        
        // Explicit arithmetic
        .def("add", [](const Tensor& self, const Tensor& other, Scalar alpha) {
            return self.add(other, alpha);
        }, "other"_a, "alpha"_a = Scalar(1))
        .def("add", [](const Tensor& self, Scalar other, Scalar alpha) {
            Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
            return self.add(other_t, alpha);
        }, "other"_a, "alpha"_a = Scalar(1))
        .def("add_", [](nb::object self_obj, const Tensor& other, Scalar alpha) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.add_(other, alpha);
            return self_obj;
        }, "other"_a, "alpha"_a = Scalar(1))
        .def("add_", [](nb::object self_obj, Scalar other, Scalar alpha) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
            self.add_(other_t, alpha);
            return self_obj;
        }, "other"_a, "alpha"_a = Scalar(1))
        .def("sub", [](const Tensor& self, const Tensor& other, Scalar alpha) {
            return self.sub(other, alpha);
        }, "other"_a, "alpha"_a = Scalar(1))
        .def("sub", [](const Tensor& self, Scalar other, Scalar alpha) {
            Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
            return self.sub(other_t, alpha);
        }, "other"_a, "alpha"_a = Scalar(1))
        .def("sub_", [](nb::object self_obj, const Tensor& other, Scalar alpha) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.sub_(other, alpha);
            return self_obj;
        }, "other"_a, "alpha"_a = Scalar(1))
        .def("sub_", [](nb::object self_obj, Scalar other, Scalar alpha) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
            self.sub_(other_t, alpha);
            return self_obj;
        }, "other"_a, "alpha"_a = Scalar(1))
        .def("mul", [](const Tensor& self, const Tensor& other) {
            return self.mul(other);
        }, "other"_a)
        .def("mul", [](const Tensor& self, Scalar other) {
            Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
            return self.mul(other_t);
        }, "other"_a)
        .def("mul_", [](nb::object self_obj, const Tensor& other) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.mul_(other);
            return self_obj;
        }, "other"_a)
        .def("mul_", [](nb::object self_obj, Scalar other) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
            self.mul_(other_t);
            return self_obj;
        }, "other"_a)
        .def("div", [](const Tensor& self, const Tensor& other) {
            return self.div(other);
        }, "other"_a)
        .def("div", [](const Tensor& self, Scalar other) {
            Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
            return self.div(other_t);
        }, "other"_a)
        .def("div_", [](nb::object self_obj, const Tensor& other) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.div_(other);
            return self_obj;
        }, "other"_a)
        .def("div_", [](nb::object self_obj, Scalar other) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            Tensor other_t = Tensor::full({}, other, self.dtype(), self.device());
            self.div_(other_t);
            return self_obj;
        }, "other"_a)
        .def("bernoulli", static_cast<Tensor(Tensor::*)() const>(&Tensor::bernoulli))
        .def("poisson", static_cast<Tensor(Tensor::*)() const>(&Tensor::poisson))
        .def("mm", [](const Tensor& self, const Tensor& other) { return self.mm(other); }, "other"_a)
        .def("matmul", [](const Tensor& self, const Tensor& other) { return self.matmul(other); }, "other"_a)
        .def("__matmul__", [](const Tensor& self, const Tensor& other) { return self.matmul(other); }, "other"_a)

        // Comparison operators
        .def("__eq__", [](const Tensor& self, const Tensor& other) { return self.eq(other); })
        .def("__eq__", [](const Tensor& self, Scalar other) { return self.eq(other); })
        .def("__ne__", [](const Tensor& self, const Tensor& other) { return self.ne(other); })
        .def("__ne__", [](const Tensor& self, Scalar other) { return self.ne(other); })
        .def("__lt__", [](const Tensor& self, const Tensor& other) { return self.lt(other); })
        .def("__lt__", [](const Tensor& self, Scalar other) { return self.lt(other); })
        .def("__le__", [](const Tensor& self, const Tensor& other) { return self.le(other); })
        .def("__le__", [](const Tensor& self, Scalar other) { return self.le(other); })
        .def("__gt__", [](const Tensor& self, const Tensor& other) { return self.gt(other); })
        .def("__gt__", [](const Tensor& self, Scalar other) { return self.gt(other); })
        .def("__ge__", [](const Tensor& self, const Tensor& other) { return self.ge(other); })
        .def("__ge__", [](const Tensor& self, Scalar other) { return self.ge(other); })

        // Pointwise ops
        .def("abs", static_cast<Tensor(Tensor::*)() const>(&Tensor::abs))
        .def("acos", static_cast<Tensor(Tensor::*)() const>(&Tensor::acos))
        .def("acosh", static_cast<Tensor(Tensor::*)() const>(&Tensor::acosh))
        .def("angle", static_cast<Tensor(Tensor::*)() const>(&Tensor::angle))
        .def("asin", static_cast<Tensor(Tensor::*)() const>(&Tensor::asin))
        .def("asinh", static_cast<Tensor(Tensor::*)() const>(&Tensor::asinh))
        .def("atan", static_cast<Tensor(Tensor::*)() const>(&Tensor::atan))
        .def("atanh", static_cast<Tensor(Tensor::*)() const>(&Tensor::atanh))
        .def("ceil", static_cast<Tensor(Tensor::*)() const>(&Tensor::ceil))
        .def("clamp", static_cast<Tensor(Tensor::*)(std::optional<Scalar>, std::optional<Scalar>) const>(&Tensor::clamp), "min"_a = nb::none(), "max"_a = nb::none())
        .def("cos", static_cast<Tensor(Tensor::*)() const>(&Tensor::cos))
        .def("cosh", static_cast<Tensor(Tensor::*)() const>(&Tensor::cosh))
        .def("exp", static_cast<Tensor(Tensor::*)() const>(&Tensor::exp))
        .def("floor", static_cast<Tensor(Tensor::*)() const>(&Tensor::floor))
        .def("gelu", static_cast<Tensor(Tensor::*)() const>(&Tensor::gelu))
        .def("lerp", static_cast<Tensor(Tensor::*)(const Tensor&, Scalar) const>(&Tensor::lerp), "end"_a, "weight"_a)
        .def("lerp", static_cast<Tensor(Tensor::*)(const Tensor&, const Tensor&) const>(&Tensor::lerp), "end"_a, "weight"_a)
        .def("log", static_cast<Tensor(Tensor::*)() const>(&Tensor::log))
        .def("neg", static_cast<Tensor(Tensor::*)() const>(&Tensor::neg))
        .def("pow", static_cast<Tensor(Tensor::*)(Scalar) const>(&Tensor::pow), "exponent"_a)
        .def("pow", static_cast<Tensor(Tensor::*)(const Tensor&) const>(&Tensor::pow), "exponent"_a)
        .def("__pow__", static_cast<Tensor(Tensor::*)(Scalar) const>(&Tensor::pow), "exponent"_a)
        .def("__pow__", static_cast<Tensor(Tensor::*)(const Tensor&) const>(&Tensor::pow), "exponent"_a)
        .def("__rpow__", [](const Tensor& self, Scalar base) {
            Tensor base_t = Tensor::full({}, base, self.dtype(), self.device());
            return base_t.pow(self);
        })
        .def("relu", static_cast<Tensor(Tensor::*)() const>(&Tensor::relu))
        .def("round", static_cast<Tensor(Tensor::*)() const>(&Tensor::round))
        .def("rsqrt", static_cast<Tensor(Tensor::*)() const>(&Tensor::rsqrt))
        .def("sigmoid", static_cast<Tensor(Tensor::*)() const>(&Tensor::sigmoid))
        .def("silu", static_cast<Tensor(Tensor::*)() const>(&Tensor::silu))
        .def("sign", static_cast<Tensor(Tensor::*)() const>(&Tensor::sign))
        .def("sin", static_cast<Tensor(Tensor::*)() const>(&Tensor::sin))
        .def("sinh", static_cast<Tensor(Tensor::*)() const>(&Tensor::sinh))
        .def("softmax", static_cast<Tensor(Tensor::*)(int64_t, DType) const>(&Tensor::softmax), "dim"_a, "dtype"_a = DType::Undefined)
        .def("sqrt", static_cast<Tensor(Tensor::*)() const>(&Tensor::sqrt))
        .def("square", static_cast<Tensor(Tensor::*)() const>(&Tensor::square))
        .def("tan", static_cast<Tensor(Tensor::*)() const>(&Tensor::tan))
        .def("tanh", static_cast<Tensor(Tensor::*)() const>(&Tensor::tanh))

        // DLPack
        .def("__dlpack__", [](nb::object self_obj, std::optional<int64_t> stream) {
            return to_dlpack(self_obj, stream);
        }, "stream"_a = nb::none())
        .def("__dlpack_device__", [](const Tensor& self) {
            DLDevice d = to_dlpack_device(self.device());
            return nb::make_tuple(d.device_type, d.device_id);
        })

        // String repr
        .def("__repr__", &Tensor::toString)
        .def("__str__", &Tensor::toString);
}