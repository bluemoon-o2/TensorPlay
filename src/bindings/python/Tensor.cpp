#include "python_bindings.h"
#include "utils.h"

using namespace tensorplay::python;

// Helper function implementation
Tensor create_tensor(nb::object data, std::optional<DType> dtype, std::optional<Device> device) {
    Tensor t;
    
    // Handle nb::ndarray (NumPy array)
    if (nb::isinstance<nb::ndarray<>>(data)) {
         nb::ndarray<> array = nb::cast<nb::ndarray<>>(data);
         
         std::vector<int64_t> shape;
         for (size_t i = 0; i < array.ndim(); ++i) {
             shape.push_back(static_cast<int64_t>(array.shape(i)));
         }
         
         nb::dlpack::dtype dt = array.dtype();
         DType inferred_dtype = DType::Undefined;
         
         // Map dlpack dtype to TensorPlay DType
         // dlpack codes: 0=Int, 1=UInt, 2=Float, 3=Opaque, 4=Bfloat, 5=Complex, 6=Bool
         uint8_t code = dt.code;
         uint8_t bits = dt.bits;
         
         if (code == 2 && bits == 32) inferred_dtype = DType::Float32;
         else if (code == 2 && bits == 64) inferred_dtype = DType::Float64;
         else if (code == 0 && bits == 32) inferred_dtype = DType::Int32;
         else if (code == 0 && bits == 64) inferred_dtype = DType::Int64;
         else if (code == 1 && bits == 32) inferred_dtype = DType::UInt32;
         else if (code == 1 && bits == 64) inferred_dtype = DType::UInt64;
         else if (code == 6 || (code == 1 && bits == 8)) inferred_dtype = DType::Bool; // Bool or UInt8 (often used for bool)
         else {
             // Try to guess from itemsize if simple type
             TP_THROW(TypeError, "Unsupported NumPy/DLPack dtype: code=" + std::to_string(code) + ", bits=" + std::to_string(bits));
         }
         
         DType final_dtype = dtype.value_or(inferred_dtype);
         t = Tensor(shape, final_dtype, device.value_or(Device(DeviceType::CPU)));
         
         // Copy data
         size_t numel = 1;
         for(auto s : shape) numel *= s;
         size_t total_bytes = numel * (bits / 8); // simplified check
         
         if (final_dtype == inferred_dtype) {
             std::memcpy(t.data_ptr(), array.data(), total_bytes);
         } else {
             // Create a temporary tensor with the inferred dtype to hold data
             Tensor src(shape, inferred_dtype, Device(DeviceType::CPU));
             std::memcpy(src.data_ptr(), array.data(), total_bytes);
             
             // Copy to the destination tensor (copy_ handles casting)
             t.copy_(src);
         }
         
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
        Tensor new_t(static_cast<std::vector<int64_t>>(t.shape()), *dtype, device.value_or(Device(DeviceType::CPU)));
        convert_tensor_data(t, new_t);
        t = new_t;
    }
    
    // Handle device movement if needed
    // Note: list_to_tensor returns CPU tensor.
    if (device.has_value() && t.device() != *device) {
        Tensor new_t(static_cast<std::vector<int64_t>>(t.shape()), t.dtype(), *device);
        new_t.copy_(t);
        t = new_t;
    }

    return t;
}

void init_tensor(nb::module_& m) {
    nb::class_<Tensor> tensor(m, "TensorBase");
    
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
        .def_prop_rw("requires_grad", &Tensor::requires_grad, &Tensor::set_requires_grad)
        .def_prop_ro("is_leaf", &Tensor::is_leaf)
        .def_prop_ro("grad_fn", [](const Tensor& self) { return nb::none(); })
        .def_prop_ro("is_cuda", [](const Tensor& self) { return self.device().type() == DeviceType::CUDA; })
        .def_prop_rw("grad", 
            [](const Tensor& self) -> std::optional<Tensor> {
                Tensor g = self.grad();
                if (g.defined()) return g;
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
        .def("detach", &Tensor::detach)
        .def("detach_", [](nb::object self_obj) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            self.set_requires_grad(false);
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
        }, "from"_a = 0.0, "to"_a = 1.0)
        .def("transpose", &Tensor::transpose, "dim0"_a, "dim1"_a)
        .def("t", &Tensor::t)
        .def("permute", &Tensor::permute, "dims"_a)
        .def("squeeze", nb::overload_cast<>(&Tensor::squeeze, nb::const_))
        .def("squeeze", nb::overload_cast<int64_t>(&Tensor::squeeze, nb::const_), "dim"_a)
        .def("unsqueeze", &Tensor::unsqueeze, "dim"_a)
        .def("split", nb::overload_cast<int64_t, int64_t>(&Tensor::split, nb::const_), "split_size"_a, "dim"_a = 0)
        .def("split", nb::overload_cast<const std::vector<int64_t>&, int64_t>(&Tensor::split, nb::const_), "split_sizes"_a, "dim"_a = 0)
        .def("chunk", &Tensor::chunk, "chunks"_a, "dim"_a = 0)
        .def("unbind", &Tensor::unbind, "dim"_a = 0)
        
        .def("sum", [](const Tensor& self, std::optional<DType> dtype) {
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

        .def("numpy", [](nb::object self_obj) {
            Tensor& self = nb::cast<Tensor&>(self_obj);
            if (self.device().type() != DeviceType::CPU) {
                TP_THROW(RuntimeError, "Can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.");
            }
            
            size_t ndim = self.dim();
            std::vector<size_t> shape(ndim);
            std::vector<int64_t> strides(ndim);
            size_t itemsize = self.itemsize();
            
            for(size_t i=0; i<ndim; ++i) {
                shape[i] = static_cast<size_t>(self.size(i));
                strides[i] = self.stride(i);
            }
            
            nb::dlpack::dtype dt;
            switch(self.dtype()) {
                case DType::Float32: dt = {2, 32, 1}; break;
                case DType::Float64: dt = {2, 64, 1}; break;
                case DType::Int32: dt = {0, 32, 1}; break;
                case DType::Int64: dt = {0, 64, 1}; break;
                case DType::UInt32: dt = {1, 32, 1}; break;
                case DType::UInt64: dt = {1, 64, 1}; break;
                case DType::Bool: dt = {1, 8, 1}; break;
                default: TP_THROW(RuntimeError, "Unsupported dtype for numpy conversion");
            }
            
            return nb::ndarray<nb::numpy>(
                self.data_ptr(),
                ndim,
                shape.data(),
                self_obj,
                strides.data(),
                dt
            );
        })
        
        .def("item", [](const Tensor& self) -> nb::object {
            switch (self.dtype()) {
                case DType::Float32: return nb::float_(self.item<float>());
                case DType::Float64: return nb::float_(self.item<double>());
                case DType::Int32: return nb::int_(self.item<int32_t>());
                case DType::Int64: return nb::int_(self.item<int64_t>());
                case DType::Bool: return nb::bool_(self.item<bool>());
                default: TP_THROW(NotImplementedError, "item() not implemented for this dtype");
            }
        })
        
        // Indexing
        .def("__getitem__", [](const Tensor& self, nb::object index) {
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
        .def("add", &Tensor::add, "other"_a, "alpha"_a = Scalar(1))
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
        .def("sub", &Tensor::sub, "other"_a, "alpha"_a = Scalar(1))
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
        .def("mul", &Tensor::mul, "other"_a)
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
        .def("div", &Tensor::div, "other"_a)
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
        .def("mm", &Tensor::mm, "other"_a)
        .def("matmul", &Tensor::matmul, "other"_a)
        .def("__matmul__", &Tensor::matmul, "other"_a)

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

        // String repr
        .def("__repr__", &Tensor::toString)
        .def("__str__", &Tensor::toString)
        
        // Static factories
        .def_static("empty", &Tensor::empty, "size"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("zeros", &Tensor::zeros, "shape"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("ones", &Tensor::ones, "shape"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("full", &Tensor::full, "shape"_a, "fill_value"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("eye", &Tensor::eye, "n"_a, "m"_a = -1, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("arange", static_cast<Tensor(*)(Scalar, Scalar, Scalar, DType, Device, bool)>(&Tensor::arange), "start"_a, "end"_a, "step"_a = Scalar(1), nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("arange", static_cast<Tensor(*)(Scalar, DType, Device, bool)>(&Tensor::arange), "end"_a, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("rand", &Tensor::rand, "size"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("rand_like", &Tensor::rand_like, "input"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false)
        .def_static("randint", &Tensor::randint, "low"_a, "high"_a, "size"_a, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("randint_like", &Tensor::randint_like, "input"_a, "low"_a, "high"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false)
        .def_static("randn", &Tensor::randn, "size"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("randn_like", &Tensor::randn_like, "input"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false)
        .def_static("randperm", &Tensor::randperm, "n"_a, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("normal", &Tensor::normal, "mean"_a, "std"_a)
        .def_static("linspace", &Tensor::linspace, "start"_a, "end"_a, "steps"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("logspace", &Tensor::logspace, "start"_a, "end"_a, "steps"_a, "base"_a = 10.0, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false)
        .def_static("cat", &Tensor::cat, "tensors"_a, "dim"_a = 0)
        .def_static("stack", &Tensor::stack, "tensors"_a, "dim"_a = 0)
        
        // *_like factories
        .def_static("empty_like", &Tensor::empty_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none())
        .def_static("zeros_like", &Tensor::zeros_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none())
        .def_static("ones_like", &Tensor::ones_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none())
        .def_static("full_like", &Tensor::full_like, "input"_a, "fill_value"_a, "dtype"_a = nb::none(), "device"_a = nb::none());
}