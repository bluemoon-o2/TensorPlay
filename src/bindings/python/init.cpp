#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/operators.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>
#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Device.h"
#include "tensorplay/core/DType.h"
#include "utils.h"

namespace nb = nanobind;
using namespace nb::literals; // for _a
using namespace tensorplay;
using namespace tensorplay::python;

NB_MODULE(tensorplay, m) {
    m.doc() = R"(The tensorplay package offers a simple deep-learning framework 
    designed for educational purposes and small-scale experiments. It defines a 
    data structure for multi-dimensional arrays called Tensor, on which it 
    encapsulates mathematical operations.)";

    // Bind DType enum
    nb::enum_<DType>(m, "DType")
        .value("float32", DType::Float32)
        .value("float64", DType::Float64)
        .value("int32", DType::Int32)
        .value("int64", DType::Int64)
        .value("uint32", DType::UInt32)
        .value("uint64", DType::UInt64)
        .value("bool", DType::Bool)
        .export_values();

    // Bind DeviceType enum
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

    // Bind Device class
    nb::class_<Device>(m, "Device")
        .def(nb::init<DeviceType, int64_t>(), "type"_a, "index"_a = -1)
        .def_prop_ro("type", &Device::type)
        .def_prop_ro("index", &Device::index)
        .def("is_cpu", &Device::is_cpu)
        .def("is_cuda", &Device::is_cuda)
        .def("__repr__", &Device::toString)
        .def("__str__", &Device::toString)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    // Bind Scalar class
    nb::class_<Scalar>(m, "Scalar")
        .def(nb::init<double>())
        .def(nb::init<int64_t>())
        .def(nb::init<bool>())
        .def("__repr__", &Scalar::toString);

    nb::implicitly_convertible<double, Scalar>();
    nb::implicitly_convertible<int64_t, Scalar>();
    nb::implicitly_convertible<bool, Scalar>();

    // Bind Size class
    nb::class_<Size>(m, "Size")
        .def(nb::init<std::vector<int64_t>>())
        .def("__len__", &Size::size)
        .def("__getitem__", [](const Size& s, int64_t i) {
            if (i < 0) i += s.size();
            if (i < 0 || i >= (int64_t)s.size()) throw nb::index_error();
            return s[i];
        })
        .def("__iter__", [](const Size& s) {
            return nb::make_iterator(nb::type<Size>(), "iterator", s.begin(), s.end());
        }, nb::keep_alive<0, 1>())
        .def("__repr__", &Size::toString)
        .def("__str__", &Size::toString)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    // Bind Tensor class
    nb::class_<Tensor>(m, "Tensor")
        // Default constructor
        .def(nb::init<>())
        
        // Constructor from data (torch.tensor equivalent)
        .def("__init__", [](Tensor* self, nb::object data, std::optional<DType> dtype, std::optional<Device> device) {
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
                     throw std::runtime_error("Unsupported NumPy/DLPack dtype: code=" + std::to_string(code) + ", bits=" + std::to_string(bits));
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
                    new (self) Tensor();
                    return;
                 }
                 try {
                     nb::list l(data);
                     t = list_to_tensor(l.ptr());
                 } catch (...) {
                     throw std::runtime_error("Unsupported data type for Tensor creation");
                 }
            }
            
            if (dtype.has_value() && t.dtype() != *dtype) {
                Tensor new_t(static_cast<std::vector<int64_t>>(t.shape()), *dtype, device.value_or(Device(DeviceType::CPU)));
                convert_tensor_data(t, new_t);
                t = new_t;
            }
            
            new (self) Tensor(std::move(t));
        }, "data"_a, "dtype"_a = nb::none(), "device"_a = nb::none())
        
        // Properties
        .def_prop_ro("shape", &Tensor::shape)
        .def_prop_ro("dtype", &Tensor::dtype)
        .def_prop_ro("device", &Tensor::device)
        .def_prop_ro("ndim", &Tensor::dim)
        .def("dim", &Tensor::dim)
        .def("numel", &Tensor::numel)
        .def("itemsize", &Tensor::itemsize)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("requires_grad", &Tensor::requires_grad)
        
        // Methods
        .def("size", [](const Tensor& self) {
            return self.shape();
        })
        .def("size", [](const Tensor& self, int64_t dim) {
            return self.size(dim);
        })
        .def("view", &Tensor::view)
        .def("reshape", &Tensor::reshape)
        .def("expand", &Tensor::expand)
        .def("as_strided", &Tensor::as_strided, "size"_a, "stride"_a, "storage_offset"_a = nb::none())
        .def("copy_", &Tensor::copy_)
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
        
        .def("item", [](const Tensor& self) -> nb::object {
            switch (self.dtype()) {
                case DType::Float32: return nb::float_(self.item<float>());
                case DType::Float64: return nb::float_(self.item<double>());
                case DType::Int32: return nb::int_(self.item<int32_t>());
                case DType::Int64: return nb::int_(self.item<int64_t>());
                case DType::Bool: return nb::bool_(self.item<bool>());
                default: throw std::runtime_error("item() not implemented for this dtype");
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
                         throw std::runtime_error("Unsupported index type in tuple");
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
            throw std::runtime_error("Unsupported index type");
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
        
        // String repr
        .def("__repr__", &Tensor::toString)
        .def("__str__", &Tensor::toString)
        
        // Static factories
        .def_static("empty", &Tensor::empty, "size"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU))
        .def_static("zeros", &Tensor::zeros, "shape"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU))
        .def_static("ones", &Tensor::ones, "shape"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU))
        .def_static("full", &Tensor::full, "shape"_a, "fill_value"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU))
        .def_static("eye", &Tensor::eye, "n"_a, "m"_a = -1, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU))
        .def_static("arange", [](double start, double end, double step, DType dtype, const Device& device) {
             return Tensor::arange(Scalar(start), Scalar(end), Scalar(step), dtype, device);
        }, "start"_a, "end"_a, "step"_a = 1, "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU))
        .def_static("rand", &Tensor::rand, "size"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU))
        
        // *_like factories
        .def_static("empty_like", &Tensor::empty_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none())
        .def_static("zeros_like", &Tensor::zeros_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none())
        .def_static("ones_like", &Tensor::ones_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none())
        .def_static("full_like", &Tensor::full_like, "input"_a, "fill_value"_a, "dtype"_a = nb::none(), "device"_a = nb::none());

    // Module functions
    m.def("zeros", &Tensor::zeros, "shape"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU));
    m.def("ones", &Tensor::ones, "shape"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU));
    m.def("eye", &Tensor::eye, "n"_a, "m"_a = -1, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU));
    m.def("empty", &Tensor::empty, "size"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU));
    m.def("rand", &Tensor::rand, "size"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU));
    m.def("full", &Tensor::full, "shape"_a, "fill_value"_a, "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU));
    m.def("arange", [](double start, double end, double step, DType dtype, const Device& device) {
             return Tensor::arange(Scalar(start), Scalar(end), Scalar(step), dtype, device);
        }, "start"_a, "end"_a, "step"_a = 1, "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU));
        
    m.def("empty_like", &Tensor::empty_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none());
    m.def("zeros_like", &Tensor::zeros_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none());
    m.def("ones_like", &Tensor::ones_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none());
    m.def("full_like", &Tensor::full_like, "input"_a, "fill_value"_a, "dtype"_a = nb::none(), "device"_a = nb::none());
}
