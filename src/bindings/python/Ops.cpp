#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "utils.h"
#include <filesystem>

// Declaration of create_tensor (defined in Tensor.cpp)
Tensor create_tensor(nb::object data, std::optional<DType> dtype, std::optional<Device> device);

void init_ops(nb::module_& m) {
    // Module functions
    m.def("tensor", [](nb::object data, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
         Tensor t = create_tensor(data, dtype, device);
         if (requires_grad) {
             t.set_requires_grad(true);
         }
         return t;
    }, "data"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false);

    m.def("zeros", &Tensor::zeros, "shape"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);

    // Ops submodule
    nb::module_ ops = m.def_submodule("ops", "Operator registry");
    ops.def("load_library", [](const std::string& path) {
        namespace fs = std::filesystem;
        fs::path p(path);
        if (!fs::exists(p)) {
            TP_THROW(RuntimeError, "Library file not found: " + path);
        }
        
        nb::object importlib_util = nb::module_::import_("importlib.util");
        std::string name = p.stem().string();
        
        // Remove ABI tags (everything after first dot)
        size_t first_dot = name.find('.');
        if (first_dot != std::string::npos) {
            name = name.substr(0, first_dot);
        }
        
        // Remove "lib" prefix if present (common in Unix)
        if (name.size() > 3 && name.rfind("lib", 0) == 0) {
            name = name.substr(3);
        }

        nb::object spec = importlib_util.attr("spec_from_file_location")(name, path);
        if (spec.is_none()) {
            TP_THROW(RuntimeError, "Could not load library specification from: " + path);
        }
        
        nb::object module = importlib_util.attr("module_from_spec")(spec);
        spec.attr("loader").attr("exec_module")(module);
        
        // Register under tensorplay.ops
        nb::object tp = nb::module_::import_("tensorplay");
        if (nb::hasattr(tp, "ops")) {
            tp.attr("ops").attr(name.c_str()) = module;
        }
    }, "path"_a);

    m.def("ones", &Tensor::ones, "shape"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
    m.def("eye", &Tensor::eye, "n"_a, "m"_a = -1, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
    m.def("empty", &Tensor::empty, "size"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
    m.def("rand", &Tensor::rand, "size"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
    m.def("rand_like", [](const Tensor& input, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return Tensor::rand_like(input, dtype.value_or(DType::Undefined), device.value_or(Device(DeviceType::CPU)), requires_grad);
    }, "input"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false);

    m.def("randint", &Tensor::randint, "low"_a, "high"_a, "size"_a, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);

    m.def("randint_like", [](const Tensor& input, int64_t low, int64_t high, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return Tensor::randint_like(input, low, high, dtype.value_or(DType::Undefined), device.value_or(Device(DeviceType::CPU)), requires_grad);
    }, "input"_a, "low"_a, "high"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false);

    m.def("randn", &Tensor::randn, "size"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);

    m.def("randn_like", [](const Tensor& input, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
        return Tensor::randn_like(input, dtype.value_or(DType::Undefined), device.value_or(Device(DeviceType::CPU)), requires_grad);
    }, "input"_a, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false);
    m.def("randperm", &Tensor::randperm, "n"_a, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
    m.def("bernoulli", [](const Tensor& input) { return input.bernoulli(); }, "input"_a);
    m.def("normal", &Tensor::normal, "mean"_a, "std"_a);
    m.def("poisson", [](const Tensor& input) { return input.poisson(); }, "input"_a);
    m.def("full", &Tensor::full, "shape"_a, "fill_value"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
    m.def("arange", static_cast<Tensor(*)(Scalar, Scalar, Scalar, DType, Device, bool)>(&Tensor::arange), "start"_a, "end"_a, "step"_a = 1, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
    m.def("arange", static_cast<Tensor(*)(Scalar, DType, Device, bool)>(&Tensor::arange), "end"_a, nb::kw_only(), "dtype"_a = DType::Int64, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
        
    m.def("empty_like", &Tensor::empty_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none());
    m.def("zeros_like", &Tensor::zeros_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none());
    m.def("ones_like", &Tensor::ones_like, "input"_a, "dtype"_a = nb::none(), "device"_a = nb::none());
    m.def("full_like", &Tensor::full_like, "input"_a, "fill_value"_a, "dtype"_a = nb::none(), "device"_a = nb::none());

    // Config
    m.def("_show_config", &tensorplay::show_config);

    m.def("mm", [](const Tensor& self, const Tensor& other) { return self.mm(other); }, "input"_a, "mat2"_a);
    m.def("matmul", [](const Tensor& self, const Tensor& other) { return self.matmul(other); }, "input"_a, "other"_a);

    m.def("conv2d", &Tensor::conv2d, "input"_a, "weight"_a, "bias"_a = Tensor(), "stride"_a = std::vector<int64_t>{1, 1}, "padding"_a = std::vector<int64_t>{0, 0}, "dilation"_a = std::vector<int64_t>{1, 1}, "groups"_a = 1);

    m.def("cat", &Tensor::cat, "tensors"_a, "dim"_a = 0);
    m.def("stack", &Tensor::stack, "tensors"_a, "dim"_a = 0);
    m.def("transpose", [](const Tensor& t, int64_t dim0, int64_t dim1) { return t.transpose(dim0, dim1); }, "input"_a, "dim0"_a, "dim1"_a);
    m.def("permute", [](const Tensor& t, const std::vector<int64_t>& dims) { return t.permute(dims); }, "input"_a, "dims"_a);
    m.def("squeeze", [](const Tensor& t) { return t.squeeze(); }, "input"_a);
    m.def("squeeze", [](const Tensor& t, int64_t dim) { return t.squeeze(dim); }, "input"_a, "dim"_a);
    m.def("unsqueeze", [](const Tensor& t, int64_t dim) { return t.unsqueeze(dim); }, "input"_a, "dim"_a);
    m.def("split", [](const Tensor& t, int64_t split_size, int64_t dim) { return t.split(split_size, dim); }, "input"_a, "split_size"_a, "dim"_a = 0);
    m.def("split", [](const Tensor& t, const std::vector<int64_t>& split_sizes, int64_t dim) { return t.split(split_sizes, dim); }, "input"_a, "split_sizes"_a, "dim"_a = 0);
    m.def("chunk", [](const Tensor& t, int64_t chunks, int64_t dim) { return t.chunk(chunks, dim); }, "input"_a, "chunks"_a, "dim"_a = 0);
    m.def("t", [](const Tensor& t) { return t.t(); }, "input"_a);

    m.def("linspace", &Tensor::linspace, "start"_a, "end"_a, "steps"_a, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);
    m.def("logspace", &Tensor::logspace, "start"_a, "end"_a, "steps"_a, "base"_a = 10.0, nb::kw_only(), "dtype"_a = DType::Float32, "device"_a = Device(DeviceType::CPU), "requires_grad"_a = false);

    m.def("unbind", &Tensor::unbind, "input"_a, "dim"_a = 0);

    m.def("sum", [](const Tensor& input, std::optional<DType> dtype) {
        return input.sum(dtype.value_or(DType::Undefined));
    }, "input"_a, nb::kw_only(), "dtype"_a = nb::none());

    m.def("sum", [](const Tensor& input, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype) {
        return input.sum(dim, keepdim, dtype.value_or(DType::Undefined));
    }, "input"_a, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none());
    
    m.def("sum", [](const Tensor& input, int64_t dim, bool keepdim, std::optional<DType> dtype) {
        return input.sum({dim}, keepdim, dtype.value_or(DType::Undefined));
    }, "input"_a, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none());

    m.def("mean", [](const Tensor& input, std::optional<DType> dtype) {
        return input.mean(dtype.value_or(DType::Undefined));
    }, "input"_a, nb::kw_only(), "dtype"_a = nb::none());

    m.def("mean", [](const Tensor& input, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype) {
        return input.mean(dim, keepdim, dtype.value_or(DType::Undefined));
    }, "input"_a, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none());

    m.def("mean", [](const Tensor& input, int64_t dim, bool keepdim, std::optional<DType> dtype) {
        return input.mean({dim}, keepdim, dtype.value_or(DType::Undefined));
    }, "input"_a, "dim"_a, "keepdim"_a = false, nb::kw_only(), "dtype"_a = nb::none());
}