#include "python_bindings.h"

#include "Device.h"
#include "DType.h"
#include "Exception.h"
#include "autocast_mode.h"

#include <string>

namespace tensorplay {
namespace {

DeviceType device_type_from_string(const std::string& device_type) {
    if (device_type == "cpu") return DeviceType::CPU;
    if (device_type == "cuda") return DeviceType::CUDA;
    TP_THROW(ValueError, "Expected one of cpu or cuda device types, but got " + device_type);
}

} // anonymous namespace
} // namespace tensorplay

void init_autocast(py::module_& m) {
    using namespace tensorplay;
    namespace autocast = tensorplay::autocast;

    m.def("_is_autocast_available", [](const std::string& device_type) {
        return autocast::is_autocast_available(device_type_from_string(device_type));
    }, "device_type"_a);

    m.def("is_autocast_enabled", [](const std::string& device_type) {
        return autocast::is_autocast_enabled(device_type_from_string(device_type));
    }, "device_type"_a = std::string("cuda"));

    m.def("get_autocast_dtype", [](const std::string& device_type) {
        return autocast::get_autocast_dtype(device_type_from_string(device_type));
    }, "device_type"_a = std::string("cuda"));

    m.def("set_autocast_enabled", [](const std::string& device_type, bool enabled) {
        autocast::set_autocast_enabled(device_type_from_string(device_type), enabled);
    }, "device_type"_a, "enabled"_a);

    m.def("set_autocast_dtype", [](const std::string& device_type, DType dtype) {
        autocast::set_autocast_dtype(device_type_from_string(device_type), dtype);
    }, "device_type"_a, "dtype"_a);

    // deprecated CUDA/CPU-specific autocast APIs (kept for BC, mirroring the
    // deprecated at::autocast inline helpers)
    m.def("get_autocast_gpu_dtype", []() {
        return autocast::get_autocast_dtype(DeviceType::CUDA);
    });
    m.def("get_autocast_cpu_dtype", []() {
        return autocast::get_autocast_dtype(DeviceType::CPU);
    });

    m.def("autocast_increment_nesting", []() {
        return autocast::increment_nesting();
    });
    m.def("autocast_decrement_nesting", []() {
        return autocast::decrement_nesting();
    });

    m.def("clear_autocast_cache", []() {
        autocast::clear_cache();
    });

    m.def("is_autocast_cache_enabled", []() {
        return autocast::is_autocast_cache_enabled();
    });

    m.def("set_autocast_cache_enabled", [](bool enabled) {
        autocast::set_autocast_cache_enabled(enabled);
    }, "enabled"_a);
}
