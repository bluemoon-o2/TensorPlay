#include "python_bindings.h"

void init_device(nb::module_& m) {
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

    nb::class_<Device>(m, "Device")
        .def(nb::init<DeviceType, int64_t>(), "type"_a, "index"_a = -1)
        .def(nb::init<const std::string&>(), "device"_a)
        .def(nb::init<const std::string&, int64_t>(), "type"_a, "index"_a)
        .def_prop_ro("type", [](const Device& d) {
            std::string s = d.toString();
            size_t colon = s.find(':');
            if (colon != std::string::npos) {
                return s.substr(0, colon);
            }
            return s;
        })
        .def_prop_ro("index", &Device::index)
        .def("is_cpu", &Device::is_cpu)
        .def("is_cuda", &Device::is_cuda)
        .def("__repr__", &Device::toString)
        .def("__str__", &Device::toString)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    nb::implicitly_convertible<std::string, Device>();
        
    // CUDA submodule
    nb::module_ cuda = m.def_submodule("cuda", "CUDA computation backend");
    cuda.def("is_available", []() {
#ifdef TENSORPLAY_USE_CUDA
        return true;
#else
        return false;
#endif
    });
}