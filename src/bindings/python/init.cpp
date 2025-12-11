#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include <cstdlib>

// Extern declarations (if not in header)
void init_scalar(nb::module_& m);

NB_MODULE(_C, m) {
    m.doc() = "The C extension module of tensorplay";

    // Exception translation
    nb::register_exception_translator([](const std::exception_ptr &p, void * /* unused */) {
        auto set_error = [](PyObject* type, const tensorplay::Exception& e) {
            std::string msg = e.msg();
            const char* env_val = std::getenv("TENSORPLAY_SHOW_CPP_STACKTRACES");
            if (env_val && std::string(env_val) == "1" && !e.stacktrace().empty()) {
                msg += "\n\n" + e.stacktrace();
            }
            PyErr_SetString(type, msg.c_str());
        };

        try {
            std::rethrow_exception(p);
        } catch (const tensorplay::IndexError &e) {
            set_error(PyExc_IndexError, e);
        } catch (const tensorplay::ValueError &e) {
            set_error(PyExc_ValueError, e);
        } catch (const tensorplay::TypeError &e) {
            set_error(PyExc_TypeError, e);
        } catch (const tensorplay::NotImplementedError &e) {
            set_error(PyExc_NotImplementedError, e);
        } catch (const tensorplay::Exception &e) {
            set_error(PyExc_RuntimeError, e);
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // Warning handler
    tensorplay::setWarningHandler([](const tensorplay::SourceLocation& source, const std::string& msg) {
        nb::gil_scoped_acquire gil;
        PyErr_WarnEx(PyExc_UserWarning, msg.c_str(), 1);
    });

    init_dtype(m);
    init_device(m);
    init_scalar(m); // Initialize scalar after DType/Device as it might be used? Actually scalar is independent.
    init_size(m);
    init_generator(m);
    init_tensor(m);
    init_autograd(m);
    init_ops(m);

    // CUDA availability
    m.def("is_cuda_available", []() {
#ifdef TENSORPLAY_USE_CUDA
        return true;
#else
        return false;
#endif
    });

    // Numpy availability
    m.def("is_numpy_available", []() {
        try {
            nb::module_::import_("numpy");
            return true;
        } catch (...) {
            return false;
        }
    });

    // Config
    m.def("_show_config", &tensorplay::show_config);
}