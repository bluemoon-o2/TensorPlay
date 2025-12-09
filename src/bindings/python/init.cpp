#include "python_bindings.h"
#include "tensorplay/ops/Config.h"

// Extern declarations (if not in header)
void init_scalar(nb::module_& m);

NB_MODULE(_C, m) {
    m.doc() = "The C extension module of tensorplay";

    // Exception translation
    nb::register_exception_translator([](const std::exception_ptr &p, void * /* unused */) {
        auto set_error = [](PyObject* type, const tensorplay::Exception& e) {
            std::string msg = e.msg();
            if (!e.stacktrace().empty()) {
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
    m.def("cuda_is_available", []() {
        return false;
    });
}