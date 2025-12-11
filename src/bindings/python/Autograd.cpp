#include "python_bindings.h"
#include "Node.h"

void init_autograd(nb::module_& m) {
    nb::class_<tensorplay::tpx::Node>(m, "Node");

    nb::module_ autograd = m.def_submodule("autograd", "Autograd mechanism");
    
    // Profiler submodule
    nb::module_ profiler = m.def_submodule("profiler", "Profiler");
    
    // Parallel submodule
    nb::module_ parallel = m.def_submodule("parallel", "Parallel computing");
}