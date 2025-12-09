#pragma once
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
#include "tensorplay/core/Exception.h"
#include "tensorplay/core/Generator.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace tensorplay;

void init_tensor(nb::module_& m);
void init_device(nb::module_& m);
void init_dtype(nb::module_& m);
void init_size(nb::module_& m);
void init_generator(nb::module_& m);
void init_autograd(nb::module_& m);
void init_ops(nb::module_& m);
void init_scalar(nb::module_& m);
