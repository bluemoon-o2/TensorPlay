cmake_minimum_required(VERSION 3.18)
project(TensorPlay LANGUAGES CXX)
set(CMAKE_CUDA_COMPILER OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_compile_options(/std:c++20 /EHsc)
    if(MSVC)
        add_compile_options(/wd4251 /wd4275)
    endif()
else()
    add_compile_options(-std=c++20 -O3 -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-fPIC)
    endif()
endif()

include_directories(include)

# Nanobind requires "Python" (not Python3) package to be found first
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Codegen
set(CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen.py")
set(YAML_FILE "${CMAKE_CURRENT_SOURCE_DIR}/native_functions.yaml")
set(GEN_BASE_DIR "${CMAKE_BINARY_DIR}/generated")
set(GEN_INCLUDE_DIR "${GEN_BASE_DIR}/tensorplay/core/generated")
set(GEN_HEADER "${GEN_INCLUDE_DIR}/TensorGenerated.h")
set(GEN_CPP "${GEN_INCLUDE_DIR}/TensorGenerated.cpp")

file(MAKE_DIRECTORY ${GEN_INCLUDE_DIR})

add_custom_command(
    OUTPUT ${GEN_HEADER} ${GEN_CPP}
    COMMAND ${Python_EXECUTABLE} ${CODEGEN_SCRIPT} --yaml ${YAML_FILE} --out_dir ${GEN_INCLUDE_DIR}
    DEPENDS ${CODEGEN_SCRIPT} ${YAML_FILE}
    COMMENT "Generating Tensor methods from native_functions.yaml"
)

include_directories(${GEN_BASE_DIR})

# Nanobind via pip installation
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Core Source files (excluding Python bindings and converters)
file(GLOB_RECURSE CORE_SOURCES
    "src/core/*.cpp"
    "src/backend/cpu/*.cpp"
    "include/tensorplay/core/*.h"
)
list(APPEND CORE_SOURCES ${GEN_CPP})

# Python Extension Sources
set(EXTENSION_SOURCES
    ${CORE_SOURCES}
    src/bindings/python/utils.h
    src/bindings/python/init.cpp
)

# Create Python extension using nanobind
nanobind_add_module(tensorplay ${EXTENSION_SOURCES})

# Generate .pyi stub
if (NOT CMAKE_CROSSCOMPILING)
    add_custom_command(TARGET tensorplay POST_BUILD
        COMMAND "${Python_EXECUTABLE}" -m nanobind.stubgen -m tensorplay -o "${CMAKE_CURRENT_SOURCE_DIR}/tensorplay.pyi"
        WORKING_DIRECTORY $<TARGET_FILE_DIR:tensorplay>
        COMMENT "Generating .pyi stub for tensorplay"
    )
endif()

# Installation
install(TARGETS tensorplay
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

# Add custom target for building python extension
add_custom_target(build_python
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target tensorplay
    COMMENT "Building Python extension"
)

# Debug Executable (Pure C++)
add_executable(debug_tensor debug/debug_tensor.cpp ${CORE_SOURCES})
target_include_directories(debug_tensor PRIVATE include)

# Testing support
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
        add_subdirectory(test)
    endif()
endif()
