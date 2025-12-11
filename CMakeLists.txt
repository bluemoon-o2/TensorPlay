cmake_minimum_required(VERSION 3.18)
project(TensorPlay LANGUAGES CXX)
set(CMAKE_CUDA_COMPILER OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_compile_options(/std:c++20 /EHsc)
    if(MSVC)
        add_compile_options(/wd4251 /wd4275)
        # Enable multi-processor compilation
        add_compile_options(/MP)
        # Disable minimal rebuild for better parallelism
        add_compile_options(/Gm-)
        # Enable optimization for Release builds
        add_compile_options($<$<CONFIG:Release>:/O2>)
    endif()
else()
    add_compile_options(-std=c++20 -O3 -Wall -Wextra -Wpedantic)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-fPIC)
    endif()
endif()

include_directories(include)

# Nanobind requires "Python" package to be found first
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Derive Python Root if not set
if(NOT DEFINED Python_ROOT_DIR)
    get_filename_component(PYTHON_BIN_DIR "${Python_EXECUTABLE}" DIRECTORY)
    # Check if executable is in Scripts or bin or root
    if(EXISTS "${PYTHON_BIN_DIR}/python.exe" OR EXISTS "${PYTHON_BIN_DIR}/python")
        set(Python_ROOT_DIR "${PYTHON_BIN_DIR}")
    endif()
endif()
message(STATUS "Python Root: ${Python_ROOT_DIR}")

# Code generation variables
set(GEN_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/generated")
set(GEN_INCLUDE_DIR "${GEN_BASE_DIR}/tensorplay/ops")
set(GEN_HEADER "${GEN_INCLUDE_DIR}/TensorGenerated.h")
set(GEN_CPP "${GEN_INCLUDE_DIR}/TensorGenerated.cpp")
set(GEN_BINDINGS_HEADER "${GEN_INCLUDE_DIR}/TensorBindingsGenerated.h")
set(GEN_AUTOGRAD_NODES_H "${GEN_INCLUDE_DIR}/AutogradNodesGenerated.h")
set(GEN_TPX_OPS_H "${GEN_INCLUDE_DIR}/TPXOpsGenerated.h")
set(GEN_TPX_OPS_CPP "${GEN_INCLUDE_DIR}/TPXOpsGenerated.cpp")

# Python Interface Stub Generation
set(PYI_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/templates/_C.pyi.in")
set(TP_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tensorplay")
set(PYI_OUTPUT "${TP_PACKAGE_DIR}/_C.pyi")

# Code generation
add_custom_command(
            OUTPUT ${GEN_HEADER} ${GEN_CPP} ${GEN_BINDINGS_HEADER} ${PYI_OUTPUT} 
                   ${GEN_AUTOGRAD_NODES_H} ${GEN_TPX_OPS_H} ${GEN_TPX_OPS_CPP}
            COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen.py 
                    --yaml ${CMAKE_CURRENT_SOURCE_DIR}/config/native_functions.yaml 
                    --derivatives ${CMAKE_CURRENT_SOURCE_DIR}/config/derivatives.yaml
                    --out_dir ${GEN_INCLUDE_DIR}
                    --pyi_template ${PYI_TEMPLATE}
                    --pyi_out ${PYI_OUTPUT}
                    --functional_out ${CMAKE_CURRENT_SOURCE_DIR}/tensorplay/functional.py
            DEPENDS 
                ${CMAKE_CURRENT_SOURCE_DIR}/config/native_functions.yaml
                ${CMAKE_CURRENT_SOURCE_DIR}/config/derivatives.yaml
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/gen.py
                ${PYI_TEMPLATE}
            COMMENT "Generating Tensor methods, bindings and pyi stubs from native_functions.yaml"
        )

# Add a custom target to ensure code generation runs before library building
add_custom_target(generate_code DEPENDS ${GEN_HEADER} ${GEN_CPP} ${GEN_TPX_OPS_CPP})

include_directories(${GEN_BASE_DIR})

# Nanobind via pip installation
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# --- Compute Backend Selection ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    add_compile_options(${OpenMP_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

option(USE_BLAS "Enable BLAS linear algebra acceleration" ON)
if(NOT USE_BLAS)
  message(STATUS "Disabled BLAS acceleration (USE_BLAS=OFF)")
  return()
endif()

set(BLAS_PROVIDER "MKL" CACHE STRING "BLAS provider (MKL/OpenBLAS)")
set_property(CACHE BLAS_PROVIDER PROPERTY STRINGS "MKL" "OpenBLAS")

set(COMPUTE_LIBS "")
set(BLAS_COMPILE_DEFS "")
set(BLAS_COMPILE_OPTIONS "")

if(BLAS_PROVIDER STREQUAL "MKL")
  # Common Windows Paths for OneAPI MKL
  list(APPEND CMAKE_PREFIX_PATH 
      "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
      "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl"
  )
  
  # Add Python environment paths (where pip installs mkl-include)
  if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH "${Python_ROOT_DIR}/Library")
    include_directories(SYSTEM "${Python_ROOT_DIR}/Library/include")
  else()
    list(APPEND CMAKE_PREFIX_PATH "${Python_ROOT_DIR}")
    include_directories(SYSTEM "${Python_ROOT_DIR}/include")
  endif()

  # 1. Try MKL Config mode (Best for OneAPI)
  find_package(MKL CONFIG QUIET)
  
  if(NOT MKL_FOUND)
      # Check if pip packages are installed to avoid re-running pip install every time
      execute_process(
          COMMAND "${Python_EXECUTABLE}" -m pip show mkl-include mkl mkl-static
          RESULT_VARIABLE MKL_PIP_CHECK
          OUTPUT_QUIET ERROR_QUIET
      )

      if(NOT MKL_PIP_CHECK EQUAL 0)
          # Attempt to auto-install MKL via pip if not found
          message(STATUS "⚠️ MKL not found. Attempting to install 'mkl-include', 'mkl', and 'mkl-static' via pip...")
          execute_process(
              COMMAND "${Python_EXECUTABLE}" -m pip install mkl-include mkl mkl-static
              RESULT_VARIABLE MKL_INSTALL_RES
          )
      else()
          message(STATUS "MKL pip packages detected (skipping install).")
          set(MKL_INSTALL_RES 0)
      endif()

      if(MKL_INSTALL_RES EQUAL 0)
          message(STATUS "✅ Successfully installed MKL packages via pip.")
          # Re-add paths just in case
          if(WIN32)
            list(APPEND CMAKE_PREFIX_PATH "${Python_ROOT_DIR}/Library")
            include_directories(SYSTEM "${Python_ROOT_DIR}/Library/include")
          else()
            list(APPEND CMAKE_PREFIX_PATH "${Python_ROOT_DIR}")
            include_directories(SYSTEM "${Python_ROOT_DIR}/include")
          endif()
          
          # Retry finding
          find_package(MKL CONFIG QUIET)
      else()
          message(WARNING "Failed to install MKL via pip.")
      endif()
  endif()

  if(MKL_FOUND)
    message(STATUS "Found MKL (Config): ${MKL_DIR}")
    list(APPEND COMPUTE_LIBS MKL::MKL)
    set(BLAS_FOUND TRUE)
  else()
    # 2. Fallback to FindBLAS with expanded paths
    # Set preference for Intel MKL 64-bit LP interface
    set(BLA_VENDOR "Intel10_64lp")
    
    # Check manual paths for pip installed MKL libs
    if(WIN32)
        set(MKL_PIP_LIB_PATH "${Python_ROOT_DIR}/Library/lib")
    else()
        set(MKL_PIP_LIB_PATH "${Python_ROOT_DIR}/lib")
    endif()
    
    # Try finding BLAS (Dynamic first)
    find_package(BLAS QUIET HINTS ${MKL_PIP_LIB_PATH})
    
    # If not found, try Static (common for mkl-static pip package)
    if(NOT BLAS_FOUND)
        set(BLA_STATIC ON)
        find_package(BLAS QUIET HINTS ${MKL_PIP_LIB_PATH})
    endif()
    
    # Manual Fallback for pip-installed MKL (Static libs)
    if(NOT BLAS_FOUND)
        find_library(MKL_INTEL_LIB NAMES mkl_intel_lp64 HINTS ${MKL_PIP_LIB_PATH})
        find_library(MKL_CORE_LIB NAMES mkl_core HINTS ${MKL_PIP_LIB_PATH})
        find_library(MKL_THREAD_LIB NAMES mkl_intel_thread HINTS ${MKL_PIP_LIB_PATH})
        find_library(MKL_IOMP5_LIB NAMES libiomp5md HINTS ${MKL_PIP_LIB_PATH})
        
        if(MKL_INTEL_LIB AND MKL_CORE_LIB AND MKL_THREAD_LIB AND MKL_IOMP5_LIB)
            message(STATUS "Found MKL (Manual Pip Check): ${MKL_INTEL_LIB}")
            set(BLAS_LIBRARIES ${MKL_INTEL_LIB} ${MKL_THREAD_LIB} ${MKL_CORE_LIB} ${MKL_IOMP5_LIB})
            set(BLAS_FOUND TRUE)
            
            # Find headers
            find_path(MKL_INCLUDE_DIR mkl.h 
                HINTS "${Python_ROOT_DIR}/Library/include"
                "${Python_ROOT_DIR}/include"
            )
            if(MKL_INCLUDE_DIR)
              include_directories(SYSTEM ${MKL_INCLUDE_DIR})
            endif()
        endif()
    endif()
    
    if(BLAS_FOUND)
      string(FIND "${BLAS_LIBRARIES}" "mkl" MKL_LIB_FOUND)
      if(MKL_LIB_FOUND GREATER -1)
        message(STATUS "Found MKL via BLAS: ${BLAS_LIBRARIES}")
        list(APPEND COMPUTE_LIBS ${BLAS_LIBRARIES})
        
        # Find headers
        find_path(MKL_INCLUDE_DIR mkl.h 
            HINTS $ENV{MKL_ROOT}/include 
            "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include"
            "${Python_ROOT_DIR}/Library/include"
            "${Python_ROOT_DIR}/include"
            PATH_SUFFIXES include
        )
        if(MKL_INCLUDE_DIR)
          include_directories(SYSTEM ${MKL_INCLUDE_DIR})
        endif()
      else()
        set(BLAS_FOUND FALSE)
      endif()
    endif()
  endif()

  if(BLAS_FOUND)
      list(APPEND BLAS_COMPILE_DEFS "USE_MKL")
      message(STATUS "✅ Successfully configured Intel MKL")
  else()
      message(WARNING "Could not find MKL. Please install OneAPI MKL or check pip installation.")
  endif()

elseif(BLAS_PROVIDER STREQUAL "OpenBLAS")
  # Try Config first
  find_package(OpenBLAS CONFIG QUIET)
  if(OpenBLAS_FOUND)
     list(APPEND COMPUTE_LIBS OpenBLAS::OpenBLAS)
     set(BLAS_FOUND TRUE)
  else()
     # Fallback to PkgConfig or manual search
     find_package(PkgConfig QUIET)
     if(PKG_CONFIG_FOUND)
        pkg_check_modules(OpenBLAS_PC openblas QUIET)
        if(OpenBLAS_PC_FOUND)
            list(APPEND COMPUTE_LIBS ${OpenBLAS_PC_LIBRARIES})
            include_directories(SYSTEM ${OpenBLAS_PC_INCLUDE_DIRS})
            set(BLAS_FOUND TRUE)
        endif()
     endif()
  endif()

  if(BLAS_FOUND)
      list(APPEND BLAS_COMPILE_DEFS "USE_OPENBLAS")
      list(APPEND BLAS_COMPILE_DEFS "USE_BLAS")
      message(STATUS "✅ Successfully configured OpenBLAS")
  endif()
endif()

add_compile_definitions(${BLAS_COMPILE_DEFS})

if(MSVC)
  list(APPEND BLAS_COMPILE_OPTIONS "/O2" "/arch:AVX2")
else()
  list(APPEND BLAS_COMPILE_OPTIONS "-march=native" "-O3")
endif()

if(NOT BLAS_FOUND)
    if(WIN32)
        message(STATUS "⚠️ Compute library not found. Attempting to auto-download OpenBLAS (Windows Pre-built)...")
        include(FetchContent)
        
        # Use OpenBLAS 0.3.26 which is stable and has pre-built binaries
        FetchContent_Declare(
            openblas_prebuilt
            URL https://github.com/xianyi/OpenBLAS/releases/download/v0.3.26/OpenBLAS-0.3.26-x64.zip
        )
        
        FetchContent_GetProperties(openblas_prebuilt)
        if(NOT openblas_prebuilt_POPULATED)
            message(STATUS "Downloading OpenBLAS 0.3.26...")
            FetchContent_Populate(openblas_prebuilt)
        endif()
        
        set(OpenBLAS_ROOT ${openblas_prebuilt_SOURCE_DIR})
        set(OpenBLAS_INCLUDE_DIR "${OpenBLAS_ROOT}/include")
        set(OpenBLAS_LIBRARY "${OpenBLAS_ROOT}/lib/libopenblas.lib")
        
        if(EXISTS "${OpenBLAS_INCLUDE_DIR}" AND EXISTS "${OpenBLAS_LIBRARY}")
             message(STATUS "✅ Successfully downloaded OpenBLAS to ${OpenBLAS_ROOT}")
             
             include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
             list(APPEND COMPUTE_LIBS ${OpenBLAS_LIBRARY})
             list(APPEND BLAS_COMPILE_DEFS "USE_OPENBLAS")
             list(APPEND BLAS_COMPILE_DEFS "USE_BLAS")
             set(BLAS_FOUND TRUE)
             
             # Copy DLLs to binary directories for runtime
             file(GLOB OPENBLAS_DLLS "${OpenBLAS_ROOT}/bin/*.dll")
             foreach(DLL ${OPENBLAS_DLLS})
                 # Copy to build output dirs
                 file(COPY ${DLL} DESTINATION ${CMAKE_BINARY_DIR}/Release)
                 file(COPY ${DLL} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
                 # Copy to Python package lib dir (for import tensorplay)
                file(COPY ${DLL} DESTINATION ${CMAKE_SOURCE_DIR}/tensorplay/lib)
                 message(STATUS "  - Copied DLL: ${DLL}")
             endforeach()
        else()
             message(WARNING "Downloaded OpenBLAS but verification failed (include/lib not found).")
        endif()
    else()
        message(WARNING "No accelerated compute library (MKL/BLAS) found. Auto-download is only supported on Windows. Using naive implementation.")
    endif()
endif()

if(NOT BLAS_FOUND)
    message(WARNING "No accelerated compute library (MKL/BLAS) found. Using naive implementation.")
endif()

# --- Config Generation ---
set(TP_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT TP_BUILD_TYPE)
    set(TP_BUILD_TYPE "Release")
endif()

if(BLAS_FOUND)
    set(BLAS_FOUND_STR "YES")
else()
    set(BLAS_FOUND_STR "NO")
endif()

set(CONFIG_CONTENT "
#pragma once
#include <string>

namespace tensorplay {
    inline std::string show_config() {
        std::string s = \"TensorPlay built with:\\n\";
        s += \"  - C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\\n\";
        s += \"  - Build type: ${TP_BUILD_TYPE}\\n\";
        s += \"  - BLAS Info: ${BLAS_PROVIDER}\\n\";
        s += \"  - BLAS Found: ${BLAS_FOUND_STR}\\n\";
        s += \"  - OpenMP: ${OpenMP_CXX_FOUND}\\n\";
        return s;
    }
}
")
file(WRITE "${GEN_BASE_DIR}/tensorplay/ops/Config.h" "${CONFIG_CONTENT}")

# Add p10
add_subdirectory(p10)

# Add tpx
add_subdirectory(tpx)

# Add stax (Optional)
add_subdirectory(stax)

# Add nn (Optional)
add_subdirectory(nn)

# Python Bindings
nanobind_add_module(_C 
    src/bindings/python/init.cpp
    src/bindings/python/DType.cpp
    src/bindings/python/Scalar.cpp
    src/bindings/python/Device.cpp
    src/bindings/python/Size.cpp
    src/bindings/python/Generator.cpp
    src/bindings/python/Autograd.cpp
    src/bindings/python/Ops.cpp
    src/bindings/python/Tensor.cpp
)
target_link_libraries(_C PRIVATE p10 tpx nn stax)
target_include_directories(_C PRIVATE ${GEN_BASE_DIR} include)

set(TP_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tensorplay")
set(TP_LIB_DIR "${TP_PACKAGE_DIR}/lib")
file(MAKE_DIRECTORY "${TP_LIB_DIR}")

set_target_properties(_C PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${TP_PACKAGE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TP_PACKAGE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TP_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TP_PACKAGE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${TP_PACKAGE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${TP_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TP_PACKAGE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${TP_PACKAGE_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${TP_PACKAGE_DIR}"
)

# Installation
install(TARGETS _C
    LIBRARY DESTINATION tensorplay
    RUNTIME DESTINATION tensorplay
)

# Install the generated stub file
install(FILES "${PYI_OUTPUT}" DESTINATION tensorplay)


# Add custom target for building python extension
add_custom_target(build_python
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target _C
    COMMENT "Building Python extension"
)

# Debug Executable (Pure C++)
add_executable(debug_tensor debug/debug_tensor.cpp)
target_link_libraries(debug_tensor PRIVATE p10)

# Testing support
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
        add_subdirectory(test)
    endif()
endif()
