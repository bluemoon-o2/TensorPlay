cmake_minimum_required(VERSION 3.18)

project(p10 LANGUAGES CXX)

set(P10_SOURCES
    src/Allocator.cpp
    src/Device.cpp
    src/Dispatcher.cpp
    src/Exception.cpp
    src/Generator.cpp
    src/Scalar.cpp
    src/SizesAndStrides.cpp
    src/Stacktrace.cpp
    src/TensorImpl.cpp
    # src/AutogradMetaInterface.cpp (Moved to TPX)
    src/Tensor.cpp
    
    # Kernels
    src/backend/cpu/ArithmeticKernels.cpp
    src/backend/cpu/ComparisonKernels.cpp
    src/backend/cpu/ConvKernels.cpp
    src/backend/cpu/CopyKernels.cpp
    src/backend/cpu/FactoryKernels.cpp
    src/backend/cpu/LinearAlgebraKernels.cpp
    src/backend/cpu/PointwiseKernels.cpp
    src/backend/cpu/RandomKernels.cpp
    src/backend/cpu/ReductionKernels.cpp
    src/backend/cpu/ViewKernels.cpp
    
    # Generated
    ${GEN_CPP}
)

add_library(p10 SHARED ${P10_SOURCES})
add_dependencies(p10 generate_code)

target_include_directories(p10 PUBLIC  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GEN_BASE_DIR}
)

target_compile_definitions(p10 PRIVATE P10_EXPORTS)
target_compile_definitions(p10 PRIVATE ${BLAS_COMPILE_DEFS})

if(WIN32)
    target_link_libraries(p10 PRIVATE Dbghelp)
endif()

if(COMPUTE_LIBS)
    target_link_libraries(p10 PRIVATE ${COMPUTE_LIBS})
endif()

# Installation logic for p10
set(TP_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/tensorplay")
set(TP_LIB_DIR "${TP_PACKAGE_DIR}/lib")

set_target_properties(p10 PROPERTIES
    OUTPUT_NAME "p10"
    RUNTIME_OUTPUT_DIRECTORY "${TP_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TP_LIB_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${TP_LIB_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${TP_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${TP_LIB_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${TP_LIB_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${TP_LIB_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${TP_LIB_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${TP_LIB_DIR}"
)

install(TARGETS p10
    RUNTIME DESTINATION tensorplay/lib
    LIBRARY DESTINATION tensorplay/lib
    ARCHIVE DESTINATION tensorplay/lib
)
