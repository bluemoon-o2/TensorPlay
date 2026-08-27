
#pragma once
#include <string>
#include <map>
#include "Parallel.h"

namespace tensorplay {
    inline std::string show_config() {
        std::string s = "TensorPlay built with:\n";
        s += "   - C++ Version: 202002\n";
        s += "   - GNU 11.4.0\n";
        s += "   - Intel(R) oneAPI Math Kernel Library Version 2026.1.0 for Intel(R) 64 architecture applications\n";
        s += "   - Intel(R) MKL-DNN v\n";
        s += "   - OpenMP 4.5\n";
        s += "   - CUDA disabled\n";
        s += "   - cuDNN disabled\n";
        s += "   - LAPACK is enabled (usually provided by MKL)\n";
        s += "   - CPU capability usage: Native\n";
        s += "   - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS=, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TENSORPLAY_VERSION=1.0.0rc0, USE_OPENMP=TRUE, USE_MKL=TRUE, USE_MKLDNN=TRUE, USE_CUDA=OFF, USE_CUDNN=OFF\n";
        return s;
    }

    inline std::string _cxx_flags() {
        return "";
    }

    inline std::string _parallel_info() {
        return tensorplay::parallel::get_parallel_info();
    }

    inline std::map<std::string, std::string> get_build_info() {
        std::map<std::string, std::string> info;
        info["CXX_VERSION"] = "202002";
        info["COMPILER_INFO"] = "GNU 11.4.0";
        info["MKL_INFO"] = "Intel(R) oneAPI Math Kernel Library Version 2026.1.0 for Intel(R) 64 architecture applications";
        info["ONEDNN_INFO"] = "Intel(R) MKL-DNN v";
        info["OPENMP_INFO"] = "OpenMP 4.5";
        info["CUDA_INFO"] = "CUDA disabled";
        info["CUDNN_INFO"] = "cuDNN disabled";
        info["LAPACK_INFO"] = "LAPACK is enabled (usually provided by MKL)";
        info["CPU_CAPABILITY"] = "Native";
        info["BUILD_SETTINGS"] = "BLAS_INFO=mkl, BUILD_TYPE=Release, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS=, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TENSORPLAY_VERSION=1.0.0rc0, USE_OPENMP=TRUE, USE_MKL=TRUE, USE_MKLDNN=TRUE, USE_CUDA=OFF, USE_CUDNN=OFF";
        return info;
    }
}
