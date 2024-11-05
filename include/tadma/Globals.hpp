#pragma once
#include "Defines.hpp"
#include <cuda_runtime.h>
#include <cublasLt.h>

namespace tadma {

thread_local cudaStream_t stream = nullptr;
thread_local cublasLtHandle_t cublasLtHandle = nullptr;

void init() {
    check_cuda(cudaStreamCreate(&stream));
    check_cublas(cublasLtCreate(&cublasLtHandle));
}

void deinit() {
    cudaStreamDestroy(stream);
    cublasLtDestroy(cublasLtHandle);
}

struct InitGuard {
    InitGuard() { init(); }
    ~InitGuard() { deinit(); }
};


struct CUDAStreamGuard {
    cudaStream_t oldStream;
    CUDAStreamGuard(cudaStream_t newStream) {
        oldStream = stream;
        stream = newStream;
    }

    ~CUDAStreamGuard() {
        stream = oldStream;
    }
};


struct CUDAState {
    cudaStream_t stream;
    cublasLtHandle_t cublasLtHandle;
};

};
