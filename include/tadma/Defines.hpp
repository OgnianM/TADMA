#pragma once
#ifdef __CUDACC__
#define __multi__ __device__ __host__
#else
#define __multi__
#endif

// #define __multi__ __host__ __device__

#define check_cuda(expr) { \
    cudaError_t status = expr; \
    if (status != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(status)); \
    } \
}


#define check_cublas(expr) { \
    cublasStatus_t status = expr; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("CUBLAS error " + std::to_string(status)); \
    } \
}

constexpr auto ForwardFunction = [] __multi__ (const auto& x) { return x; };
