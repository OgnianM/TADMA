#pragma once
#include "tadma/Allocator.hpp"
#include "tadma/Tensor.hpp"

namespace tadma {
__global__
void EltwiseVariadicNDKernel(auto f, AnyCudaTensor auto result, AnyCudaTensor auto x, AnyCudaTensor auto... ys) {
    auto indices = result.memoryIndexToTensorIndex(blockIdx.x * blockDim.x + threadIdx.x);
    result(indices) = f(indices, x, ys...);
}

/**
 * @brief Create a tensor of OutputShape and fill it such that result(i, j, k, ...) = f(std::array {i, j, k, ...}, x, ys...)
 * where x, ys... are the input tensors and i, j, k, ... are the indices of the output tensor
 * @tparam OutputShape
 * @param f
 * @param x
 * @param ys
 * @return The newly created tensor
 */
template<AnySequence OutputShape>
auto EltwiseVariadicNDNode(auto&& f, AnyTensor auto&& x, AnyTensor auto&&... ys) {
    using Allocator = CommonAllocator<typename TYPE(x)::Allocator, typename TYPE(ys)::Allocator...>;

    if constexpr (Allocator::device == kCUDA) {
        Tensor<typename TYPE(x)::ValueType, Allocator, OutputShape> result;
        EltwiseVariadicNDKernel<<<(result.Size + 255) / 256, 256, 0, stream>>>(f, result, x, ys...);
        return result;
    } else if constexpr (Allocator::device == kCPU) {
        Tensor<typename TYPE(x)::ValueType, Allocator, OutputShape> result;
        #pragma omp parallel for
        for (size_t i = 0; i < result.Size; i++) {
            auto indices = result.memoryIndexToTensorIndex(i);
            result(indices) = f(indices, x, ys...);
        }
        return result;
    } else assert(false);
}
}
