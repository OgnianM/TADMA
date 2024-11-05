#pragma once
#include "tadma/Tensor.hpp"

/**
 * @brief Reduction kernel for a tensor along a given dimension
 * @note CUDA only
 */
namespace tadma {

template<int Threads, AnyTensor T, AnyTensor RT, typename Reduce> requires(Commutative<Reduce>)
__global__ void ReduceKernel(T t, RT result, Reduce reduce, auto preprocess, auto postprocess) {
    constexpr auto Size = T::Dim(0);

    using ValueType = typename T::ValueType;
    using ReduceType = decltype(reduce(preprocess(ValueType()), preprocess(ValueType())));

    static_assert(std::is_same_v<typename RT::ValueType, decltype(postprocess(ReduceType()))>,
        "Result tensor must have the same value type as the reduction function");

    auto tid = threadIdx.x;
    auto group = blockIdx.x;

    auto index = [group](uint32_t i, auto& t) -> auto {
        auto tmp = t(i);
        return tmp(tmp.memoryIndexToTensorIndex(group));
    };
    __shared__ ReduceType shared[Threads];

    auto idx = tid;
    if (idx < Size) {
        ReduceType value = preprocess(index(idx, t));

#pragma unroll
        for (idx += Threads; idx < Size; idx += Threads) {
            value = reduce(value, preprocess(index(idx, t)));
        }
        shared[tid] = value;
    }
    __syncthreads();

    // Tree reduction
    for (int i = Threads >> 1; i > 0; i >>= 1) {
        if constexpr (Size >= Threads) {
            if (tid < i) {
                shared[tid] = reduce(shared[tid], shared[tid + i]);
            }
        } else {
            if (tid < i && (tid + i) < Size) {
                shared[tid] = reduce(shared[tid], shared[tid + i]);
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        index(0, result).scalar() = postprocess(shared[0]);
    }
}

template<int Dim, AnyTensor T>
requires(Dim >= 0 && Dim < T::Rank  && T::Rank <= 4 && T::device == kCUDA)
auto ReduceNode(const T& x, auto&& f, auto&& preprocess, auto&& postprocess) {
    Tensor<decltype(postprocess(f(typename T::ValueType(), typename T::ValueType()))),
           Allocator<T::device>, typename T::Dims::template Set<Dim, 1>> result;

    constexpr int Threads = std::min(1024, 1 << (63 - std::countl_zero(uint64_t(T::Dim(Dim)))));

    auto xx = x.template transpose<Dim, 0>();
    auto yy = result.template transpose<Dim, 0>();
    ReduceKernel<Threads><<<yy.Size, Threads, 0, stream>>>(xx, yy, f, preprocess, postprocess);
    return result;
}

} // namespace tadma