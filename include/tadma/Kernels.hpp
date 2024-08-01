#pragma once
#include "Tensor.hpp"

namespace tadma {

template<int Dim, int Threads, AnyTensor T, AnyTensor RT, typename Reduce> requires(Commutative<Reduce>)
__global__ void ReduceKernel(T t, RT result, Reduce reduce, auto postprocess) {
    constexpr auto Size = T::Dim(Dim);

    using ValueType = typename T::ValueType;
    using ReduceType = decltype(reduce(ValueType(), ValueType()));

    static_assert(std::is_same_v<typename RT::ValueType, decltype(postprocess(ReduceType()))>,
        "Result tensor must have the same value type as the reduction function");

    auto tid = threadIdx.x;

    auto index = [](int i, auto& t) -> auto& {
        if constexpr (T::Rank == 1) {
            return t(i);
        } else if constexpr(T::Rank == 2) {
            if constexpr (Dim == 0) {
                return t(i, blockIdx.x);
            } else {
                return t(blockIdx.x, i);
            }
        } else if constexpr (T::Rank == 3) {
            if constexpr (Dim == 0) {
                return t(i, blockIdx.x, blockIdx.y);
            } else if constexpr (Dim == 1) {
                return t(blockIdx.x, i, blockIdx.y);
            } else {
                return t(blockIdx.x, blockIdx.y, i);
            }
        } else if constexpr (T::Rank == 4) {
            if constexpr (Dim == 0) {
                return t(i, blockIdx.x, blockIdx.y, blockIdx.z);
            } else if constexpr (Dim == 1) {
                return t(blockIdx.x, i, blockIdx.y, blockIdx.z);
            } else if constexpr (Dim == 2) {
                return t(blockIdx.x, blockIdx.y, i, blockIdx.z);
            } else {
                return t(blockIdx.x, blockIdx.y, blockIdx.z, i);
            }
        } else {
            static_assert(false, "Unsupported rank");
        }
    };
    __shared__ ReduceType shared[Threads];

    auto idx = tid;
    if (idx < Size) {
        auto value = index(idx, t);

#pragma unroll
        for (idx += Threads; idx < Size; idx += Threads) {
            value = reduce(value, index(idx, t));
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
        index(0, result) = postprocess(shared[0]);
    }
}

template<int Dim, AnyTensor T>
requires(Dim >= 0 && Dim < T::Rank  && T::Rank <= 4 && T::device == kCUDA)
auto ReduceNode(const T& x, auto&& f, auto&& postprocess) {
    Tensor<decltype(postprocess(f(typename T::ValueType(), typename T::ValueType()))),
           Allocator<T::device>, typename T::Dims::template Set<Dim, 1>> result;

    // TODO: Reshape launch params to better fit input data
    dim3 grid;
    dim3 block(1024);

    int l = 0;

    constexpr_for<0, T::Rank>([&]<int I>() {
        if constexpr (I != Dim) {
            switch (l) {
                case 0: grid.x = T::Dim(I); break;
                case 1: grid.y = T::Dim(I); break;
                case 2: grid.z = T::Dim(I); break;
            }
            l++;
        }
    });

    constexpr int Threads = std::min(1024, 1 << (63 - std::countl_zero(uint64_t(T::Dim(Dim)))));
    block.x = Threads;

    ReduceKernel<Dim, Threads><<<grid, block, 0, stream>>>(x, result, f, postprocess);
    return result;
}

__global__ void EltwiseCombineVariadicKernel(auto f, AnyTensor auto x, AnyTensor auto... ys)
requires(SameDims<TYPE(x), TYPE(ys)> && ...) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < x.Size) f(x[i], ys[i]...);
}

template<AnyCudaTensor T1, AnyCudaTensor... Ts>
void CombineVariadicNode(auto&& f, T1&& t0, Ts&&... tensors) {
    EltwiseCombineVariadicKernel<<<(t0.Size + 255) / 256, 256, 0, stream>>>(f, t0, tensors.broadcastTo(t0)...);
}

template<AnyHostTensor T1, AnyHostTensor... Ts>
void CombineVariadicNode(auto&& f, T1&& t0, Ts&&... tensors) {
    [&](auto&&... ts) {
        #pragma omp parallel for
        for (size_t i = 0; i < t0.Size; i++) {
            f(ts[i]...);
        }
    }(t0, tensors.broadcastTo(t0)...);
}


auto InplaceNode(AnyTensor auto&& t, auto&& f) {
    CombineVariadicNode([f]__multi__(auto& x) {x = f(x);}, t.removeFakeDims());
    return t;
}

template<AnyTensor T>
auto EltwiseNode(const T& x, auto&& f) {
    Tensor<std::decay_t<decltype(f(typename T::ValueType()))>, typename T::AllocatorType, typename T::Dims, Sequence<>> y;
    CombineVariadicNode([f]__multi__(const auto& x, auto& y) {
        y = f(x);
    }, x, y);
    return y;
}

template<typename F, AnyTensor T1, AnyTensor T2>
auto CombineNode(T1& x, const T2& y, const F& f) {
    CombineVariadicNode([f]__multi__(auto& x, const auto& y) { x = f(x, y); }, x, y);
    return x;
}

template<AnyTensor T1, AnyTensor T2, typename F>
auto CombineToNode(const T1& x, const T2& y, const F& f) {
    if constexpr (!Broadcastable<T2, T1> && Broadcastable<T1, T2> && Commutative<F>) { // Commutative operations may be inverted
        return CombineToNode(y, x, f);
    } else {
        using ReturnType = std::decay_t<decltype(f(typename T1::ValueType(), typename T2::ValueType()))>;
        Tensor<ReturnType, typename T1::AllocatorType, typename T1::Dims> z;
        CombineVariadicNode([f]__multi__(const auto& x, const auto& y, auto& z) {
            z = f(x, y);
        }, x, y.broadcastTo(z), z);
        return z;
    }
}

template<bool Inplace> auto MakeEltwiseNode(AnyTensor auto&& x, auto&& f) {
    if constexpr (Inplace) {
        return InplaceNode(x, f);
    } else {
        return EltwiseNode(x, f);
    }
}

};
