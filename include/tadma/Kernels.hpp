#pragma once
#include "Tensor.hpp"

namespace tadma {

template<int Dim, int Threads=1024, AnyTensor T, AnyTensor RT, typename Reduce> requires(Commutative<Reduce>)
__global__ void ReduceKernel(T t, RT result, Reduce reduce, auto postprocess) {
    constexpr auto Size = T::Dim(Dim);

    [[assume(Threads < Size ? threadIdx.x < Size : true)]];

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

__global__ void InplaceKernel(AnyTensor auto t, auto f) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t.ContiguousSize) t[i] = f(t[i]);
}

__global__ void EltwiseKernel(AnyTensor auto in, AnyTensor auto out, auto f) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in.ContiguousSize) out[i] = f(in[i]);
}

__global__ void EltwiseCombineKernel(AnyTensor auto a, AnyTensor auto b, auto f) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.ContiguousSize) a[i] = f(a[i], b[i]);
}

__global__ void EltwiseCombineToKernel(AnyTensor auto a, AnyTensor auto b, AnyTensor auto c, auto f) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.ContiguousSize) c[i] = f(a[i], b[i]);
}

__global__ void EltwiseCombineVariadicKernel(auto f, AnyTensor auto t0, AnyTensor auto... sources) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t0.Size) f(t0[i], sources[i]...);
}


template<int Dim, AnyTensor T> requires(Dim >= 0 && Dim < T::Rank  && T::Rank <= 4 && T::device == kCUDA)
auto ReduceNode(T input, auto f, auto&& postprocess = []__multi__(const auto& x){ return x; }) {
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

    ReduceKernel<Dim><<<grid, block, 0, stream>>>(input, result, f, postprocess);
    return result;
}

auto InplaceNode(AnyTensor auto&& t, auto&& f) {
    InplaceKernel<<<(t.Size + 255) / 256, 256, 0, stream>>>(t.removeFakeDims(), f);
    return t;
}

template<AnyTensor T>
auto EltwiseNode(T t, auto&& f) {
    Tensor<std::decay_t<decltype(f(typename T::ValueType()))>, typename T::AllocatorType, typename T::Dims, Sequence<>> c;
    EltwiseKernel<<<(t.Size + 255) / 256, 256, 0, stream>>>(t, c, f);
    return c;
};

template<typename F, AnyTensor T1, AnyTensor T2>
auto CombineNode(T1& t, T2& other, const F& f) {
    EltwiseCombineKernel<<<(T1::Size + 255) / 256, 256, 0, stream>>>(t, other.broadcastTo(t), f);
    return t;
}

template<AnyTensor T1, AnyTensor T2, typename F>
auto CombineToNode(T1& a, T2& b, const F& f) {
    using ReturnType = std::decay_t<decltype(f(typename T1::ValueType(), typename T2::ValueType()))>;
    Tensor<ReturnType, typename T1::AllocatorType, typename T1::Dims> c;
    EltwiseCombineToKernel<<<(T1::Size + 255) / 256, 256, 0, stream>>>(a, b.broadcastTo(c), c, f);
    return c;
}

template<AnyTensor T1, AnyTensor... Ts>
void CombineVariadicNode(auto&& f, T1&& t0, Ts&&... tensors) requires(SameDevice<T1, Ts...>) {
    auto x = std::tuple {t0, tensors.broadcastTo(t0)...};
    EltwiseCombineVariadicKernel<<<(t0.Size + 255) / 256, 256, 0, stream>>>(f, t0, tensors.broadcastTo(t0)...);
}

template<bool Inplace> auto MakeEltwiseNode(AnyTensor auto&& t, auto f) {
    if constexpr (Inplace) {
        return InplaceNode(t, f);
    } else {
        return EltwiseNode(t, f);
    }
}

template<bool Inplace> auto MakeEltwiseCombineNode(AnyTensor auto t, AnyTensor auto t2, auto f) {
    if constexpr (Inplace) {
        return CombineNode(t, f);
    } else {
        return CombineToNode(t, f);
    }
}

};
