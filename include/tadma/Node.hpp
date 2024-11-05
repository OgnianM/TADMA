#pragma once
#include <tuple>
#include <array>
#include <cassert>

#include "Concepts.hpp"
#include "Meta.hpp"
#include "Sequence.hpp"
#include "Utils.hpp"
#include "Tagging.hpp"

namespace tadma {

template<AnySequence ExecutionShape> requires(ExecutionShape::Size > 0)
void CPUExecutor(auto&& f, auto&&... tensors) {
    constexpr int64_t Product = ExecutionShape::Product();
//#pragma omp parallel for
    for (int64_t index = 0; index < Product; index++) {
        f(MapToShape<ExecutionShape>(index), tensors...);
    }
}

// Just a basic entrypoint for any CUDA execution
#if defined(__CUDACC__) || defined(__HIP__)
template<AnySequence ExecutionShape, int Threads>
__global__ void CUDAHIPExecutorKernel(auto f, AnyTensor auto... tensors) {
    constexpr auto Product = ExecutionShape::Product();
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if constexpr (Product % Threads != 0) {
        if (index >= Product) return;
    }

    f(MapToShape<ExecutionShape>(index), tensors...);
}
#endif

template<AnySequence ExecutionShape> requires(ExecutionShape::Size > 0)
void CUDAHIPExecutor(auto f, auto... tensors) {
    constexpr auto Threads = 512;
    constexpr auto Product = ExecutionShape::Product();
    constexpr auto Blocks = (Product + Threads - 1) / Threads;
#ifdef __CUDACC__
    CUDAHIPExecutorKernel<ExecutionShape, Threads><<<Blocks, Threads, 0, stream>>>(f, tensors...);
#else
    assert(false && "CUDA/HIP code is being executed on a non-CUDA device");
#endif
}

template<typename Tuple>
consteval std::array<int, std::tuple_size_v<Tuple>> GetPostMergeMapping() {
    std::array<int, std::tuple_size_v<Tuple>> result;
    result[0] = 0;
    auto tags = GetTupleTags<Tuple>();

    auto dims = new int64_t*[std::tuple_size_v<Tuple>];

    constexpr_for<0, std::tuple_size_v<Tuple>>([&]<int I>() {
        using T = typename std::tuple_element_t<I, Tuple>::Dims;
        auto dim = new int64_t[T::Size + 1];
        int index = 0;

        dim[index++] = T::Size;
        T::Apply([&]<auto... Ds>() {
            ((dim[index++] = Ds), ...);
        });
        dims[I] = dim;
    });

    auto equal = [&](int i, int j) {
        auto x = dims[i];
        auto y = dims[j];

        int sx = x[0];
        int sy = y[0];

        if (sx != sy) return false;

        for (int k = 0; k < sx; k++) {
            if (x[k+1] != y[k+1]) return false;
        }

        return true;
    };

    int next_index = 1;
    for (int i = 1; i < tags.size(); i++) {
        if (tags[i] == -1) {
            result[i] = next_index++;
            continue;
        }
        if (![&]() {
            for (int j = 0; j < i; j++) {
                if (tags[i] == tags[j] && equal(i, j)) {
                    result[i] = result[j];
                    return true;
                }
            }
            return false;
        }()) {
            result[i] = next_index++;
        }
    }

    for (int i = 0; i < std::tuple_size_v<Tuple>; i++) {
        delete[] dims[i];
    }
    delete[] dims;
    return result;
}

constexpr auto RemapTuple(auto TriviallyMerged) {
    constexpr auto Mapping = GetPostMergeMapping<TYPE(TriviallyMerged)>();

    return constexpr_for<0, std::tuple_size_v<TYPE(TriviallyMerged)>>([&]<int I>(auto tuple) {
        static_assert(Mapping[I] <= std::tuple_size_v<TYPE(tuple)>, "Invalid mapping");

        if constexpr (Mapping[I] == std::tuple_size_v<TYPE(tuple)>) {
            return std::tuple_cat(tuple, std::tuple {std::get<I>(TriviallyMerged) });
        } else {
            return tuple;
        }
    }, std::make_tuple());
}

// The node is a class symbolizing a piece of computation, it may have multiple inputs and outputs as well as a function
// Evaluating the node evaluates all of its inputs and then dispatches the contained function for computation on the appropriate device
template<typename ExecutionShape_, typename F, typename... Ts>
struct Node {
    std::decay_t<F> f;
    std::tuple<std::remove_reference_t<Ts>...> tensors;
    using ExecutionShape = ExecutionShape_;
    static constexpr Memory device = tadma::CommonDevice<Ts...>();

    explicit Node(F& f, Ts... tensors) : f(f), tensors(tensors...) {}

    template<int I, typename Self>
    static consteval bool IsOutput() {
        constexpr auto Tag = GetTag<Ts...[I]>();
        if constexpr (Tag == -1) {
            return true;
        } else if constexpr (IsTagged<Self>) {
            return Tag > Self::Tag;
        } else {
            return false;
        }
    }

    void Evaluate(auto state) {
        std::apply([&](auto&&... ts) {
            if constexpr (std::is_same_v<ExecutionShape, Sequence<>>) {
                f(state, ts...);
            } else if constexpr (device == kCPU) {
                CPUExecutor<ExecutionShape>(f, ts...);
            } else if constexpr (device == kCUDA) {
                CUDAHIPExecutor<ExecutionShape>(f, ts...);
            } else {
                static_assert(device == -1, "Invalid device");
            }
        }, tensors);
    }

    // Return the const lvalue references and non-reference types in F's parameter list
    constexpr decltype(auto) outputs(this auto&& self) {
        if constexpr (sizeof...(Ts) == 1) {
            return self.tensors;
        } else {
            auto result = constexpr_for<0, sizeof...(Ts)>([&]<int I>(auto tuple) {
                if constexpr (IsOutput<I, TYPE(self)>()) {
                    return std::tuple_cat(tuple, std::tuple{std::get<I>(self.tensors)});
                } else {
                    return tuple;
                }
            }, std::tuple<>());
            static_assert(std::tuple_size_v<TYPE(result)> > 0);
            return result;
        }
    }
    auto result(this auto&& self) {
        return std::get<0>(self.outputs());
    }
};

template<AnySequence ExecutionShape>
auto make_node(auto&&... data) {
    return Node<ExecutionShape, std::remove_reference_t<decltype(data)>...>(data...);
}

auto MergeNodes(const auto& Node0, const auto& Node1) {
    static_assert(TYPE(Node0)::ExecutionShape::Product() == TYPE(Node1)::ExecutionShape::Product(), "Incompatible shapes");

    auto TrivialMerge = std::tuple_cat(Node0.tensors, Node1.tensors);
    auto FoldedMerge = RemapTuple(TrivialMerge);

    constexpr auto Size0 = std::tuple_size_v<TYPE(Node0.tensors)>;
    constexpr auto Size1 = std::tuple_size_v<TYPE(Node1.tensors)>;

    auto ComposedFunction = [f0 = Node0.f, f1 = Node1.f](auto indices, auto&&... tensors_) {
        static constexpr auto Mapping = GetPostMergeMapping<TYPE(TrivialMerge)>();

        std::apply([&](auto&&... selected) {
            f0(indices, selected...);
        }, constexpr_for<0, Size0>([&]<int I>(auto tuple) {
            return std::tuple_cat(tuple, std::make_tuple (std::ref(tensors_...[Mapping[I]])));
        }, std::make_tuple()));

        std::apply([&](auto&&... selected) {
            if constexpr (typename TYPE(Node0)::ExecutionShape() != typename TYPE(Node1)::ExecutionShape()) {
                f1(ShapeToShape<typename TYPE(Node0)::ExecutionShape, typename TYPE(Node1)::ExecutionShape>(indices), selected...);
            } else {
                f1(indices, selected...);
            }
        }, constexpr_for<Size0, Size0 + Size1>([&]<int I>(auto tuple) {
            return std::tuple_cat(tuple, std::make_tuple(std::ref(tensors_...[Mapping[I]])));
        }, std::make_tuple()));
    };
    return std::apply([&](auto... data) {
        return make_node<typename TYPE(Node0)::ExecutionShape>(ComposedFunction, data...);
    }, FoldedMerge);
}

auto MergeNodes(const auto& Node0, const auto& Node1, const auto&... Nodes) requires(sizeof...(Nodes) > 0) {
    return MergeNodes(MergeNodes(Node0, Node1), Nodes...);
}

template<typename T> concept IsNode = requires {
    typename T::ExecutionShape;
};

};
