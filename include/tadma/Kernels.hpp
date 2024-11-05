#pragma once
#include "Tensor.hpp"
#include "kernels/Reduce.hpp"
#include "kernels/EltwiseVariadicND.hpp"
#include "tadma/Node.hpp"


namespace tadma {

auto CombineVariadicNode(auto&& f, AnyTensor auto&& t0, AnyTensor auto&&... tensors) {
    return make_node<typename TYPE(t0)::Dims>([f](const auto& indices, auto&&... tensors) {
        f(tensors(indices)...);
    }, t0, tensors.broadcastTo(t0)...);
}

auto InplaceNode(AnyTensor auto& t, auto&& f) {
    return CombineVariadicNode([f](auto& x) { x = f(x); }, t.removeFakeDims());
}

template<AnyTensor T>
constexpr auto EltwiseNode(const T& x, auto&& f) {
    Tensor<std::decay_t<decltype(f(typename T::ValueType()))>, RemoveTag_t<typename T::Allocator>, typename T::Dims> y;
    return CombineVariadicNode([f](const auto& x, auto& y) {
        y = f(x);
    }, x, y);
}

template<typename F, AnyTensor T1, AnyTensor T2>
auto CombineNode(T1& x, const T2& y, const F& f) {
    return CombineVariadicNode([f](auto& x, const auto& y) { x = f(x, y); }, x, y);
}

template<AnyTensor T1, AnyTensor T2, typename F>
auto CombineToNode(const T1& x, const T2& y, const F& f) {
    using ReturnType = std::decay_t<decltype(f(typename T1::ValueType(), typename T2::ValueType()))>;

    if constexpr (!Broadcastable<T2, T1> && Broadcastable<T1, T2> && Commutative<F>) { // Commutative operations may be inverted
        return CombineToNode(y, x, f);
    } else {
        Tensor<ReturnType, RemoveTag_t<typename T1::Allocator>, typename T1::Dims> z;
        return CombineVariadicNode([f](const auto& x, const auto& y, auto& z) {
            z = f(x, y);
        }, x, y.broadcastTo(z), z);
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
