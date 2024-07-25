#pragma once
#include <type_traits>
#include <cxxabi.h>

#define TYPE(x) std::decay_t<decltype(x)>

namespace tadma {


template<typename T> concept AnyAllocator = std::is_same_v<decltype(T::template allocate<int, 69>()), Storage<int>>;

template <typename T> concept AnyTensor = requires(T t) {
    std::decay_t<T>::Dim(0);
    std::decay_t<T>::Stride(0);
    std::decay_t<T>::device;
    t.view;
    t.data;
};

template<typename T> concept AnyCudaTensor = AnyTensor<T> && (T::device == Device::kCUDA);
template<typename T> concept AnyHostTensor = AnyTensor<T> && (T::device == Device::kCPU);

template<typename T> concept AnySequence = requires {
        T::Size;
        //T::Product();
};


template<typename A, typename B> concept Broadcastable = requires(A a, B b) {
    a.template broadcastTo<typename B::Dims>();
};

template<typename T>
constexpr decltype(auto) deconst(T&& t) {
    auto&& result = const_cast<std::remove_const_t< decltype(t)>&&>(t);
    static_assert(!std::is_const_v<decltype(result)>);
    return result;
}

template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

template<auto... Xs> concept HaveCommonType = requires { std::common_type_t<decltype(Xs)...>(); };

template<typename T> concept Scalar = std::is_arithmetic_v<T>;

template<typename T, typename... Ts> concept SameDevice = ((std::decay_t<T>::device == std::decay_t<Ts>::device) && ...);
template<typename T, typename... Ts> concept SameDims = (std::is_same_v<typename T::Dims, typename Ts::Dims> && ...);
template<typename T, typename... Ts> concept SameStrides = ((T::Strides() == Ts::Strides()) && ...);
template<typename T, typename... Ts> concept SameType = (std::is_same_v<typename T::ValueType, typename Ts::ValueType> && ...);
template<typename T, typename... Ts> concept SameRank = ((T::Rank == Ts::Rank) && ...);


template<int I, int N, typename FF>
constexpr void constexpr_for(FF f) {
    if constexpr (I < N) {
        f.template operator()<I>();
        constexpr_for<I + 1, N>(f);
    }
}

template<int I, int N, typename FF>
constexpr void constexpr_rfor(FF f) {
    if constexpr (I < N) {
        constexpr_for<I + 1, N>(f);
        f.template operator()<I>();
    }
}

template<int I, int N, typename FF, typename Arg>
constexpr auto constexpr_for(FF f, Arg&& arg) {
    if constexpr (I < N) {
        return constexpr_for<I + 1, N>(f, f.template operator()<I>(std::forward<Arg>(arg)));
    } else {
        return arg;
    }
}

template<int I, int N, typename FF, typename Arg>
constexpr auto constexpr_rfor(FF f, Arg&& arg) {
    if constexpr (I < N) {
        return f.template operator()<I>(std::forward<Arg>(constexpr_for<I + 1, N>(f, arg)));
    } else {
        return arg;
    }
}
template<typename T, AnyAllocator Allocator, AnySequence Dims_, AnySequence StridesInit> struct Tensor;

template<int Dim, int Rank> constexpr int RealDim = Dim < 0 ? Rank + Dim : Dim;


template<bool Sel> decltype(auto) constexpr_select(auto&& a, auto&& b) {
    if constexpr (Sel) {
        return a;
    } else {
        return b;
    }
};

template<typename F> constexpr bool Commutative = F()(97, 13) == F()(13, 97);

inline std::string demangle(const char* name) {
    int status;
    char* demangled = abi::__cxa_demangle(name, 0, 0, &status);
    std::string result(demangled);
    free(demangled);
    return result;
}

template<typename T> std::string typename_() {
    return demangle(typeid(T).name());
}

}
