#pragma once
#include <type_traits>
#include <cxxabi.h>

#define TYPE(x) std::decay_t<decltype(x)>

namespace std {
    template<int Index>
    constexpr decltype(auto) get(auto&&... pack) {
        return []<int I = 0>(this auto&& self, auto&& x, auto&&... pack) {
            if constexpr (I == Index) return x;
            else return self.template operator()<I + 1>(pack...);
        }(pack...);
    }
};

namespace tadma {

template <typename T> concept AnyTensor = requires(T t) {
    std::decay_t<T>::Dim(0);
    std::decay_t<T>::Stride(0);
    std::decay_t<T>::device;
    t.data;
};


template<typename T> concept AnySequence = requires {
    T::Size;
    typename T::template Append<1>();
    typename T::template Prepend<1>();
};

template<typename T> concept Scalar = std::is_arithmetic_v<T>;

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


template<typename T, typename... Ts> concept SameDevice = ((std::decay_t<T>::device == std::decay_t<Ts>::device) && ...);
template<typename T, typename... Ts> concept SameDims = ((typename T::Dims() == typename Ts::Dims()) && ...);
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
        constexpr_rfor<I + 1, N>(f);
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
        return f.template operator()<I>(std::forward<Arg>(constexpr_rfor<I + 1, N>(f, arg)));
    } else {
        return arg;
    }
}


template<int Dim, int Rank> constexpr int RealDim = Dim < 0 ? Rank + Dim : Dim;

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

template<int Index>
constexpr decltype(auto) parameter_pack_replace(auto new_value, auto F, auto&&... pack) requires(Index < sizeof...(pack)) {
    auto t = std::make_tuple(pack...);
    std::get<Index>(t) = new_value;
    return std::apply(F, t);
}

template<int Size>
constexpr decltype(auto) parameter_pack_clip(auto F, auto&&... pack) requires (sizeof...(pack) >= Size) {
    std::array<std::common_type_t<decltype(pack)...>, sizeof...(pack)> original {pack...};
    std::array<std::common_type_t<decltype(pack)...>, Size> clipped;
    for (int i = 0; i < Size; i++) {
        clipped[i] = original[i];
    }
    return std::apply(F, clipped);
}

constexpr decltype(auto) parameter_pack_reverse(auto F, auto&&... pack) {
    std::array<std::common_type_t<decltype(pack)...>, sizeof...(pack)> arr {pack...};
    std::reverse(arr.begin(), arr.end());
    return std::apply(F, arr);
}

constexpr decltype(auto) parameter_pack_sort(auto F, auto&&... pack) {
    std::array<std::common_type_t<decltype(pack)...>, sizeof...(pack)> arr {pack...};
    std::sort(arr.begin(), arr.end());
    return std::apply(F, arr);
}

template<int Index>
constexpr decltype(auto) parameter_pack_index(auto&&... pack) {
    return std::get<Index>(std::make_tuple(pack...));
}

}



