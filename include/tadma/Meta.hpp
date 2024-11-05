#pragma once
#include <type_traits>
#include <algorithm>

#define TYPE(x) std::decay_t<decltype(x)>

namespace tadma {

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

template<int I, int N>
constexpr void constexpr_for(auto&& f) {
    if constexpr (I < N) {
        f.template operator()<I>();
        constexpr_for<I + 1, N>(f);
    }
}

template<int I, int N>
constexpr void constexpr_rfor(auto&& f) {
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
/*
template<int Index>
constexpr decltype(auto) parameter_pack_replace(auto new_value, auto F, auto&&... pack) requires(Index < sizeof...(pack)) {
    auto t = std::make_tuple(pack...);
    std::get<Index>(t) = new_value;
    return std::apply(F, t);
}
*/

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

template<int Start, int End, typename T> requires(Start >= 0 && End < std::tuple_size_v<T>)
auto slice_tuple(T&& t) {
    return constexpr_for<Start, End>([&]<int I>(auto tuple) {
        return std::tuple_cat(tuple, std::make_tuple(std::get<I>(t)));
    }, std::make_tuple<>());
}

}



