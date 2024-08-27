#pragma once
#include "Meta.hpp"

namespace tadma {

template<auto... Is>
struct Sequence {
    static constexpr auto Size = sizeof...(Is);
    static consteval auto Product() requires(Size > 0) { return (Is * ...); }
    static consteval auto Sum() requires(Size > 0) { return (Is + ...); }

    template<auto X> using Append = Sequence<Is..., X>;
    template<auto X> using Prepend = Sequence<X, Is...>;

    template<typename T>
    consteval operator T() const requires(Size == 1) {
        return Values(0);
    }

    static constexpr decltype(auto) Apply(auto&& f) {
        return f.template operator()<Is...>();
    }

    static constexpr auto Tuple() {
        constexpr auto tuple = std::tuple(Is...);
        return tuple;
    }

    template<typename Type>
    static consteval auto Array() {
        return std::array<Type, Size>{Is...};
    }

    static consteval auto Array() {
        return std::array<std::common_type_t<decltype(Is)...>, Size>{Is...};
    }

    template<int I> static consteval auto Values() {
        return std::get<I < 0 ? Size + I : I>(Tuple());
    }

    static consteval auto Values(int64_t I) requires (HaveCommonType<Is...>) {
        return Array()[I];
    }

    template<int A, int B, int I = 0, typename State = Sequence<>>
    static consteval auto SwapImpl() {
        if constexpr (I == Size) {
            return State{};
        } else if constexpr (I == A) {
            return SwapImpl<A, B, I + 1, typename State::template Append<Values<B>()>>();
        } else if constexpr (I == B) {
            return SwapImpl<A, B, I + 1, typename State::template Append<Values<A>()>>();
        } else {
            return SwapImpl<A, B, I + 1, typename State::template Append<Values<I>()>>();
        }
    }


    template<int N, typename State = Sequence<>> requires(N < Size)
    static consteval auto LastImpl() {
        if constexpr (N == 0) {
            return State{};
        } else {
            return LastImpl<N - 1, typename State::template Append<Values<Size - N>()>>();
        }
    }

    template<int N, int I = 0, typename State = Sequence<>> requires(N < Size)
    static consteval auto FirstImpl() {
        if constexpr (N == I) {
            return State{};
        } else {
            return FirstImpl<N, I + 1, typename State::template Append<Values<I>()>>();
        }
    }

    template<typename Seq, typename State = Sequence<Is...>>
    static consteval auto MergeImpl() {
        if constexpr (Seq::Size == 0) {
            return State{};
        } else {
            return MergeImpl<typename Seq::template Last<Seq::Size - 1>, typename State::template Append<Seq::template Values<0>()>>();
        }
    }

    template<int I, int C = 0, typename State = Sequence<>>
    static consteval auto RemoveImpl() {
        if constexpr (C == Size) {
            return State{};
        } else if constexpr (C == I) {
            return RemoveImpl<I, C + 1, State>();
        } else {
            return RemoveImpl<I, C + 1, typename State::template Append<Values<C>()>>();
        }
    }

    template<int I, auto V, int C = 0, typename State = Sequence<>>
    static consteval auto InsertImpl() {
        if constexpr (C == Size) {
            return State{};
        } else if constexpr (C == I) {
            return InsertImpl<I, V, C + 1, typename State::template Append<V>::template Append<Values<C>()>>();
        } else {
            return InsertImpl<I, V, C + 1, typename State::template Append<Values<C>()>>();
        }
    }

    template<int I, auto V, int C = 0, typename State = Sequence<>>
    static consteval auto SetImpl() {
        if constexpr (C == Size) {
            return State{};
        } else if constexpr (C == I) {
            return SetImpl<I, V, C + 1, typename State::template Append<V>>();
        } else {
            return SetImpl<I, V, C + 1, typename State::template Append<Values<C>()>>();
        }
    }

    template<typename Other, int ThisI = 0, int OtherI = 0>
    static consteval bool ContainsOrderedSubsetImpl() {
        if constexpr(OtherI == Other::Size) {
            return true;
        } else if constexpr (ThisI == Size) {
            return false;
        } else if constexpr (Values<ThisI>() == Other::template Values<OtherI>()) {
            return ContainsOrderedSubsetImpl<Other, ThisI + 1, OtherI + 1>();
        } else {
            return ContainsOrderedSubsetImpl<Other, ThisI + 1, OtherI>();
        }
    }

public:
    template<int A, int B> using Swap = decltype(SwapImpl<A, B>());
    template<int N> using First = decltype(FirstImpl<N>());
    template<int N> using Last = decltype(LastImpl<N>());
    template<typename Seq> using Merge = decltype(MergeImpl<Seq>());
    template<int I> using Remove = decltype(RemoveImpl<I>());
    template<int I, auto V> using Insert = decltype(InsertImpl<I, V>());
    template<int I, auto V> using Set = decltype(SetImpl<I, V>());


   	template<typename Other>
    consteval bool operator==(const Other& other) const {
        if constexpr (Size != Other::Size) {
            return false;
        } else {
            return constexpr_for<0, Size>([&]<int I>(bool x) {
                return x && Values(I) == Other::Values(I);
            }, true);
        }
    }

    template<auto X> static constexpr int IndexOf = constexpr_for<0, Size>([]<int I>(auto x) {
        if constexpr (Values<I>() == X) {
            return I;
        } else return x;
    }, -1);
    /**
     * @brief Does this set contain another set as an ordered subset?
     * @note The other set doesn't need to be contiguous in this set
     * @example Sequence<5, 10, 11>::ContainsOrderedSubset<Sequence<5, 11>>() == true
     */
    template<typename U> static consteval bool ContainsOrderedSubset() {
        return ContainsOrderedSubsetImpl<U>();
    }

    static constexpr bool IsSorted = constexpr_for<0, Size - 1>([]<int I>(bool x) {
        return x && Values<I>() <= Values<I + 1>();
    }, true);
};


template<AnySequence S> using Sort = decltype([]<AnySequence Seq = S>(this auto&& self) {
   using Sorted = decltype(constexpr_for<0, Seq::Size - 1>([]<int I>(auto seq) {
       using T = decltype(seq);
       if constexpr (T::Values(I) > T::Values(I + 1)) {
           return typename T::template Swap<I, I + 1>();
       } else {
           return T();
      }
   }, Seq()));
   if constexpr (!Sorted::IsSorted) {
       return self.template operator()<Sorted>();
   } else {
       return Sorted();
   }
}());

static_assert(Sequence<5, 10, 11>::ContainsOrderedSubset<Sequence<5, 11>>(), "Sanity check failed");

template<auto... Is> std::ostream& operator<<(std::ostream& os, Sequence<Is...>) {
    return ((os << Is << ", "), ...);
}

#define OPERATOR(op) \
template<AnySequence S0, AnySequence S1> requires(S0::Size == S1::Size)\
consteval auto operator op(S0, S1) {\
    return constexpr_for<0, S0::Size>([]<int I>(auto seq) {\
        return typename TYPE(seq)::template Append<S0::Values(I) op S1::Values(I)>();\
    }, Sequence<>());\
}\
template<AnySequence S0, AnySequence S1> requires(S1::Size == 1)\
consteval auto operator op(S0, S1) {\
    return constexpr_for<0, S0::Size>([&]<int I>(auto seq) {\
        return typename TYPE(seq)::template Append<S0::Values(I) op S1::Values(0)>();\
    }, Sequence<>());\
}

OPERATOR(+) OPERATOR(-) OPERATOR(*) OPERATOR(/) OPERATOR(%) OPERATOR(&) OPERATOR(|) OPERATOR(^) OPERATOR(!=)
#undef OPERATOR

template<AnySequence S0, AnySequence S1> requires(S0::Size == S1::Size)
consteval auto Equal(S0, S1) {
    return constexpr_for<0, S0::Size>([]<int I>(auto seq) {
        return typename TYPE(seq)::template Append<S0::Values(I) == S1::Values(I)>();
    }, Sequence<>());
}
template<AnySequence S0, AnySequence S1> requires(S1::Size == 1)
consteval auto Equal(S0, S1) {
    return constexpr_for<0, S0::Size>([&]<int I>(auto seq) {
        return typename TYPE(seq)::template Append<S0::Values(I) == S1::Values(0)>();
    }, Sequence<>());
}

template<AnySequence Cond, AnySequence X, AnySequence Y> requires(Cond::Size == X::Size && X::Size == Y::Size)
consteval auto where(Cond, X, Y) {
    return constexpr_for<0, Cond::Size>([&]<int I>(auto seq) {
        return typename TYPE(seq)::template Append<Cond::Values(I) ? X::Values(I) : Y::Values(I)>();
    }, Sequence<>());
}

template<int Axis = 0, AnySequence... S> requires(Axis == 0)
auto concat(S... s) {

    return constexpr_for<0, sizeof...(S)>([]<int I>(auto seq) {
        return typename TYPE(seq)::template Merge<TYPE(std::get<I>(std::make_tuple(s...)))>();
    }, Sequence<>());

}


};
