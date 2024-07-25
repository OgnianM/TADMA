#pragma once

namespace tadma {

template<auto... Is>
struct Sequence {
    static constexpr auto Size = sizeof...(Is);
    static consteval auto Product() requires(Size > 0) {
        return (Is * ...);
    }

    template<auto X> using Append = Sequence<Is..., X>;
    template<auto X> using Prepend = Sequence<X, Is...>;


    static constexpr void Apply(auto&& f) {
        f.template operator()<Is...>();
    }

    static constexpr auto Tuple() {
        constexpr auto tuple = std::tuple(Is...);
        return tuple;
    }

    template<int I> static consteval auto Values() {
        return std::get<I>(Tuple());
    }

    static consteval auto Values(int I) requires (HaveCommonType<Is...>) {
        constexpr std::array<std::common_type_t<decltype(Is)...>, Size> arr{Is...};
        return arr[I];
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

    consteval bool operator==(const Sequence& other) const { return true; }

    template<typename U> requires (!std::is_same_v<U, Sequence>)
    consteval bool operator==(const U& other) const { return false; }

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
};

static_assert(Sequence<5, 10, 11>::ContainsOrderedSubset<Sequence<5, 11>>(), "Sanity check failed");

template<int... Is>
std::ostream& operator<<(std::ostream& os, Sequence<Is...>) {
    return ((os << Is << ", "), ...);
}

};
