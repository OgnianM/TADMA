#pragma once
#include <type_traits>
#include "Tensor.hpp"

namespace tadma {
    // Serves to uniquely identify an object at compile time
    template<int Tag_, typename Self>
    struct Tagged : Self {
        using UntaggedType = Self;
        static constexpr int Tag = Tag_;
    };

    template<typename T> concept IsTagged = requires { T::Tag; };

    template<typename T>
    consteval int GetTag() {
        if constexpr (AnyTensor<T>) {
            return GetTag<typename T::Allocator>();
        } else if constexpr (IsTagged<T>) {
            return T::Tag;
        } else {
            return -1;
        }
    }

    template<int Tag, typename T_>
    constexpr auto AddTag(T_&& t_) {
        using T = std::decay_t<T_>;
        auto& t = const_cast<T&>(t_);

        if constexpr (GetTag<T>() == Tag) {
            return t;
        } else if constexpr (AnyTensor<T>) {
            /// Tensors have mutable types, so a tag attached to the type would be dropped after something like a broadcast
            /// So inject the tag inside of the tensor's template arguments, making it persistent across all permutations
            /// Here we wrap the Allocator type in a Tagged<Tag, Allocator> type
            /// TODO: Anything that piggybacks off a tensor's allocator to create a new tensor is going to erroneously duplicate the tag

            static_assert(!IsTagged<typename T::Allocator>, "Duplicate tag.");
            return reinterpret_cast<
                Tensor<typename T::ValueType, Tagged<Tag, typename T::Allocator>, typename T::Dims, typename T::Strides>&>(t);
        } else {
            static_assert(!IsTagged<T>, "Duplicate tag.");
            return reinterpret_cast<Tagged<Tag, T>&>(t);
        }
    }



    template<typename T_>
    constexpr auto RemoveTag(T_&& t) {
        using T = std::decay_t<T_>;
        if constexpr (AnyTensor<T>) {
            if constexpr (IsTagged<typename T::Allocator>) {
                return reinterpret_cast<
                    Tensor<typename T::ValueType, typename T::Allocator::UntaggedType, typename T::Dims, typename T::Strides>&>(t);
            } else {
                return t;
            }
        } else if constexpr (IsTagged<T>) {
            return reinterpret_cast<typename T::UntaggedType&>(t);
        } else {
            return t;
        }
    }


    template<typename Tuple>
    consteval std::array<int, std::tuple_size_v<Tuple>> GetTupleTags() {
        constexpr auto Count = std::tuple_size_v<Tuple>;
        std::array<int, Count> Tags;
        constexpr_for<0, Count>([&]<int I>() {
            Tags[I] = GetTag<std::tuple_element_t<I, Tuple>>();
        });
        return Tags;
    }

    template<typename T, typename Tuple>
    consteval int FindTagInTuple() {
        constexpr auto Tag = GetTag<T>();
        static_assert(Tag != -1, "The searched type must be tagged.");
        constexpr auto Tags = GetTupleTags<Tuple>();

        for (int i = 0; i < Tags.size(); i++) {
            if (Tags[i] == Tag)
                return i;
        }
        return -1;
    }


    template<typename T> using AddTag_t = TYPE(AddTag(std::declval<T&>()));
    template<typename T> using RemoveTag_t = TYPE(RemoveTag(std::declval<T&>()));
};