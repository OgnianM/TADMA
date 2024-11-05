#pragma once
#include "Node.hpp"


namespace tadma {

template<typename TensorTuple, int Start>
consteval std::array<int, std::tuple_size_v<TensorTuple>> AssignTags() {
    auto Tags = GetTupleTags<TensorTuple>();
    for (int NextTag = Start, i = 0; i < Tags.size(); i++) {
        if (Tags[i] == -1) {
            Tags[i] = NextTag++;
        }
    }
    return Tags;
}

/**
 * @brief Used to store a computation graph in a linearized format
 * Roughly, the format looks like [Node0, Node0Output0, Node0Output1, Node1, Node1Output0, ...]
 * The outputs are arbitrary tensors, each of them is tagged with their index in the tape through the Tagged<...> template
 * This provides a convenient way to disambiguate inputs from outputs, if a tensor passed to a node is tagged,
 * then it must necessarily have been returned by another node through the ExecutionTape, if it's not tagged,
 * then it's an output of the current node
 */
template<int NextTag = 0, typename TupleType_ = std::tuple<>>
struct ExecutionTape {
    using TupleType = TupleType_;

    TupleType data;
    static constexpr auto Size = std::tuple_size_v<TupleType>;

    ExecutionTape() = default;
    ExecutionTape(const TupleType& tuple) : data(tuple) {}

    /**
     * Take a node and append it to the tape such that
     * NewTape = Tape<Data...>{}.Append(Node) -> Tape<Data..., NodeX, TaggedNodeXOutput0, TaggedNodeXOutput1, ...>
     * @return tuple { NewTape, TaggedNodeXOutput0, TaggedNodeXOutput1, ... }
    */
    template<bool DoNotMerge=false>
    constexpr auto Append(auto&& node0) {
        using Node = TYPE(node0);
        using NodeTupleType = TYPE(node0.tensors);
        constexpr auto AssignedTags = AssignTags<NodeTupleType, NextTag+1>();

        using TaggedTupleType = decltype(constexpr_for<0, std::tuple_size_v<NodeTupleType>>([&]<int I>(auto tuple) {
            if constexpr (!IsTagged<std::tuple_element_t<I, NodeTupleType>>) {
                return std::tuple_cat(tuple, std::tuple{AddTag<AssignedTags[I]>(std::get<I>(node0.tensors))});
            } else {
                static_assert(GetTag<std::tuple_element_t<I, NodeTupleType>>() == AssignedTags[I], "Sanity check: Tag mismatch");
                return std::tuple_cat(tuple, std::tuple{std::get<I>(node0.tensors)});
            }
        }, std::make_tuple()));

        auto& TaggedTuple = reinterpret_cast<TaggedTupleType&>(node0.tensors);
        auto FullyTagged = [&]<typename... Args>(const std::tuple<Args...>&) {
            return reinterpret_cast<Tagged<NextTag, tadma::Node<typename TYPE(node0)::ExecutionShape, decltype(TYPE(node0)::f), Args...>>&>(node0);
        }(TaggedTuple);
        constexpr auto MergeIndex = FindMergeable<TYPE(AddTag<NextTag>(node0))>();
        constexpr auto NextNextTag = std::max(NextTag + 1, std::ranges::max(AssignedTags) + 1);

        auto NewData = [&] {
            if constexpr (MergeIndex == -1 || DoNotMerge) {
                return std::tuple_cat(data, std::tuple{FullyTagged});
            } else {
                return tuple_replace<MergeIndex>(data, MergeNodes(std::get<MergeIndex>(data), FullyTagged));
            }
        }();
        return std::tuple_cat(std::tuple {tadma::ExecutionTape<NextNextTag, TYPE(NewData)>(NewData)}, FullyTagged.outputs());
    }

    // Given a complete tape, eliminate all dangling tensors
    constexpr auto Optimize() {
        constexpr auto Refs = CountTensorRefs();

        auto NewTape = constexpr_for<0, Size>([&]<int I>(auto NewTape) {
            using T = std::tuple_element_t<I, TupleType>;
            static_assert (IsNode<T>);

            using TensorTuple = TYPE(T::tensors);
            auto NewTuple = constexpr_for<0, std::tuple_size_v<TensorTuple>>([&]<int J>(auto result) {
                using TensorType = std::tuple_element_t<J, TensorTuple>;
                if constexpr (Refs[GetTag<TensorType>()] == 1) {
                    // The tensor is only referenced within this node and may be replaced with a trampoline

                    return std::tuple_cat(result, std::tuple {
                        Tensor<typename TensorType::ValueType, Allocator<Memory::kTrampoline>, typename TensorType::Dims>()
                    });
                } else {
                    return std::tuple_cat(result, std::tuple {std::get<J>(std::get<I>(data).tensors)});
                }
            }, std::make_tuple());

            return std::apply([&](auto&&... tensors) {
                return std::tuple_cat(NewTape, std::tuple {
                    make_node<typename T::ExecutionShape>(std::get<I>(data).f, tensors...)
                });
            }, NewTuple);
        }, std::make_tuple());
        return tadma::ExecutionTape<NextTag, TYPE(NewTape)>(NewTape);
    }

    void Evaluate(auto state) {
        constexpr_for<0, Size>([&]<int I>() {
            if constexpr (IsNode<std::tuple_element_t<I, TupleType>>) {
                std::get<I>(data).Evaluate(state);
            }
        });
    }

    auto& Last() {
        return std::get<Size - 1>(data);
    }
private:
    static consteval auto CountTensorRefs() {
        std::array<int, NextTag> TensorRefs;
        TensorRefs.fill(0);
        constexpr_for<0, Size>([&]<int I>() {
            using T = std::tuple_element_t<I, TupleType>;
            if constexpr (IsNode<T>) {
                using TensorTuple = TYPE(T::tensors);
                constexpr_for<0, std::tuple_size_v<TensorTuple>>([&]<int J>() {
                    TensorRefs[GetTag<std::tuple_element_t<J, TensorTuple>>()]++;
                });
            }
        });
        return TensorRefs;
    }

    template<typename Node>
    static consteval int FindMergeable() {
        if constexpr (std::is_same_v<typename Node::ExecutionShape, Sequence<>>) {
            return -1;
        }
        constexpr int TupleSize = std::tuple_size_v<TupleType>;
        using NodeTensors = TYPE(Node::tensors);
        auto InputTags = GetTupleTags<NodeTensors>();
        auto TupleTags = GetTupleTags<TupleType>();

        // The required criterion for nodes to be compatible is that they have the same ExecutionShape::Product()
        // The compatibility score is the number of shared tensors between the two nodes, other criteria may be added later
        int MaxCompatibilityScore = -1;
        int BestMergeIndex = -1;
        int LeastAllowedIndex = 0;

        auto CountSharedTensors = [&]<int I>() {
            using T = std::tuple_element_t<I, TupleType>;
            return constexpr_for<0, std::tuple_size_v<decltype(T::tensors)>>([&]<int J>(int accumulator) {
                constexpr auto Tag = GetTag<std::tuple_element_t<J, decltype(T::tensors)>>();
                if constexpr (Tag != -1) {
                    for (auto InputTag : InputTags) {
                        if (InputTag == Tag) {
                            return accumulator + 1;
                        }
                    }
                }
                return accumulator;
            }, 0);
        };
        constexpr_rfor<0, TupleSize>([&]<int I>() {
            using T = std::tuple_element_t<I, TupleType>;

            if constexpr (IsNode<T>) {
                constexpr auto IOTags = GetTupleTags<TYPE(T::tensors)>();

                [&]() {
                    for (int i = 0; i < InputTags.size(); i++) {
                        for (int j = 0; j < IOTags.size(); j++) {
                            if (InputTags[i] == IOTags[j]) {
                                LeastAllowedIndex = I;
                                return;
                            }
                        }
                    }
                }();

                /// TODO: There's a bug here in the following hypothethical
                /// Node A produces tensor X
                /// Node B requires a reshaped tensor X (the indices in A do not correspond to B)
                /// Merging A and B can cause a thread executing B to reference data that hasn't yet been written by another thread in A!
                if constexpr (T::ExecutionShape::Product() == Node::ExecutionShape::Product()) {
                    int CompatibilityScore = CountSharedTensors.template operator()<I>();
                    if (CompatibilityScore > MaxCompatibilityScore && I >= LeastAllowedIndex) {
                        MaxCompatibilityScore = CompatibilityScore;
                        BestMergeIndex = I;
                    }
                }
            }
        });
        return BestMergeIndex;
    }
};

}