#include <cmath>
#include <iostream>
#include "include/tadma/Tadma.hpp"
#include <utility>
#include <string>
#include <string_view>
#include <concepts>
namespace T = tadma;


auto InputNode(auto&&... tensors) {
    return tadma::make_node<tadma::Sequence<>>([](const auto&...){}, std::forward<decltype(tensors)>(tensors)...);
}

auto OutputNode(auto&&... tensors) {
    return tadma::make_node<tadma::Sequence<>>([](const auto&...){}, std::forward<decltype(tensors)>(tensors)...);
}

auto MakeGraph() {
    T::ExecutionTape tape;

    auto [tape2, t1] = tape.Append(T::index_to_value<T::Sequence<10, 10>, T::kCUDA>([](float index) ->float { return cos(index); }));
    auto [tape3, t2] = tape2.Append(t1 * 5.f);
    auto [tape4, t3] = tape3.Append(t2 - 5.f);
    auto [tape5, t4] = tape4.Append(t3 / (float)M_PI);
    auto [tape6, t5] = tape5.Append(-t4);
    auto [tape7, t6] = tape6.Append(t5 - t3);
    auto [tape8, t7] = tape7.Append(t5 * t3);
    auto [tape9, t8] = tape8.Append(tanh(t7));
    auto [tape10, t9] = tape9.Append(T::index_to_value<T::Sequence<50>, T::kCUDA>([](float index) ->float { return sin(index); }));

    auto [final, _] = tape10.Append(OutputNode(t8));

    return final.Optimize();
}

int main() {
    T::InitGuard guard;
    auto graph = MakeGraph();
    graph.Evaluate(0);
    std::cout << std::get<0>(graph.Last().tensors);

    return 0;
}
