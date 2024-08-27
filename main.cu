#include <cmath>
#include <iostream>
#include "include/tadma/Tadma.hpp"
#include <utility>
#include <string>
#include <string_view>

#include "scripts/generated.cu"

int main() {
    namespace T = tadma;
    T::InitGuard guard;

    Model m("123.");

    constexpr Sequence<10> stuff;

    T::Tensor<long, T::Allocator<T::kCUDA>, T::Sequence<1L, 128L>> input;

    m.infer<1, 128>(input, input, input);

    return 0;
}
