#pragma once
#include <type_traits>

namespace tadma {
enum Memory {
    kCPU,
    kCUDA,
    kTrampoline // Special sauce for kernel-internal tensors
};

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

template<auto... Xs> concept HaveCommonType = requires { std::common_type_t<decltype(Xs)...>(); };

template<typename T, typename... Ts> concept SameDevice = ((std::decay_t<T>::device == std::decay_t<Ts>::device) && ...);
template<typename T, typename... Ts> concept SameDims = ((typename T::Dims() == typename Ts::Dims()) && ...);
template<typename T, typename... Ts> concept SameStrides = ((T::Strides() == Ts::Strides()) && ...);
template<typename T, typename... Ts> concept SameType = (std::is_same_v<typename T::ValueType, typename Ts::ValueType> && ...);
template<typename T, typename... Ts> concept SameRank = ((T::Rank == Ts::Rank) && ...);


template<typename T> concept AnyAllocator = true;
template<typename T> concept AnyCudaTensor = AnyTensor<T> && (std::decay_t<T>::device == Memory::kCUDA);
template<typename T> concept AnyHostTensor = AnyTensor<T> && (std::decay_t<T>::device == Memory::kCPU);

template<typename... Ts>
consteval Memory CommonDevice() {
    constexpr int Size = sizeof...(Ts);
    std::array<Memory, Size> devices = {Ts::device...};

    Memory device = kCUDA;
    int i = 0;
    for (; i < Size; i++) {
        if (devices[i] == kCPU || devices[i] == kCUDA) {
            device = devices[i];
            break;
        }
    }

    for (i = 0; i < Size; i++) {
        assert(devices[i] == device || devices[i] == kTrampoline);
    }

    return device;
}


} // namespace tadma