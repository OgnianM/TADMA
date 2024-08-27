#pragma once
#include "Storage.hpp"
#include "Globals.hpp"

namespace tadma {
enum Memory { kCPU, kCUDA, kConstexpr, kLocal };

template<Memory device> struct Allocator;

template<Memory device_> struct DeviceBound {
    static constexpr Memory device = device_;
};

template<typename T> consteval Memory deviceof(const T&) {
    return T::device;
}

template<> struct Allocator<kCUDA> : DeviceBound<kCUDA> {
    template<typename T, int64_t Count>
    static HeapStorage<T> allocate() {
        T* ptr;
        check_cuda(cudaMallocAsync(&ptr, sizeof(T) * Count, stream));
        return HeapStorage(ptr, [ptr]() {
            cudaFreeAsync(ptr, stream);
        });
    }
};

template<> struct Allocator<kCPU> : DeviceBound<kCPU>{
    template<typename T, int64_t Count>
    static HeapStorage<T> allocate() {
        T* ptr = new T[Count];
        return HeapStorage(ptr, [ptr]() {
            delete[] ptr;
        });
    }
};

template<typename Sequence_>
struct ConstexprAllocator : DeviceBound<kConstexpr> {
    using Sequence = Sequence_;

    template<typename T, int64_t Count> requires(Count < Sequence::Size)
    static constexpr auto allocate() {
        return typename ConstexprStorage<Sequence>::OffsetableConstexprStorage();
    }

    template<typename T, int64_t Count> requires(Count == Sequence::Size)
    static constexpr auto allocate() {
        return ConstexprStorage<Sequence>();
    }
};


template<typename T> concept AnyAllocator = true;
template<typename T> concept AnyCudaTensor = AnyTensor<T> && (std::decay_t<T>::device == Memory::kCUDA);
template<typename T> concept AnyHostTensor = AnyTensor<T> && (std::decay_t<T>::device == Memory::kCPU);
template<typename T> concept AnyConstexprTensor = AnyTensor<T> && (std::decay_t<T>::device == Memory::kConstexpr);
template<typename T, AnyAllocator Allocator, AnySequence Dims_, AnySequence StridesInit> struct Tensor;

};

