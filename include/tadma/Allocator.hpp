#pragma once
#include "Storage.hpp"
#include "Globals.hpp"
#include "Concepts.hpp"

namespace tadma {

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

template<> struct Allocator<kTrampoline> : DeviceBound<kTrampoline> {
    template<typename T, int64_t Count>
    static TrampolineStorage<T> allocate() {
        return {};
    }
};



template<typename T, AnyAllocator Allocator, AnySequence Dims_, AnySequence StridesInit> struct Tensor;

template<AnyAllocator Allocator, AnyAllocator... Allocators> requires ((Allocator::device == Allocators::device) && ...)
using CommonAllocator = Allocator;

};

