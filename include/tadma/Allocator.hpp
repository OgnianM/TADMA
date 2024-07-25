#pragma once
#include "Storage.hpp"
#include "Globals.hpp"

namespace tadma {
enum Device { kCPU, kCUDA };

template<Device device> struct Allocator { };

template<Device device_> struct DeviceBound {
    static constexpr Device device = device_;
};

template<typename T> consteval Device deviceof(const T&) {
    return T::device;
}

template<> struct Allocator<kCUDA> : DeviceBound<kCUDA> {
    template<typename T, int Count>
    static Storage<T> allocate() {
        T* ptr;
        check_cuda(cudaMallocAsync(&ptr, sizeof(T) * Count, stream));
        return Storage(ptr, [ptr]() {
            cudaFreeAsync(ptr, stream);
        });
    }
};

template<> struct Allocator<kCPU> : DeviceBound<kCPU>{
    template<typename T, int Count>
    static Storage<T> allocate() {
        T* ptr = new T[Count];
        return Storage(ptr, [ptr]() {
            delete[] ptr;
        });
    }
};
};

