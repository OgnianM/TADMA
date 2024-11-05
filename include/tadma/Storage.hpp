#pragma once
#include <functional>
#include <atomic>
#include "Defines.hpp"
#include "Concepts.hpp"

namespace tadma {

struct StorageImpl {
    std::atomic<int> refcount;
    std::function<void()> deleter;
    StorageImpl(int refcount, std::function<void()> deleter) : refcount(refcount), deleter(std::move(deleter)) {}
};

/// @brief Refcounted ptr on host side, raw ptr on device side
template<typename T>
struct HeapStorage {
        T* __restrict__ view;

#ifndef  __CUDA_ARCH__
        StorageImpl* impl;
#endif

    __multi__ HeapStorage() {
        view = nullptr;
#ifndef  __CUDA_ARCH__
        impl = nullptr;
#endif
    }

    __multi__ HeapStorage(T* dataptr, std::function<void()> deleter) {
        this->view = dataptr;
#ifndef  __CUDA_ARCH__
        this->impl = new StorageImpl(1, std::move(deleter));
#endif
    }

    __multi__ HeapStorage(T* dataptr) {
        this->view = dataptr;
#ifndef  __CUDA_ARCH__
        impl = nullptr;
#endif
    }

    __multi__ HeapStorage(const HeapStorage& other) {
        view = other.view;
#ifndef  __CUDA_ARCH__
        impl = other.impl;
        impl->refcount++;
#endif
    }

    __multi__ HeapStorage(HeapStorage&& other) {
        view = other.view;
        other.view = nullptr;
#ifndef  __CUDA_ARCH__
        impl = other.impl;
        other.impl = nullptr;
#endif
    }


    __multi__ HeapStorage& operator=(const HeapStorage& other) {
#ifndef  __CUDA_ARCH__
        if (this == &other) {
            return *this;
        }
        destroy();
        view = other.view;
        impl = other.impl;
        impl->refcount++;
#else
        //assert(false);
#endif
        return *this;
    }

    __multi__ HeapStorage& operator=(HeapStorage&& other) noexcept {
#ifndef  __CUDA_ARCH__
        if (this == &other) {
            return *this;
        }
        destroy();
        impl = other.impl;
        view = other.view;
        other.view = nullptr;
        other.impl = nullptr;
#else
        //assert(false);
#endif
        return *this;
    }

    __multi__ void destroy() noexcept {
#ifndef  __CUDA_ARCH__
        if (impl) {
            if (--impl->refcount == 0) {
                impl->deleter();
                delete impl;
            }
        }
#endif
    }

    __multi__ ~HeapStorage() noexcept {
#ifndef  __CUDA_ARCH__
        destroy();
#endif
    }


    // Storage must be able to act as a pointer
    __multi__ HeapStorage operator+(this auto&& self, int64_t offset) {
        HeapStorage new_storage = self;
        new_storage.view += offset;
        return new_storage;
    }

    __multi__ HeapStorage& operator+=(int64_t offset) {
        this->view += offset;
        return *this;
    }

    __multi__ decltype(auto) operator[](this auto&& self, int64_t index) {
        return self.view[index];
    }
};


// Similar to broadcasting a scalar tensor, every index refers to the same element, stored on the stack
template<typename T>
struct TrampolineStorage {
    T view;

    TrampolineStorage() = default;

    __multi__ auto& operator[](this auto&& self, auto) {
        return self.view;
    }

    __multi__ decltype(auto) operator+(this auto&& self, auto) {
        return self;
    }

    __multi__ decltype(auto) operator+=(this auto&& self, auto) {
        return self;
    }
};

};
