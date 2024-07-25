#pragma once
#include <functional>
#include <atomic>
#include "Defines.hpp"


namespace tadma {

struct StorageImpl {
    std::atomic<int> refcount;
    std::function<void()> deleter;
    StorageImpl(int refcount, std::function<void()> deleter) : refcount(refcount), deleter(std::move(deleter)) {}
};

/// @brief Refcounted ptr on host side, raw ptr on device side
template<typename T>
struct Storage {
        T* dataptr;
#ifndef  __CUDA_ARCH__
        StorageImpl* impl;
#endif

    __multi__ Storage() {
        dataptr = nullptr;
#ifndef  __CUDA_ARCH__
        impl = nullptr;
#endif
    }

    __multi__ Storage(T* dataptr, std::function<void()> deleter) {
        this->dataptr = dataptr;
#ifndef  __CUDA_ARCH__
        this->impl = new StorageImpl(1, std::move(deleter));
#endif
    }

    __multi__ Storage(T* dataptr) {
        this->dataptr = dataptr;
#ifndef  __CUDA_ARCH__
        impl = nullptr;
#endif
    }

    __multi__ Storage(const Storage& other) {
        dataptr = other.dataptr;
#ifndef  __CUDA_ARCH__
        impl = other.impl;
        impl->refcount++;
#endif
    }

    __multi__ Storage(Storage&& other) {
        dataptr = other.dataptr;
        other.dataptr = nullptr;
#ifndef  __CUDA_ARCH__
        impl = other.impl;
        other.impl = nullptr;
#endif
    }


    __multi__ Storage& operator=(const Storage& other) {
#ifndef  __CUDA_ARCH__
        if (this == &other) {
            return *this;
        }
        destroy();
        dataptr = other.dataptr;
        impl = other.impl;
        impl->refcount++;
#else
        //assert(false);
#endif
        return *this;
    }

    __multi__ Storage& operator=(Storage&& other) noexcept {
#ifndef  __CUDA_ARCH__
        if (this == &other) {
            return *this;
        }
        destroy();
        impl = other.impl;
        dataptr = other.dataptr;
        other.dataptr = nullptr;
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

    __multi__ ~Storage() noexcept {
#ifndef  __CUDA_ARCH__
        destroy();
#endif
    }
};

};
