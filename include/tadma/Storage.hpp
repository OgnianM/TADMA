#pragma once
#include <functional>
#include <atomic>
#include "Defines.hpp"
#include "Sequence.hpp"

namespace tadma {

struct StorageImpl {
    std::atomic<int> refcount;
    std::function<void()> deleter;
    StorageImpl(int refcount, std::function<void()> deleter) : refcount(refcount), deleter(std::move(deleter)) {}
};

/// @brief Refcounted ptr on host side, raw ptr on device side
template<typename T>
struct HeapStorage {
        T* view;

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
    __multi__ HeapStorage operator+(int64_t offset) {
        HeapStorage new_storage = *this;
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

template<typename T, int64_t N>
struct LocalStorage {
    T data[N];
    int64_t offset;

    // Storage must be able to act as a pointer
    __multi__ LocalStorage operator+(int64_t offset) const {
        LocalStorage new_storage = *this;
        new_storage.offset += offset;
        return new_storage;
    }

    __multi__ LocalStorage& operator+=(int64_t offset) {
        this->offset += offset;
        return *this;
    }

    __multi__ decltype(auto) operator[](this auto&& self, int64_t index) {
        return self.data[index + self.offset];
    }
};

template<AnySequence Seq>
struct ConstexprStorage {
    using Sequence = Seq;
    static constexpr auto array = Seq::Array();

    struct OffsetableConstexprStorage {
        int64_t offset;
        OffsetableConstexprStorage(int64_t offset = 0) : offset(offset) {}

        // Storage must be able to act as a pointer
        __multi__ OffsetableConstexprStorage operator+(int64_t offset) const {
            return OffsetableConstexprStorage(this->offset + offset);
        }

        __multi__ OffsetableConstexprStorage& operator+=(int64_t offset) {
            this->offset += offset;
            return *this;
        }

        __multi__ constexpr auto operator[](int64_t index) const {
            return array[index + offset];
        }
    };

    // Storage must be able to act as a pointer
    __multi__ OffsetableConstexprStorage operator+(int64_t offset) const {
        return OffsetableConstexprStorage(offset);
    }


    template<int64_t offset>
    constexpr auto cexpr_offset() {
        return typename Seq::template Last<Seq::Size - offset>();
    }

    __multi__ constexpr auto operator[](int64_t index) const {
        return array[index];
    }
};

};
