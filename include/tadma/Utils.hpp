#pragma once
#include "Concepts.hpp"
#include <cstdint>
#include <array>
#include <cxxabi.h>

namespace tadma {

    template<AnySequence Shape> requires (Shape::Size > 0)
    static constexpr std::array<int64_t, Shape::Size> MapToShape(int64_t index) {
        std::array<int64_t, Shape::Size> result;
        constexpr_rfor<0, Shape::Size>([&]<int64_t I>() {
            result[I] = index % Shape::Values(I);
            index /= Shape::Values(I);
        });
        return result;
    }

    template<AnySequence Shape0, AnySequence Shape1> requires(Shape0::Product() == Shape1::Product() && Shape0::Size > 0)
    static constexpr std::array<int64_t, Shape1::Size> ShapeToShape(const std::array<int64_t, Shape0::Size>& indices) {
        int64_t index = 0;
        constexpr_for<0, Shape0::Size>([&]<int I>(int64_t scale) {
            index += indices[I] * scale;
            return scale * Shape0::Values(I);
        }, 1);
        return MapToShape<Shape1>(index);
    }

    inline std::string demangle(const char* name) {
        int status;
        char* demangled = abi::__cxa_demangle(name, 0, 0, &status);
        std::string result(demangled);
        free(demangled);
        return result;
    }

    template<typename T> std::string typename_() {
        return demangle(typeid(T).name());
    }


    template<int Index>
    constexpr auto tuple_replace(auto& tuple, auto&& new_value) {
        return constexpr_for<0, std::tuple_size_v<TYPE(tuple)>>([&]<int I>(auto new_tuple) {
            if constexpr (I == Index) {
                return std::tuple_cat(new_tuple, std::tuple {new_value});
            } else {
                return std::tuple_cat(new_tuple, std::tuple {std::get<I>(tuple)});
            }
        }, std::make_tuple());
    }

};