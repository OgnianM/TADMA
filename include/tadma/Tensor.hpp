#pragma once
#include <fstream>
#include <cassert>
#include "Allocator.hpp"
#include "Meta.hpp"
#include "Sequence.hpp"

namespace tadma {

template<typename T, AnyAllocator Allocator, AnySequence Dims_, AnySequence StridesInit = Sequence<>>
struct Tensor {
    static_assert(Dims_::Size > 0, "Tensor must have at least one dimension");

    // Points to the start of the tensor view
    T* view;

    // Points to the start of the allocated memory
    Storage<T> data;

    static constexpr Device device = Allocator::device;

    using Dims = Dims_;
    using AllocatorType = Allocator;
    using ValueType = T;
    static constexpr int Rank = Dims::Size;
    //static constexpr auto DimsArray = Dims::Values;
    static constexpr int Size = Dims::Product();
    static constexpr int SizeBytes = sizeof(T) * Size;

    template<AnySequence State = Sequence<>>
    static consteval auto DefaultStrides() {
        if constexpr (State::Size == (Rank - 1)) {
            return typename State::template Append<1>();
        } else {
            return DefaultStrides<typename State::template Append<Dims::template Last<Dims::Size - State::Size - 1>::Product()> >();
        }
    }

    using Strides = std::conditional_t<StridesInit::Size == 0, decltype(DefaultStrides()), StridesInit>;

    static constexpr int ContiguousSize = constexpr_for<0, Rank>([]<int I>(int acc) { return std::max(acc, Dims::Values(I) * Strides::Values(I)); }, 0);
    static constexpr int ContiguousSizeBytes = sizeof(T) * ContiguousSize;
    static constexpr bool IsContiguous = ContiguousSize == Size;

    // Not sliced and not transposed
    static constexpr bool NormalStrides = Strides() != DefaultStrides();

    static constexpr int HasBroadcastDims = constexpr_for<0, Dims::Size>([]<int I>(bool result) {return result || (Strides::Values(I) == 0 && Dims::Values(I) != 1); }, false);


    //region Construction and Assignment
    Tensor() : data(Allocator::template allocate<T, ContiguousSize>()) {
        view = data.dataptr;
    }

    __multi__ Tensor(const Storage<T>& data_, T* view = nullptr) : data(data_) {
        this->view = view ? view : data.dataptr;
    }

    __multi__ Tensor(const T* data, T* view = nullptr) { // Non-owning
        this->data = Storage<T>(const_cast<T*>(data));
        this->view = view ? view : this->data.dataptr;
    }

    Tensor(const std::array<T, ContiguousSize>& array) : Tensor() {
        tadma::Tensor<T, tadma::Allocator<kCPU>, Sequence<ContiguousSize>> tmp(array.data());
        tmp.copyTo(*this);
    }

    Tensor(std::ifstream& file, unsigned long long offset) : Tensor() {
        file.seekg(offset);

        if constexpr (device == kCPU) {
            file.read((char*)view, SizeBytes);
        } else {
            auto hostData = new uint8_t[SizeBytes];
            file.read((char*)hostData, SizeBytes);
            check_cuda(cudaMemcpyAsync(view, hostData, SizeBytes, cudaMemcpyHostToDevice, stream));
            delete[] hostData;
        }
    }

    __multi__ Tensor(const Tensor& other) noexcept : data(other.data), view(other.view) {}

    __multi__ Tensor(Tensor&& other) noexcept {
        data = std::move(other.data);
        view = other.view;
        other.view = nullptr;
    }

    __multi__ Tensor& operator=(const Tensor& other) noexcept {
        if (this == &other) {
            return *this;
        }
        data = other.data;
        view = other.view;
        return *this;
    }

    __multi__ Tensor& operator=(Tensor&& other) noexcept {
        data = std::move(other.data);
        view = other.view;
        other.view = nullptr;
        return *this;
    }
    // endregion


    Tensor& operator=(const Scalar auto& value) requires(device == kCPU) {
        for (int i = 0; i < Dim(0); i++) {
            this->operator()(i) = value;
        }
        return *this;
    }

    template<int... NewOrder> requires(sizeof...(NewOrder) == Rank && ((NewOrder <= Rank) && ...))
    __multi__ auto transpose() {
        constexpr int Arr[] = { NewOrder... };
        using NewDims = decltype(constexpr_for<0, sizeof...(NewOrder)>([]<int I>(auto dims) {
            return typename TYPE(dims)::template Append<Dims::Values(Arr[I])>();
        }, Sequence<>()));
        using NewStrides = decltype(constexpr_for<0, sizeof...(NewOrder)>([]<int I>(auto dims) {
            return typename TYPE(dims)::template Append<Strides::Values(Arr[I])>();
        }, Sequence<>()));
        return Tensor<T, Allocator, NewDims, NewStrides>(data, view);
    }

    template<typename NewDims> requires(NewDims::Product() == Size || (NewDims::template IndexOf<-1> != -1))
    __multi__ auto reshape(this auto&& self) {
        auto reshape_impl = [&](auto&& self) {
            constexpr auto in1 = NewDims::template IndexOf<-1>;
            if constexpr (in1 != -1) {
                return Tensor<T, Allocator, typename NewDims::template Set<in1, -Size / (NewDims::Product())>>(self.data, self.view);
            } else return Tensor<T, Allocator, NewDims>(self.data, self.view);
        };
        if constexpr (!NormalStrides) {
            // Don't know how to easily do this without cloning
            return reshape_impl(self.clone());
        } else {
            return reshape_impl(self);
        }
    }

    template<int... Dims>
    __multi__ auto reshape(this auto&& self) { return self.template reshape<Sequence<Dims...>>(); }

    template<typename... Is> requires(sizeof...(Is) <= Rank && sizeof...(Is) > 0)
    __multi__ decltype(auto) operator()(this auto&& self, Is... indices) {
        int indicesArray[sizeof...(Is)] = {int (indices)...};

        int offset = constexpr_for<0, sizeof...(Is)>([&]<int I>(int offset) {
            //assert(indicesArray[I] < Dim(I));
            return offset + Strides::Values(I) * indicesArray[I];
        }, 0);

        if constexpr (sizeof...(Is) == Rank) {
            return self.view[offset];
        } else {
            return Tensor<T, Allocator, typename Dims::template Last<Rank - sizeof...(Is)>,
                    typename Strides::template Last<Rank - sizeof...(Is)>> (self.data, self.view + offset);
        }
    }

    template<int Axis> requires(Axis >= 0 && Axis < Rank)
    __multi__ auto index(int i) {
        auto offset = i * Strides::Values(Axis);
        return Tensor<T, Allocator, typename Dims::template Remove<Axis>, typename Strides::template Remove<Axis>>(data, view + offset);
    }

    static consteval int Dim(int i) { return Dims::Values(i); }
    static consteval int Stride(int i) { return Strides::Values(i); }

    template<int Dim = 0, int Start = 0, int End = -1, int Step = 1>
    constexpr auto slice() {
        constexpr int DimIndex = Dim < 0 ? Rank + Dim : Dim;
        constexpr int DimSize = Dims::Values(DimIndex);
        constexpr int StartIndex = Start < 0 ? DimSize + Start : Start;
        constexpr int EndIndex = End < 0 ? DimSize + End + 1 : End;

        static_assert(StartIndex >= 0 and StartIndex < DimSize, "Start index out of bounds");
        static_assert(EndIndex >= 0 and EndIndex <= DimSize, "End index out of bounds");
        static_assert(StartIndex <= EndIndex || Step < 0, "Start index must be less than or equal to end index");

        constexpr int Span = std::max(0, (EndIndex - StartIndex - (Step < 0)) / Step);

        using NewDims = typename Dims::template Set<DimIndex, Span>;
        using NewStrides = typename Strides::template Set<DimIndex, Strides::Values(DimIndex) * Step>;

        return Tensor<T, Allocator, NewDims, NewStrides>(data, view + Strides::Values(DimIndex) * StartIndex);
    }

    template<int Dim = -1> requires(Dims::Values(Dim < 0 ? Rank + Dim : Dim) == 1)
    constexpr auto squeeze() {
        constexpr int DimIndex = Dim < 0 ? Rank + Dim : Dim;
        return Tensor<T, Allocator, typename Dims::template Remove<DimIndex>, typename Strides::template Remove<DimIndex>>(data, view);
    }

    template<int Dim = -1, int Broadcast = 1> requires((Dim >= 0 && Dim <= Rank) || (Dim < 0 && Rank + Dim + 1 <= Rank))
    constexpr auto unsqueeze(this auto&& self) {
        constexpr int DimIndex = Dim < 0 ? Rank + Dim + 1 : Dim;
        if constexpr (DimIndex == Rank) {
            return Tensor<T, Allocator, typename Dims::template Append<Broadcast>, typename Strides::template Append<0>>(self.data, self.view);
        } else {
            return Tensor<T, Allocator, typename Dims::template Insert<DimIndex, Broadcast>, typename Strides::template Insert<DimIndex, 0>>(
                    self.data, self.view);
        }
    }

    template<int Dim, int Size> requires(Dims::Values(Dim) == 1)
    constexpr auto broadcast(this auto&& self) {
        return Tensor<T, Allocator, typename Dims::template Set<Dim, Size>, typename Strides::template Set<Dim, 0>>(self.data, self.view);
    }

    template<typename U> requires(Dims() == typename U::Dims() && Strides() == typename U::Strides())
    void copyTo(U& other) const {
        if constexpr (device == kCPU && U::device == kCPU) {
            memcpy(other.view, view, ContiguousSizeBytes);
        } else if constexpr (device == kCPU && U::device == kCUDA) {
            check_cuda(cudaMemcpyAsync(other.view, view, ContiguousSizeBytes, cudaMemcpyHostToDevice, stream));
        } else if constexpr (device == kCUDA && U::device == kCPU) {
            check_cuda(cudaMemcpyAsync(other.view, view, ContiguousSizeBytes, cudaMemcpyDeviceToHost, stream));
        } else if constexpr (device == kCUDA && U::device == kCUDA) {
            check_cuda(cudaMemcpyAsync(other.view, view, ContiguousSizeBytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    template<typename U> requires(Dims() == typename U::Dims() && Strides() != typename U::Strides())
    void copyTo(U& other) const { CombineVariadicNode([] __multi__ (auto& dst, const auto& src) { dst = src; }, other, *this); }

    template<typename U> requires(ContiguousSizeBytes == U::ContiguousSizeBytes)
    __device__ void copyTo(U& other) const {
        memcpy(other.view, view, ContiguousSizeBytes);
    }

    auto clone() const {
        Tensor<T, Allocator, Dims> result;
        copyTo(result);
        return result;
    }

    template<Device newDevice> auto to() const {
        if constexpr (newDevice == device) {
            return *this;
        }
        Tensor<T, tadma::Allocator<newDevice>, Dims> newTensor;
        copyTo(newTensor);
        return newTensor;
    }

    template<typename U> auto to() const {
        if constexpr (std::is_same_v<ValueType, U>) return *this;
        else return EltwiseNode(*this, [] __multi__ (const T& x) { return (U)x; });
    }

    auto contiguous() const requires(device == kCUDA) {
        if constexpr (IsContiguous) return *this;
        else return EltwiseNode(*this, [&] __multi__ (const T& x) { return x; });
    }

    __multi__ decltype(auto) operator[](int index) {
        if constexpr (IsContiguous) {
            return (T&)view[index];
        } else {
            int final = 0;

            constexpr_rfor<0, Rank>([&]<int I>() {
                final += (index % Dim(I)) * Stride(I);
                index /= Dim(I);
            });
            return (const T&)view[final];
        }
    }

    /// @brief Dimensions with size=1 or stride=0 are considered 'fake' dimensions
    auto removeFakeDims(this auto&& self) {
        return Tensor<T, Allocator, decltype(RemoveFakeDimsImpl_t::first), decltype(RemoveFakeDimsImpl_t::second)>(self.data, self.view);
    }

    /// @brief Returns a tensor with a minimized rank, which is defined by the number of distinct discontiguities in the dimension strides
    /// For example, a contiguous tensor will have rank=1 and stride=1 after this operation
    auto minimumRank() {
        return removeFakeDims().removeContiguousDims();
    }

    template<typename U>
    bool operator==(const U& other) const {
        return Strides() == U::Strides() && Dims() == U::Dims() && view == other.view;
    }

    template<int D = Rank-1, int ExpectedStride = 1, typename NewDims = Sequence<>, typename NewStrides = Sequence<>>
    auto removeContiguousDims() {
        if constexpr (IsContiguous) {
            return Tensor<T, Allocator, Sequence<Size>>(data, view);
        } else {
            if constexpr (D == -1) {
                return Tensor<T, Allocator, NewDims, NewStrides>(data, view);
            } else if constexpr (Strides::Values(D) == ExpectedStride) {
                return removeContiguousDims<D - 1, ExpectedStride * Dims::Values(D), NewDims, NewStrides>();
            } else {
                return removeContiguousDims<D - 1, typename NewDims::template Prepend<Dims::Values(D)>,
                        typename NewStrides::template Prepend<Strides::Values(D)>>();
            }
        }
    }
private:
    using RemoveFakeDimsImpl_t = decltype(constexpr_for<0, Rank>([]<int I>(auto dims) {
        if constexpr (Dim(I) == 1 || Stride(I) == 0) return dims;
        else return std::pair { typename decltype(dims.first)::template Append<Dim(I)>(), typename decltype(dims.second)::template Append<Stride(I)>() };
    }, std::pair<Sequence<>, Sequence<>>()));

public:
    Tensor& operator=(const Scalar auto& value) {
        InplaceNode(*this, [value] __multi__ (const ValueType& x) { return value; }); return *this;
    }

    // Reductions
    template<int D, typename Post = decltype(ForwardFunction)> auto reduce(auto&& f, Post postprocess = ForwardFunction) {
        return ReduceNode<D < 0 ? Rank + D : D>(*this, f, postprocess);
    }
    template<int D> decltype(auto) sum() { return reduce<D>([] __multi__ (const auto& a, const auto& b) { return a + b; }, ForwardFunction); }
    template<int D> decltype(auto) min() { return reduce<D>([] __multi__ (const auto& a, const auto& b) { return std::min(a, b); }); }
    template<int D> decltype(auto) max() { return reduce<D>([] __multi__ (const auto& a, const auto& b) { return std::max(a, b); }); }
    template<int D> decltype(auto) mean() { return reduce<D>([] __multi__ (const auto& a, const auto& b) { return a + b; }, []__multi__(const auto& x) { return x / Dim(D); }); }

    // Every dimension != 1 in the source must exist in the target, and they must be in the same order, otherwise the broadcast fails
    template<AnySequence TargetDims, typename Self>
    requires (std::is_same_v<TargetDims, Dims> || (Rank == 1 && Dim(0) == 1) || TargetDims:: template ContainsOrderedSubset<decltype(RemoveFakeDimsImpl_t::first)>())
    constexpr auto broadcastTo(this Self&& self) {
        if constexpr (std::is_same_v<TargetDims, Dims>) {
            return self;
        } else if constexpr(Rank == 1 && Dim(0) == 1) { // Scalar tensor can be broadcasted to any size
            using Strides = decltype(constexpr_for<0, TargetDims::Size>([]<int I>(auto strides) {
                return typename decltype(strides)::template Append<0>();
            }, Sequence<>()));
            return Tensor<T, Allocator, TargetDims, Strides>(self.data, self.view);
        } else {
            auto clean = self.removeFakeDims();
            // Now we have dimensions like [42, 69, 113] and we must ensure they exist in the target
            using CleanDims = typename decltype(clean)::Dims;

            return constexpr_for<0, TargetDims::Size>([]<int I>(auto tensor) {
                if constexpr(I == decltype(tensor)::Rank) {
                    return tensor.template unsqueeze<-1>().template broadcast<I, TargetDims::template Values<I>()>();
                } else if constexpr(decltype(tensor)::Dims::template Values<I>() != TargetDims::template Values<I>()) {
                    return tensor.template unsqueeze<I>().template broadcast<I, TargetDims::template Values<I>()>();
                } else return tensor;
            }, clean);
        }
    }

    template<AnyTensor U> auto broadcastTo(const U& other) { return broadcastTo<typename U::Dims>(); }
    template<AnyTensor U> auto broadcastTo(const U& other) const { return broadcastTo<typename U::Dims>(); }
};

template<tadma::AnyTensor T> requires(T::device == tadma::kCPU)
std::ostream& operator<<(std::ostream& os, const T& tensor) {
    os << "[";
    for (int i = 0; i < T::Dim(0); i++) {
        os << tensor(i) << ", ";
    }
    os << "]\n";
    return os;
}

template<tadma::AnyTensor T> requires(T::device == tadma::kCUDA)
std::ostream& operator<<(std::ostream& os, T tensor) {
    auto tmp = tensor.contiguous().template to<tadma::kCPU>();
    check_cuda(cudaStreamSynchronize(tadma::stream));
    return os << tmp;
}

using NoTensor = Tensor<bool, Allocator<kCPU>, Sequence<0>>;

template<AnyTensor T> using TensorLike = Tensor<typename T::ValueType, typename T::AllocatorType, typename T::Dims>;

template<typename T, typename Dims, typename Strides = Sequence<>> using CpuTensor = Tensor<T, Allocator<kCPU>, Dims, Strides>;
template<typename T, typename Dims, typename Strides = Sequence<>> using CudaTensor = Tensor<T, Allocator<kCUDA>, Dims, Strides>;

template<AnyTensor T> using BaseTensor = Tensor<typename T::ValueType, typename T::AllocatorType, typename T::Dims, Sequence<>>;

};
