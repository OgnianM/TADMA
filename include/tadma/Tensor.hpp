#pragma once
#include <fstream>
#include <cassert>
#include "Allocator.hpp"
#include "Meta.hpp"
#include "Sequence.hpp"
#include "Tagging.hpp"
#include "Utils.hpp"
#include "Tagging.hpp"

namespace tadma {

/**
 * @brief Tensor class with static dimensions and strides
 * @tparam T Dataype of the tensor
 * @tparam Allocator Allocator to use for memory management
 * @tparam Dims_ Dimensions of the tensor, an empty sequence represents a scalar tensor
 * @tparam StridesInit Strides of the tensor, an empty sequence represents default strides
 */
template<typename T, AnyAllocator Allocator_, AnySequence Dims_, AnySequence StridesInit = Sequence<>>
struct Tensor {
    using Allocator = Allocator_;
    static constexpr Memory device = Allocator::device;
    using ValueType = T;

    using Dims = Dims_;
	static constexpr int64_t Rank = Dims::Size;
    static constexpr int64_t Size = Rank == 0 ? 1 : Dims::Product();
    static constexpr int64_t SizeBytes = sizeof(T) * Size;

    template<AnySequence State = Sequence<>>
    static consteval auto DefaultStrides() {
        if constexpr (Rank == 0) return State();
        else if constexpr (State::Size == (Rank - 1)) {
            return typename State::template Append<1>();
        } else {
            return DefaultStrides<typename State::template Append<Dims::template Last<Dims::Size - State::Size - 1>::Product()> >();
        }
    }
    using Strides = std::conditional_t<StridesInit::Size == 0, decltype(DefaultStrides()), StridesInit>;

private:
    // max(Dims[i] * Strides[i])
    static constexpr auto MaxDimSize = constexpr_for<0, Rank>([]<int64_t I>(int64_t acc) {
        return std::max<int64_t>(acc, Dims::template Values<I>() * Strides::template Values<I>());
    }, 0);
public:
    static constexpr int64_t ContiguousSize = Rank == 0 || MaxDimSize == 0 ? 1 : MaxDimSize;
    static constexpr int64_t ContiguousSizeBytes = sizeof(T) * ContiguousSize;
    static constexpr bool IsContiguous = ContiguousSize == Size;

    // Not a slice and not transposed
    static constexpr bool NormalStrides = Strides() == DefaultStrides();

    static constexpr int64_t HasBroadcastDims = constexpr_for<0, Dims::Size>([]<int I>(bool result) {
        return result || (Strides::Values(I) == 0 && Dims::Values(I) != 1);
    }, false);

    static constexpr bool IsScalar = Size == 1;

    using StorageType = decltype(Allocator::template allocate<T, ContiguousSize>());
    StorageType data;

    static consteval auto Dim(int i) { return Dims::Values(i); }
    static consteval auto Stride(int i) { return Strides::Values(i); }


    //region Construction and Assignment
    constexpr Tensor() : data(Allocator::template allocate<T, ContiguousSize>()) {
        static_assert(!IsTagged<Allocator>, "Allocating a new tensor with an already tagged allocator, this is probably a mistake.");
    }
    constexpr __multi__ Tensor(const StorageType& data_) : data(data_) { }

    __multi__ Tensor(const T* data) requires(std::is_same_v<StorageType, HeapStorage<T>>) {// Non-owning
        this->data = HeapStorage<T>(const_cast<T*>(data));
    }

    __multi__ Tensor(void* data) : Tensor((T*)data) {} // Non-owning

    T* disown_data() {
        if (data.impl) data.impl->refcount = 0;
        return data.dataptr;
    }


    Tensor(std::ifstream& file, uint64_t offset) : Tensor() {
        file.seekg(offset);

        if constexpr (device == kCPU) {
            file.read((char*)&data[0], SizeBytes);
        } else {
            auto hostData = new uint8_t[SizeBytes];
            file.read((char*)hostData, SizeBytes);
            // Doing this asynchronously because the allocator is asynchronous
            check_cuda(cudaMemcpyAsync(&data[0], hostData, SizeBytes, cudaMemcpyHostToDevice, stream));
            check_cuda(cudaStreamSynchronize(stream));
            delete[] hostData;
        }
    }

    __multi__ Tensor(const Tensor& other) noexcept : data(other.data) {}

    __multi__ Tensor(Tensor&& other) noexcept {
        data = std::move(other.data);
    }

    /*
    __multi__ Tensor& operator=(const Tensor& other) noexcept {
        if (this == &other) {
            return *this;
        }
        data = other.data;
        return *this;
    }
    */

    __multi__ Tensor& operator=(Tensor&& other) noexcept {
        std::swap(data, other.data);
        return *this;
    }

    template<AnyTensor Other> requires(Dims() == typename Other::Dims())
    __multi__ Tensor& operator=(this auto&& self, const Other& other) {
        for (int64_t i = 0; i < Size; i++) {
            self[i] = other[i];
        }
    }

    // endregion

    template<int64_t... Dims>
    __multi__ auto reshape(this auto&& self) {
        return self.template reshape<Sequence<Dims...>>();
    }

    template<Scalar IndexType, unsigned long Size>
    __multi__ constexpr decltype(auto) operator()(this auto&& self, const std::array<IndexType, Size>& indicesArray) {
        if constexpr (Size == 0) {
            return self;
        } else {
            auto offset = constexpr_for<0, Size>([&]<IndexType I>(IndexType offset) {
                ASSERT(indicesArray[I] < Dim(I));
                return offset + Strides::Values(I) * indicesArray[I];
            }, 0);

            if constexpr (Size == Rank) {
                return self.data[offset];
            } else {
                return Tensor<T, RemoveTag_t<Allocator>, typename Dims::template Last<Rank - Size>,
                        typename Strides::template Last<Rank - Size>> (self.data + offset);
            }
        }
    }

    template<Scalar... Is> requires(sizeof...(Is) <= Rank)
    __multi__ constexpr decltype(auto) operator()(this auto&& self, Is... indices) {
        return self(std::array<std::common_type_t<Is...>, sizeof...(indices)>{indices...});
    }

    __multi__ constexpr decltype(auto) operator[](this auto&& self, int64_t index) {
        if constexpr (NormalStrides) {
            ASSERT(index < Size);
            return self.data[index];
        } else {
            int64_t final = 0;

            constexpr_rfor<0, Rank>([&]<int64_t I>() {
                final += (index % Dim(I)) * Stride(I);
                index /= Dim(I);
            });
            ASSERT(final < Size);

            // If there are broadcast dims the reference is const as there may be multiple references to the same memory location
            if constexpr (HasBroadcastDims) {
                return const_cast<const TYPE(self)&>(self).data[final];
            } else {
                return self.data[final];
            }
        }
    }


    template<int Axis> requires(Axis >= 0 && Axis < Rank)
    __multi__ constexpr decltype(auto) index(this auto&& self, auto&& i) {
        return Tensor<T, RemoveTag_t<Allocator>, typename Dims::template Remove<Axis>, typename Strides::template Remove<Axis>>(
            self.data + i * Strides::Values(Axis));
    }

    template<typename U> requires(Dims() == typename U::Dims() && Strides() == typename U::Strides())
    void copyTo(U& other) const {
        if constexpr (device == kCPU && U::device == kCPU) {
            memcpy(&other[0], &data[0], ContiguousSizeBytes);
        } else if constexpr (device == kCPU && U::device == kCUDA) {
            check_cuda(cudaMemcpyAsync(&other[0], &data[0], ContiguousSizeBytes, cudaMemcpyHostToDevice, stream));
        } else if constexpr (device == kCUDA && U::device == kCPU) {
            check_cuda(cudaMemcpyAsync(&other[0], &data[0], ContiguousSizeBytes, cudaMemcpyDeviceToHost, stream));
        } else if constexpr (device == kCUDA && U::device == kCUDA) {
            check_cuda(cudaMemcpyAsync(&other[0], &data[0], ContiguousSizeBytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    template<typename U> requires(Dims() == typename U::Dims() && Strides() != typename U::Strides())
    void copyTo(U& other) const { CombineVariadicNode([] __multi__ (auto& dst, const auto& src) { dst = src; }, other, *this); }

    template<typename U> requires(NormalStrides && ContiguousSizeBytes == std::decay_t<U>::ContiguousSizeBytes)
    __device__ void copyTo(U&& other) const {
        memcpy(other.view, &data[0], ContiguousSizeBytes);
    }

    auto clone() const {
        Tensor<T, RemoveTag_t<Allocator>, Dims> result;
        copyTo(result);
        return result;
    }

    template<Memory newDevice> auto to() const {
        if constexpr (newDevice == device) {
            return *this;
        }
        Tensor<T, tadma::Allocator<newDevice>, Dims> newTensor;
        copyTo(newTensor);
        return newTensor;
    }

    template<typename U> auto to() const {
        if constexpr (std::is_same_v<ValueType, U>) return *this;
        else return EltwiseNode(*this, [] __multi__ (const T& x) ->U { return x; });
    }

    auto contiguous() const {
        if constexpr (NormalStrides) return *this;
        else return EltwiseNode(*this, [&] __multi__ (const T& x) { return x; });
    }

    /// @brief Returns the tensor dimension indices corresponding to data[index] as an array of int64_t
    static constexpr std::array<int64_t, Rank> memoryIndexToTensorIndex(int64_t index) {
        return MapToShape<Dims>(index);
    }


    /// @brief Dimensions with size=1 or stride=0 are considered 'fake' dimensions
    auto removeFakeDims(this auto&& self) {
        using RemoveFakeDimsImpl_t = decltype(constexpr_for<0, Rank>([]<int I>(auto dims) {
            if constexpr (Dim(I) == 1 || Stride(I) == 0) return dims;
            else return std::pair { typename decltype(dims.first)::template Append<Dim(I)>(), typename decltype(dims.second)::template Append<Stride(I)>() };
        }, std::pair<Sequence<>, Sequence<>>()));

        return Tensor<T, Allocator, decltype(RemoveFakeDimsImpl_t::first), decltype(RemoveFakeDimsImpl_t::second)>(self.data);
    }



    template<int D = Rank-1, int ExpectedStride = 1, typename NewDims = Sequence<>, typename NewStrides = Sequence<>>
    auto removeContiguousDims() {
        if constexpr (IsContiguous) {
            return Tensor<T, Allocator, Sequence<Size>>(data);
        } else {
            if constexpr (D == -1) {
                return Tensor<T, Allocator, NewDims, NewStrides>(data);
            } else if constexpr (Strides::Values(D) == ExpectedStride) {
                return removeContiguousDims<D - 1, ExpectedStride * Dims::Values(D), NewDims, NewStrides>();
            } else {
                return removeContiguousDims<D - 1, typename NewDims::template Prepend<Dims::Values(D)>,
                        typename NewStrides::template Prepend<Strides::Values(D)>>();
            }
        }
    }

    /// @brief Returns a tensor with a minimized rank, which is defined by the number of distinct discontiguities in the dimension strides
    /// For example, a contiguous tensor will have rank=1 and stride=1 after this operation
    auto minimumRank() {
        return removeFakeDims().removeContiguousDims();
    }

private:
    using RemoveFakeDimsImpl_t = decltype(constexpr_for<0, Rank>([]<int I>(auto dims) {
    	if constexpr (IsScalar) return dims;
        else if constexpr (Dim(I) == 1 || Stride(I) == 0) return dims;
        else return std::pair { typename decltype(dims.first)::template Append<Dim(I)>(), typename decltype(dims.second)::template Append<Stride(I)>() };
    }, std::pair<Sequence<>, Sequence<>>()));

public:

    Tensor& operator=(const Scalar auto& value) {
        InplaceNode(*this, [value] __multi__ (const ValueType& x) { return value; }); return *this;
    }

    __multi__ Tensor& operator=(const ValueType& value) {
         for (int i = 0; i < Size; i++) {
            (*this)[i] = value;
         }
    }


    template<int Start, int End, int Axis, int Step = 1>
    constexpr auto slice() {
        constexpr int DimIndex = Axis < 0 ? Rank + Axis : Axis;
        constexpr int DimSize = Dims::Values(DimIndex);
        constexpr int StartIndex = Start < 0 ? DimSize + Start : Start;
        constexpr int EndIndex = End < 0 ? DimSize + End + 1 : End;

        static_assert(StartIndex >= 0 and StartIndex < DimSize, "Start index out of bounds");
        static_assert(EndIndex >= 0 and EndIndex <= DimSize, "End index out of bounds");
        static_assert(StartIndex <= EndIndex || Step < 0, "Start index must be less than or equal to end index");

        constexpr int Span = std::max(0, (EndIndex - StartIndex - (Step < 0)) / Step);

        using NewDims = typename Dims::template Set<DimIndex, Span>;
        using NewStrides = typename Strides::template Set<DimIndex, Stride(DimIndex) * Step>;

        if constexpr (NewDims() == Dims() && NewStrides() == Strides()) {
            return *this;
        } else {
            return Tensor<T, Allocator, NewDims, NewStrides>(data + Stride(DimIndex) * StartIndex);
        }
    }

    template<int... Dim> requires((Dims::Values(Dim < 0 ? Rank + Dim : Dim) == 1) && ...)
    constexpr auto squeeze() {

    }

    template<int Dim = -1, int Broadcast = 1> requires((Dim >= 0 && Dim <= Rank) || (Dim < 0 && Rank + Dim + 1 <= Rank))
    constexpr auto unsqueeze(this auto&& self) {
        constexpr int DimIndex = Dim < 0 ? Rank + Dim + 1 : Dim;
        if constexpr (DimIndex == Rank) {
            return Tensor<T, Allocator, typename Dims::template Append<Broadcast>, typename Strides::template Append<0>>(self.data);
        } else {
            return Tensor<T, Allocator, typename Dims::template Insert<DimIndex, Broadcast>, typename Strides::template Insert<DimIndex, 0>>(self.data);
        }
    }

    template<int Dim, int Size> requires(Dims::Values(Dim) == 1)
    constexpr auto broadcast(this auto&& self) {
        return Tensor<T, Allocator, typename Dims::template Set<Dim, Size>, typename Strides::template Set<Dim, 0>>(self.data);
    }


    /*
    template<int... NewOrder> requires(sizeof...(NewOrder) == Rank && ((NewOrder <= Rank) && ...))
    constexpr auto transpose() {
        constexpr int Arr[] = { NewOrder... };
        using NewDims = decltype(constexpr_for<0, sizeof...(NewOrder)>([]<int I>(auto dims) {
            return typename TYPE(dims)::template Append<Dims::Values(Arr[I])>();
        }, Sequence<>()));
        using NewStrides = decltype(constexpr_for<0, sizeof...(NewOrder)>([]<int I>(auto dims) {
            return typename TYPE(dims)::template Append<Strides::Values(Arr[I])>();
        }, Sequence<>()));
        return Tensor<T, Allocator, NewDims, NewStrides>(data);
    }

    template<int... NewOrder> requires(sizeof...(NewOrder) == 0 && Rank == 2)
    constexpr auto transpose() {
        return Tensor<T, Allocator, Sequence<Dims::Values(1), Dims::Values(0)>, Sequence<Strides::Values(1), Strides::Values(0)>>(data);
    }
    */

    template<int D0_, int D1_>
    constexpr auto transpose(this auto&& self) {
        constexpr int D0 = D0_ < 0 ? Rank + D0_ : D0_;
        constexpr int D1 = D1_ < 0 ? Rank + D1_ : D1_;
        static_assert(D0 < Rank && D1 < Rank, "Invalid dimensions for transpose");
        return Tensor<T, Allocator, typename Dims::template Swap<D0, D1>, typename Strides::template Swap<D0, D1>>(self.data);
    }

    template<typename NewDims> requires(NewDims::Product() == Size || (NewDims::template IndexOf<-1> != -1))
    constexpr auto reshape(this auto&& self) {
        auto reshape_impl = [&](auto&& self) {
            constexpr auto in1 = NewDims::template IndexOf<-1>;
            if constexpr (in1 != -1) {
                return Tensor<T, Allocator, typename NewDims::template Set<in1, -Size / (NewDims::Product())>>(self.data);
            } else return Tensor<T, Allocator, NewDims>(self.data);
        };
        if constexpr (!NormalStrides) {
            // Don't know how to easily do this without cloning
            return reshape_impl(self.clone());
        } else {
            return reshape_impl(self);
        }
    }


    // Every dimension != 1 in the source must exist in the target, and they must be in the same order, otherwise the broadcast fails
    template<AnySequence TargetDims>
    requires (std::is_same_v<TargetDims, Dims> || (Rank == 1 && Dim(0) == 1) ||
        TargetDims:: template ContainsOrderedSubset<decltype(RemoveFakeDimsImpl_t::first)>())
    constexpr auto broadcastTo(this auto&& self) {
        if constexpr (std::is_same_v<TargetDims, Dims>) {
            return self;
        } else if constexpr(Rank == 1 && Dim(0) == 1) { // Scalar tensor can be broadcasted to any size
            using Strides = decltype(constexpr_for<0, TargetDims::Size>([]<int I>(auto strides) {
                return typename decltype(strides)::template Append<0>();
            }, Sequence<>()));
            return Tensor<T, Allocator, TargetDims, Strides>(self.data);
        } else {
            auto clean = self.removeFakeDims();

            return constexpr_for<0, TargetDims::Size>([]<int I>(auto tensor) {
                if constexpr(I == decltype(tensor)::Rank) {
                    return tensor.template unsqueeze<-1, TargetDims::template Values<I>()>();
                } else if constexpr(decltype(tensor)::Dims::template Values<I>() != TargetDims::template Values<I>()) {
                    return tensor.template unsqueeze<I, TargetDims::template Values<I>()>();
                } else return tensor;
            }, clean);
        }
    }

    template<AnyTensor U> auto broadcastTo(this auto&& self, const U&) { return self.template broadcastTo<typename U::Dims>(); }

    template<int Axis, AnySequence Sections, bool View = true> requires(Sections::Sum() == Dim(Axis))
    constexpr auto split() {
        int offset = 0;
        return constexpr_for<0, Sections::Size>([&]<int I>(auto tuple) {
            auto tensor = Tensor<T, Allocator, typename Dims::template Set<Axis, Sections::Values(I)>, Strides>(data + offset);
            if constexpr (!View) {
                tensor = tensor.clone();
            }
            auto result = std::tuple_cat(tuple, tensor);
            offset += Sections::Values(I) * Strides::Values(Axis);
            return result;
        }, std::make_tuple());
    }

    constexpr decltype(auto) scalar() const noexcept requires(IsScalar) {
        return data[0];
    }

    constexpr operator decltype(auto)() const noexcept requires(IsScalar) {
        return data[0];
    }




    // Reductions
    template<int D, bool KeepDims = true, typename Pre = decltype(ForwardFunction), typename Post = decltype(ForwardFunction)>
    decltype(auto) reduce(auto&& f, Post postprocess = ForwardFunction, Pre preprocess = ForwardFunction) {
        auto result = ReduceNode<D < 0 ? Rank + D : D>(*this, f, preprocess, postprocess);
        if constexpr (!KeepDims) {
            return result.template squeeze<D>();
        } else {
            return result;
        }
    }
    template<int D, bool KeepDims = true> decltype(auto) sum() { return reduce<D, KeepDims>([](const auto& a, const auto& b) { return a + b; }); }
    template<int D, bool KeepDims = true> decltype(auto) min() { return reduce<D, KeepDims>([](const auto& a, const auto& b) { return std::min(a, b); }); }
    template<int D, bool KeepDims = true> decltype(auto) max() { return reduce<D, KeepDims>([](const auto& a, const auto& b) { return std::max(a, b); }); }
    template<int D, bool KeepDims = true> decltype(auto) mean() { return reduce<D, KeepDims>([](const auto& a, const auto& b) { return a + b; }, []__multi__(const auto& x) { return x / Dim(D); }); }


    // Part of the node system, evaluating a tensor is a no-op
    //auto evaluate() { return *this; }
};

template<AnyTensor T> requires(T::device != kCUDA)
std::ostream& operator<<(std::ostream& os, const T& tensor) {
    [&](this auto&& self, auto... indices) {
        static_assert(sizeof...(indices) <= T::Rank);
        if constexpr (sizeof...(indices) == T::Rank) {
            os << tensor(indices...) << ',' << ' ';
        } else {
            os << '[';
            for (int i = 0; i < T::Dim(sizeof...(indices)); i++) {
                self(indices..., i);
            }
            os << "],\n";
        }
    }();
    return os;
}

template<AnyTensor T> requires(std::decay_t<T>::device == kCUDA)
std::ostream& operator<<(std::ostream& os, T&& tensor) {
    auto tmp = tensor.contiguous().template to<tadma::kCPU>();
    check_cuda(cudaStreamSynchronize(tadma::stream));
    return os << tmp;
}

using NoTensor = Tensor<bool, Allocator<kCPU>, Sequence<0>>;

template<AnyTensor T> using TensorLike = Tensor<typename T::ValueType, typename T::AllocatorType, typename T::Dims>;

template<typename T, typename Dims, typename Strides = Sequence<>> using CpuTensor = Tensor<T, Allocator<kCPU>, Dims, Strides>;
template<typename T, typename Dims, typename Strides = Sequence<>> using CudaTensor = Tensor<T, Allocator<kCUDA>, Dims, Strides>;

template<AnyTensor T> using BaseTensor = Tensor<typename T::ValueType, typename T::AllocatorType, typename T::Dims, Sequence<>>;


template<AnyTensor T>
constexpr auto shape(const T&) {
    return tfs(typename T::Dims());
}

constexpr decltype(auto) operator+=(AnyTensor auto& t, const Scalar auto& value) { return InplaceNode(t, [value] (const auto& x) { return x + value; }); }
constexpr decltype(auto) operator-=(AnyTensor auto& t, const Scalar auto& value) { return InplaceNode(t, [value] (const auto& x) { return x - value; }); }
constexpr decltype(auto) operator*=(AnyTensor auto& t, const Scalar auto& value) { return InplaceNode(t, [value] (const auto& x) { return x * value; }); }
constexpr decltype(auto) operator/=(AnyTensor auto& t, const Scalar auto& value) { return InplaceNode(t, [value] (const auto& x) { return x / value; }); }

constexpr auto operator+(const AnyTensor auto& t, const Scalar auto& value) { return EltwiseNode(t, [value] (const auto& x) { return x + value; }); }
constexpr auto operator-(const AnyTensor auto& t, const Scalar auto& value) { return EltwiseNode(t, [value] (const auto& x) { return x - value; }); }
constexpr auto operator*(const AnyTensor auto& t, const Scalar auto& value) { return EltwiseNode(t, [value] (const auto& x) { return x * value; }); }
constexpr auto operator/(const AnyTensor auto& t, const Scalar auto& value) { return EltwiseNode(t, [value] (const auto& x) { return x / value; }); }

constexpr auto operator-(const Scalar auto& value, AnyTensor auto& t) { return EltwiseNode(t, [value] (const auto& x) { return value - x; }); }
constexpr auto operator/(const Scalar auto& value, AnyTensor auto& t) { return EltwiseNode(t, [value] (const auto& x) { return value / x; }); }

constexpr auto operator> (const AnyTensor auto& t, const Scalar auto& s) { return EltwiseNode(t, [s] (const auto& a) { return a > s; }); }
constexpr auto operator< (const AnyTensor auto& t, const Scalar auto& s) { return EltwiseNode(t, [s] (const auto& a) { return a < s; }); }
constexpr auto operator==(const AnyTensor auto& t, const Scalar auto& s) { return EltwiseNode(t, [s] (const auto& a) { return a == s; }); }
constexpr auto operator!=(const AnyTensor auto& t, const Scalar auto& s) { return EltwiseNode(t, [s] (const auto& a) { return a != s; }); }

constexpr auto operator> (const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a > b; }); }
constexpr auto operator< (const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a < b; }); }
constexpr auto operator==(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a == b; }); }
constexpr auto operator!=(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a != b; }); }


decltype(auto) operator+=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] (const auto& a, const auto& b) { return a + b; }); }
decltype(auto) operator-=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] (const auto& a, const auto& b) { return a - b; }); }
decltype(auto) operator*=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] (const auto& a, const auto& b) { return a * b; }); }
decltype(auto) operator/=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] (const auto& a, const auto& b) { return a / b; }); }


constexpr auto operator+(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a + b; }); }
constexpr auto operator-(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a - b; }); }
constexpr auto operator*(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a * b; }); }
constexpr auto operator/(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a / b; }); }

constexpr auto operator-(const AnyTensor auto& t) { return EltwiseNode(t, [] (const auto& x) { return -x; }); }

constexpr auto operator&(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a & b; }); }
constexpr auto operator|(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a | b; }); }
constexpr auto operator^(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a ^ b; }); }

constexpr decltype(auto) operator&=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] (const auto& a, const auto& b) { return a & b; }); }
constexpr decltype(auto) operator|=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] (const auto& a, const auto& b) { return a | b; }); }
constexpr decltype(auto) operator^=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] (const auto& a, const auto& b) { return a ^ b; }); }

constexpr auto operator&&(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a && b; }); }
constexpr auto operator||(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] (const auto& a, const auto& b) { return a || b; }); }

};
