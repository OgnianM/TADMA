#pragma once
#include <random>
#include "Tensor.hpp"
#include "Kernels.hpp"
#include <cuda_fp16.h>

namespace tadma {


template<auto alpha_ = 1, auto beta_ = 0, AnyTensor A_, AnyTensor B_, AnyTensor Copt_ = NoTensor>
requires(SameDevice<A_, B_> && A_::device == kCUDA)
auto matmul(const A_& A, const B_& B, Copt_&& C__ = NoTensor()) {
    using T = typename A_::ValueType;

    using DimsA = typename A_::Dims;
    using DimsB = typename B_::Dims;

    using StridesA = typename A_::Strides;
    using StridesB = typename B_::Strides;

    constexpr auto RankA = DimsA::Size;
    constexpr auto RankB = DimsB::Size;


    if constexpr (RankA == 3 && RankB == 3 && A_::Dim(0) == B_::Dim(0)) { // Base case BxNxK * BxKxM
        using T = typename A_::ValueType;

        constexpr auto Rank = A_::Rank;

        constexpr int M = A_::Dim(Rank - 2);
        constexpr int N = B_::Dim(Rank - 1);
        constexpr int K = A_::Dim(Rank - 1);

        static_assert(B_::Dim(Rank - 2) == K, "K dimensions must match");

        constexpr int Batch = A_::Dim(0);

        Tensor<T, Allocator<kCUDA>, Sequence<Batch, M, N>> result;

        using R = TYPE(result);

        auto C = [&]() { if constexpr (beta_ == 0) return result; else return C__.broadcastTo(result); }();

        using C_ = TYPE(C);

        float alpha = alpha_;
        float beta = beta_;

        auto transa = CUBLAS_OP_N;
        auto transb = CUBLAS_OP_N;

        int lda = A_::Stride(1);
        int ldb = B_::Stride(1);
        int ldc = C_::Stride(1);
        int ldr = R::Stride(1);

        int64_t stridea = A_::Stride(0);
        int64_t strideb = B_::Stride(0);
        int64_t stridec = C_::Stride(0);
        int64_t strider = R::Stride(0);

        int rowMajor = CUBLASLT_ORDER_ROW;

        cublasLtMatmulDesc_t operationDesc = NULL;
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, resultDesc = NULL;

        auto get_type = []<typename TT>() -> cudaDataType {
            using T = typename TT::ValueType;
            if constexpr (std::is_same_v<T, float>) {
                return CUDA_R_32F;
            } else if constexpr (std::is_same_v<T, double>) {
                return CUDA_R_64F;
            } else if constexpr (std::is_same_v<T, __half>) {
                return CUDA_R_16F;
            } else if constexpr (std::is_same_v<T, int32_t>) {
                return CUDA_R_32I;
            } else if constexpr (std::is_same_v<T, int64_t>) {
                return CUDA_R_64I;
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                return CUDA_R_32U;
            } else if constexpr (std::is_same_v<T, uint64_t>) {
                return CUDA_R_64U;
            } else if constexpr (std::is_same_v<T, int8_t>) {
                return CUDA_R_8I;
            } else if constexpr (std::is_same_v<T, uint8_t>) {
                return CUDA_R_8U;
            }
            else static_assert(false, "Unsupported type");
        };

        // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
        // set the transforms for A and B
        check_cublas(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        check_cublas(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        check_cublas(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        // create matrix descriptors, we need to configure batch size and counts in this case
        check_cublas(cublasLtMatrixLayoutCreate(&Adesc, get_type.template operator()<A_>(), transa == CUBLAS_OP_N ? M : K, transa == CUBLAS_OP_N ? K : M, lda));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch)));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));

        check_cublas(cublasLtMatrixLayoutCreate(&Bdesc, get_type.template operator()<B_>(), transb == CUBLAS_OP_N ? K : N, transb == CUBLAS_OP_N ? N : K, ldb));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch)));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));

        check_cublas(cublasLtMatrixLayoutCreate(&Cdesc, get_type.template operator()<C_>(), M, N, ldc));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch)));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));
        check_cublas(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));

        check_cublas(cublasLtMatrixLayoutCreate(&resultDesc, get_type.template operator()<R>(), M, N, ldr));
        check_cublas(cublasLtMatrixLayoutSetAttribute(resultDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch)));
        check_cublas(cublasLtMatrixLayoutSetAttribute(resultDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strider, sizeof(strider)));
        check_cublas(cublasLtMatrixLayoutSetAttribute(resultDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));


        check_cublas(cublasLtMatmul(cublasLtHandle, operationDesc, &alpha, A.data.view, Adesc, B.data.view, Bdesc, &beta,
                                    C.data.view, Cdesc, result.data.view, resultDesc, nullptr, nullptr, 0, stream));


        check_cublas(cublasLtMatmulDescDestroy(operationDesc));
        check_cublas(cublasLtMatrixLayoutDestroy(Adesc));
        check_cublas(cublasLtMatrixLayoutDestroy(Bdesc));
        check_cublas(cublasLtMatrixLayoutDestroy(Cdesc));
        check_cublas(cublasLtMatrixLayoutDestroy(resultDesc));

        return result;
    } else {
        // Recursive broadcasting rules
        if constexpr (RankA == 1) {
            return matmul(A.template unsqueeze<0>(), B, C__);
        } else if constexpr (RankB == 1) {
            return matmul(A, B.template unsqueeze<1>(), C__);
        } else if constexpr (RankA == 2) {
            return matmul(A.template unsqueeze<0>(), B, C__);
        } else if constexpr (RankB == 2) {
            return matmul(A, B.template unsqueeze<0>(), C__);
        } else if constexpr (RankA == 3 && RankB == 3 && A_::Dim(0) != B_::Dim(0)) {
            if constexpr (A_::Dim(0) == 1) {
                return matmul(A.template broadcast<0, B_::Dim(0)>(), B, C__);
            } else if constexpr (B_::Dim(0) == 1) {
                return matmul(A, B.template broadcast<0, A_::Dim(0)>(), C__);
            } else {
                static_assert(false, "Batch dimensions must be broadcastable");
            }
        } else if constexpr (RankA >= 4) {
            return matmul(A.template reshape<-1, DimsA::Values(RankA - 2), DimsA::Values(RankA - 1)>(), B, C__)
                    .template reshape<typename DimsA::template Set<RankA - 1, DimsB::Values(RankB - 1)>>();
        } else if constexpr (RankB >= 4) {
            return matmul(A, B.template reshape<-1, DimsB::Values(RankB - 2), DimsB::Values(RankB - 1)>(), C__)
                    .template reshape<typename DimsB::template Set<RankB - 2, DimsA::Values(RankA - 2)>>();
        }

        else static_assert(false, "Unsupported rank");
    }
}

__multi__ constexpr auto tanh(const Scalar auto& x) {
    auto en = exp(-x);
    auto ep = exp(x);
    return (ep - en) / (ep + en);
}

#define MKOP(name, expr) template<bool Inplace = true> auto name(AnyTensor auto&& t) { return MakeEltwiseNode<Inplace>(t, [] __multi__ (const auto& x) { return expr; }); }
#define STD(name) MKOP(name, std::name(x))

// Standard math
MKOP(abs, std::abs(x))
MKOP(sin, std::sin(x)) MKOP(cos, std::cos(x)) MKOP(tan, std::tan(x))
MKOP(asin, std::asin(x)) MKOP(acos, std::acos(x)) MKOP(atan, std::atan(x))
MKOP(asinh, std::asinh(x)) MKOP(acosh, std::acosh(x)) MKOP(atanh, std::atanh(x))
MKOP(exp, std::exp(x)) MKOP(log, std::log(x)) MKOP(sqrt, std::sqrt(x))
MKOP(erf, std::erf(x))

// Activation functions
MKOP(relu, x > 0 ? x : 0)
MKOP(sigmoid, 1 / (1 + std::exp(-x)))
MKOP(tanh, tanh(x))
MKOP(gelu, 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))));
MKOP(negate, -x)

#undef MKOP

template<bool Inplace = true> auto pow(AnyTensor auto&& t, const Scalar auto& s) {
    return MakeEltwiseNode<Inplace>(t, [s] __multi__ (const auto& x) { return std::pow(x, s); });
}

template<int Dim, bool Inplace = true> auto softmax(AnyTensor auto t) {
    auto e = exp<Inplace>(t);
    return e /= e.template sum<Dim>();
}

template<int D> decltype(auto) mean_stddev(AnyTensor auto&& t) {
    static constexpr auto size = TYPE(t)::Dim(D);
    auto mean = t.template mean<D>();
    auto stddev = CombineToNode(t, mean, []__multi__(const auto& a, const auto& b) {
        auto x = a - b; return x * x;
    }).template reduce<D>([]__multi__(const auto& a, const auto& b) { return a + b; },
                          []__multi__(const auto& x) { return std::sqrt(x / size); });
    return std::pair {mean, stddev};
}

template<int D, bool Inplace = true> decltype(auto) layer_norm(AnyTensor auto&& t, const AnyTensor auto& gamma, const AnyTensor auto& beta, float eps = 1e-5) {
    constexpr auto Dim = RealDim<D, TYPE(t)::Rank>;
    auto [mean, stddev] = mean_stddev<Dim>(t);
    if constexpr (Inplace) {
        CombineVariadicNode([eps]__multi__(auto& x, const auto& mean, const auto& stddev, const auto& gamma, const auto& beta) {
            x = (x - mean) / (stddev + eps) * gamma + beta;
        }, t, mean, stddev, gamma, beta);
        return t;
    } else {
        TensorLike<TYPE(t)> result;
        CombineVariadicNode([eps]__multi__(auto& result, const auto& x, const auto& mean, const auto& stddev, const auto& gamma, const auto& beta) {
            result = (x - mean) / (stddev + eps) * gamma + beta;
        }, result, t, mean, stddev, gamma, beta);
        return result;
    }
}

template<int D, bool Inplace = true> auto rms_norm(AnyTensor auto&& t) {
    auto rms = t.template reduce<D>([](const auto& a, const auto& b) { return a + b; }, // Reduction
                                    [](const auto& x) { return std::sqrt(x / TYPE(t)::Dim(D)); }, // Postprocess
                                    [](const auto& x) { return x * x; } // Preprocess
                                    );
    if constexpr (Inplace) {
        t /= rms;
        return t;
    } else {
        return t / rms;
    }
}

template<AnyTensor Cond, AnyTensor X_, AnyTensor Y_>
auto where(Cond condition, X_ X, Y_ Y) {
    Tensor<std::common_type_t<typename X_::ValueType, typename Y_::ValueType>, typename X_::AllocatorType, typename Cond::Dims> result;
    CombineVariadicNode([](auto& r, const auto& c, const auto& x, const auto& y) {
        r = c ? x : y;
    }, result, condition, X, Y);
    return result;
}

/// @brief https://onnx.ai/onnx/operators/onnx__Gather.html
template<int Axis=0>
auto gather(const AnyTensor auto& input, const AnyTensor auto& indices) {
    if constexpr (TYPE(indices)::IsScalar) {
        return input.template index<Axis>(indices);
    } else {
        using NewDims = decltype(constexpr_for<Axis + 1, input.Rank>([&]<int I>(auto dims){
            return typename decltype(dims):: template Append<input.Dim(I)>();
        }, typename TYPE(indices)::Dims()));

        return EltwiseVariadicNDNode<NewDims>([]__multi__(auto idx, auto input, auto indices) {
            auto new_index = constexpr_for<0, indices.Rank>([&]<int64_t I> (auto indices){
                return indices(idx[I]);
            }, indices);
            idx[indices.Rank] = new_index;
            return input(idx);
        }, input, indices);
    }
}


template<int Axis = 0, AnyTensor T, AnyTensor... Ts> requires(SameRank<T, Ts...> && Axis < T::Rank &&
    ((typename T::Dims::template Set<Axis, 0>() == typename Ts::Dims::template Set<Axis, 0>()) && ...))
auto concat(const T& input0, const Ts&... inputs) {
    constexpr auto NewSize = T::Dim(Axis) + (Ts::Dim(Axis) + ...);

    return EltwiseVariadicNDNode<typename T::Dims::template Set<Axis, NewSize>>([]__multi__(auto indices, auto&&... inputs) {
        auto axis_index = indices[Axis];

        typename T::ValueType result;

        constexpr_for<0, sizeof...(inputs)>([&]<int I>(int64_t accumulator) {
            constexpr auto Dim = TYPE(inputs...[I])::Dim(Axis);

            if (axis_index >= accumulator && accumulator + Dim > axis_index) {
                indices[Axis] = axis_index - accumulator;
                result = inputs...[I](indices);
            }
            return accumulator + Dim;
        }, 0);

        return result;
    }, input0, inputs...);
}


template<AnyTensor T> requires (T::device == kCPU)
void randn(T& t, const float& mean = 0, const float& stddev = 1) {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<typename T::ValueType> dist(mean, stddev);
    for (int i = 0; i < t.Size; i++) {
        t[i] = dist(gen);
    }
}

template<AnySequence Shape, Memory device>
decltype(auto) index_to_value(auto&& f) {
    Tensor<decltype(f(0)), Allocator<device>, Shape> result;
    return make_node<Sequence<Shape::Product()>>([f](auto indices, auto& result) {
        result[indices[0]] = f(indices[0]);
    }, result);
}

};
