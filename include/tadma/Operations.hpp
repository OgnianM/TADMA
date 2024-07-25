#pragma once
#include <random>
#include "Tensor.hpp"
#include "Kernels.hpp"

namespace tadma {

template<auto alpha_ = 1, auto beta_ = 0, AnyTensor A_, AnyTensor B_, AnyTensor Copt_ = NoTensor>
requires(SameDevice<A_, B_> && A_::device == kCUDA && SameRank<A_, B_> && A_::Rank >= 2) // TODO: matrix vector and vector matrix support
auto matmul(const A_& A__, const B_& B__, Copt_&& Copt = NoTensor()) {
    // A = B...xNxK
    // B = B...xKxM
    // C is broadcastable to B...xNxM
    // Result is B...xNxM
    using T = typename A_::ValueType;

    constexpr auto Rank = A_::Rank;

    constexpr int N = A_::Dim(Rank - 2);
    constexpr int M = B_::Dim(Rank - 1);
    constexpr int K = A_::Dim(Rank - 1);

    static_assert(B_::Dim(Rank - 2) == K, "K dimensions must match");

    Tensor<T, Allocator<A_::device>, typename A_::Dims::template First<Rank - 2>::template Append<N>::template Append<M>> result_;
    auto result = result_.template reshape<-1, N, M>();
    auto A = A__.template reshape<-1, N, K>();
    auto B = B__.template reshape<-1, K, M>();

    constexpr int Batch = A.Dim(0);
    static_assert(Batch == B.Dim(0) && Batch == result.Dim(0), "Batch dimensions must match");

    using R = TYPE(result);

    auto C = [&]() { if constexpr (beta_ == 0) return result; else return Copt.broadcastTo(result); }();

    using C_ = TYPE(C);

    T alpha = alpha_;
    T beta = beta_;

    auto transa = CUBLAS_OP_N;
    auto transb = CUBLAS_OP_N;

    int lda = A.Stride(1);
    int ldb = B.Stride(1);
    int ldc = C.Stride(1);
    int ldr = R::Stride(1);

    int64_t stridea = A.Stride(0);
    int64_t strideb = B.Stride(0);
    int64_t stridec = C.Stride(0);
    int64_t strider = R::Stride(0);

    int rowMajor = CUBLASLT_ORDER_ROW;

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, resultDesc = NULL;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    check_cublas(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    check_cublas(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    check_cublas(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we need to configure batch size and counts in this case
    check_cublas(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? M : K, transa == CUBLAS_OP_N ? K : M, lda));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));

    check_cublas(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? K : N, transb == CUBLAS_OP_N ? N : K, ldb));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));

    check_cublas(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, ldc));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));

    check_cublas(cublasLtMatrixLayoutCreate(&resultDesc, CUDA_R_32F, M, N, ldr));
    check_cublas(cublasLtMatrixLayoutSetAttribute(resultDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &Batch, sizeof(Batch)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(resultDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strider, sizeof(strider)));
    check_cublas(cublasLtMatrixLayoutSetAttribute(resultDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor, sizeof(rowMajor)));


    check_cublas(cublasLtMatmul(cublasLtHandle, operationDesc, &alpha, A.view, Adesc, B.view, Bdesc, &beta,
                                C.view, Cdesc, result.view, resultDesc, nullptr, nullptr, 0, stream));


    check_cublas(cublasLtMatmulDescDestroy(operationDesc));
    check_cublas(cublasLtMatrixLayoutDestroy(Adesc));
    check_cublas(cublasLtMatrixLayoutDestroy(Bdesc));
    check_cublas(cublasLtMatrixLayoutDestroy(Cdesc));
    check_cublas(cublasLtMatrixLayoutDestroy(resultDesc));

    return result_;
}

__multi__ auto tanh(const Scalar auto& x) {
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
MKOP(erf, std::erf)

// Activation functions
MKOP(relu, x > 0 ? x : 0)
MKOP(sigmoid, 1 / (1 + std::exp(-x)))
MKOP(tanh, tanh(x))
MKOP(gelu, 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))));


#undef MKOP

auto operator+=(AnyTensor auto&& t, const Scalar auto& value) { return InplaceNode(t, [value] __multi__ (const auto& x) { return x + value; }); }
auto operator-=(AnyTensor auto&& t, const Scalar auto& value) { return InplaceNode(t, [value] __multi__ (const auto& x) { return x - value; }); }
auto operator*=(AnyTensor auto&& t, const Scalar auto& value) { return InplaceNode(t, [value] __multi__ (const auto& x) { return x * value; }); }
auto operator/=(AnyTensor auto&& t, const Scalar auto& value) { return InplaceNode(t, [value] __multi__ (const auto& x) { return x / value; }); }

auto operator+(const AnyTensor auto& t, const Scalar auto& value) { return EltwiseNode(t, [value] __multi__ (const auto& x) { return x + value; }); }
auto operator-(const AnyTensor auto& t, const Scalar auto& value) { return EltwiseNode(t, [value] __multi__ (const auto& x) { return x - value; }); }
auto operator*(const AnyTensor auto& t, const Scalar auto& value) { return EltwiseNode(t, [value] __multi__ (const auto& x) { return x * value; }); }
auto operator/(const AnyTensor auto& t, const Scalar auto& value) { return EltwiseNode(t, [value] __multi__ (const auto& x) { return x / value; }); }

auto operator>(const AnyTensor auto& t, const Scalar auto& s) { return EltwiseNode(t, [s] __multi__ (const auto& a) { return a > s; }); }
auto operator<(const AnyTensor auto& t, const Scalar auto& s) { return EltwiseNode(t, [s] __multi__ (const auto& a) { return a < s; }); }
auto operator==(const AnyTensor auto& t, const Scalar auto& s) { return EltwiseNode(t, [s] __multi__ (const auto& a) { return a == s; }); }
auto operator!=(const AnyTensor auto& t, const Scalar auto& s) { return EltwiseNode(t, [s] __multi__ (const auto& a) { return a != s; }); }

auto operator>(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a > b; }); }
auto operator<(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a < b; }); }
auto operator==(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a == b; }); }
auto operator!=(const AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a != b; }); }


auto operator-(const Scalar auto& value, AnyTensor auto& t) { return EltwiseNode(t, [value] __multi__ (const auto& x) { return value - x; }); }
auto operator/(const Scalar auto& value, AnyTensor auto& t) { return EltwiseNode(t, [value] __multi__ (const auto& x) { return value / x; }); }


auto operator+=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a + b; }); }
auto operator-=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a - b; }); }
auto operator*=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a * b; }); }
auto operator/=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a / b; }); }


template<AnyTensor A, AnyTensor B>
auto operator+(A& t, const B& t2) {
    if constexpr (Broadcastable<B, A>) { // Commutative operations can be reversed if broadcasting doesn't work
        return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a + b; });
    } else {
        return CombineToNode(t2, t, [] __multi__ (const auto& a, const auto& b) { return a + b; });
    }
}

auto operator-(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a - b; }); }
auto operator*(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a * b; }); }
auto operator/(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a / b; }); }

auto operator>(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a > b; }); }
auto operator<(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a < b; }); }
auto operator==(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a == b; }); }
auto operator!=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a != b; }); }

auto operator&(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a & b; }); }
auto operator|(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a | b; }); }
auto operator^(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a ^ b; }); }

auto operator&=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a & b; }); }
auto operator|=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a | b; }); }
auto operator^=(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a ^ b; }); }

auto operator&&(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a && b; }); }
auto operator||(AnyTensor auto& t, const AnyTensor auto& t2) { return CombineToNode(t, t2, [] __multi__ (const auto& a, const auto& b) { return a || b; }); }


template<int Dim, bool Inplace = true> auto softmax(AnyTensor auto t) { auto e = exp<Inplace>(t); return e /= e.template sum<Dim>(); }


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

template<AnyTensor Cond, AnyTensor X_, AnyTensor Y_> requires(SameDevice<Cond, X_, Y_>)
decltype(auto) where(Cond condition, X_ X, Y_ Y) {
    Tensor<std::common_type_t<typename X_::ValueType, typename Y_::ValueType>, typename X_::AllocatorType, typename Cond::Dims> result;
    CombineVariadicNode([]__multi__(auto& r, const auto& c, const auto& x, const auto& y) {
        r = c ? x : y;
    }, result, condition, X, Y);
    return result;
}

__global__ void GatherKernel(AnyTensor auto t, AnyTensor auto indices, AnyTensor auto result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < result.Size) {
        t[indices[i]].copyTo(result[i]);
    }
}

template<int Axis=0>
auto gather(AnyTensor auto t, AnyTensor auto indices) {
    using NewDims = decltype(constexpr_for<Axis + 1, t.Rank>([&]<int I>(auto dims){
        return typename decltype(dims):: template Append<t.Dim(I)>();
    }, typename TYPE(indices)::Dims()));
    Tensor<typename TYPE(t)::ValueType, typename TYPE(t)::AllocatorType, NewDims> result;


    return result;
}

template<AnyTensor T> requires (T::device == kCPU)
void randn(T& t, const float& mean = 0, const float& stddev = 1) {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<typename T::ValueType> dist(mean, stddev);
    for (int i = 0; i < t.Size; i++) {
        t[i] = dist(gen);
    }
}

template<AnyTensor T> requires (T::device == kCUDA)
void iota(T& t, const int& start = 0) {
    InplaceNode(t, [start] __multi__ (const auto& x) { return blockIdx.x * gridDim.x + threadIdx.x + start; });
}

};
