#include <iostream>
#include <cuda_runtime.h>

#include "include/tadma/Tadma.hpp"


__device__ void iota_recursive(tadma::AnyTensor auto t, int& i) {

    if constexpr (decltype(t)::Rank == 1) {
        for(int j = 0; j < t.Dim(0); j++) {
            static_assert(std::is_same_v<std::decay_t<decltype(t(0))>, float>);
            t(j) = i++;
        }
    } else {
        for (int j = 0; j < t.Dim(0); j++) {
            iota_recursive(t(j), i);
        }
    }
}

__global__ void iota(tadma::AnyTensor auto t) requires (t.device == tadma::kCPU){
    for (int i = 0; i < t.Dim(0); i++) {
        for (int j = 0; j < t.Dim(1); j++) {
            t(i,j) = i + j;
        }
    }
}
/*

__global__ void kernel2(AnyTensor auto t) {
    for (int i = 0; i < t.Dim(0); i++) {
        for (int j = 0; j < t.Dim(1); j++) {
            for (int k = 0; k < t.Dim(2); k++) {
                printf("%d ", t(i,j,k));
            }
        }
    }
}
*/

#include <cxxabi.h>

std::string demangle(const char* name) {
    int status;
    char* res = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0) {
        std::string result(res);
        free(res);
        return result;
    } else {
        return name;
    }
}
#include <fstream>


std::string readline(std::ifstream& ifs) {
    std::string line;
    std::getline(ifs, line);
    return line;
}

struct Test {
    std::ifstream ifs;
    std::string a = readline(ifs);
    Test() : ifs("../main.cu") {

    }
};


int main() {
    namespace T = tadma;
    T::InitGuard guard;

    {

        T::Tensor<float, T::Allocator<T::kCUDA>, T::Sequence<1>> t;
        T::Tensor<float, T::Allocator<T::kCUDA>, T::Sequence<122, 5, 1, 4>> t2;
        iota(t2, 0);

        t =1;
        t2 = 1;

        auto x = matmul(t, t2);

        std::cout << x;

    }
    /*
    {


        T::Tensor<float, T::Allocator<T::kCPU>, T::Sequence<2, 28, 28>> cpu;
        T::randn(cpu);

        auto input = cpu.to<T::kCUDA>();
        T::Tensor<float, T::Allocator<T::kCUDA>, T::Sequence<784, 10>> weights;

        weights = 0.1;

        auto p = matmul(input.reshape<2, 784>(), weights);

        std::cout << p(0) << std::endl << p(1) << std::endl;

        return 0;

        exp(p);

        auto sum = p.template sum<1>();

        std::cout << p << std::endl;
        std::cout << sum << std::endl;

        softmax<1>(p);
        std::cout << p << std::endl;


        return 0;
    }
     */
    //auto t3 =  t2 / 2.0;


    //softmax<1>(t2);

    //std::cout << t3 << std::endl;

    //std::cout << t2 << std::endl;


    /*
    tx = 5.2;
    tadma::eltwise(tx, []__device__(auto x) {
        return x * x;
    });

    tx.slice<0, 3, 10>() *= -2.0;

    std::cout << tx << std::endl;
    */
    return 0;
}