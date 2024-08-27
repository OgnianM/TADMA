cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 23)

#set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang-14)

project(TADMA CXX CUDA)

option(INDEX_ONLY "Only build index" OFF)

if(NOT INDEX_ONLY)
add_executable(TADMA main.cu
        include/tadma/Tadma.hpp
        include/tadma/Sequence.hpp
        include/tadma/Meta.hpp
        include/tadma/Tensor.hpp
        include/tadma/Storage.hpp
        include/tadma/Kernels.hpp
        include/tadma/Allocator.hpp
        include/tadma/Operations.hpp
        include/tadma/Globals.hpp
        include/tadma/Defines.hpp
        include/tadma/Types.hpp
        include/tadma/Python.hpp
)

set_target_properties(TADMA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 23
)


target_link_libraries(TADMA
        PRIVATE
        cublasLt
        curand
        )

else ()
        message("Index only")
endif ()


add_executable(index
        include/tadma/Tadma.hpp
        include/tadma/Sequence.hpp
        include/tadma/Meta.hpp
        include/tadma/Tensor.hpp
        include/tadma/Storage.hpp
        include/tadma/Kernels.hpp
        include/tadma/Allocator.hpp
        include/tadma/Operations.hpp
        include/tadma/Globals.hpp
        include/tadma/Defines.hpp
        main.cpp
        include/tadma/Types.hpp
        include/tadma/Python.hpp)

set_target_properties(index PROPERTIES
        CXX_STANDARD 23
)


add_executable(generated scripts/generated.cu)

target_include_directories(generated PRIVATE include)

target_link_libraries(generated cublasLt)