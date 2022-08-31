cmake_minimum_required(VERSION 3.10)

project(BitonicSort LANGUAGES CUDA)

add_executable(BitonicSort main.cu)

set_target_properties(
        BitonicSort
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
