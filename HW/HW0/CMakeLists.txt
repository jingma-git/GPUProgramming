cmake_minimum_required(VERSION 2.8)

if(MSVC)
    set(CMAKE_CUDA_ARCHITECTURES 89) # GForceRTX 4060 Laptop GPU
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")
endif()

project(add LANGUAGES CXX CUDA)

# CUDA
find_package(CUDA REQUIRED)

file(GLOB source src/*.cpp src/*.cu)
cuda_add_executable(add ${source})
target_link_libraries(add ${CUDA_LIBRARIES})

# add_executable(test test.cpp)
# target_link_libraries(test ${OpenCV_LIBRARIES})