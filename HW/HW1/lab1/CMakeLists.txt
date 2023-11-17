cmake_minimum_required(VERSION 3.18)

if(MSVC)
    set(CMAKE_CUDA_ARCHITECTURES 89) # GForceRTX 4060 Laptop GPU
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")
endif()

project(blur LANGUAGES CXX CUDA)

# find_package(CUDA 12.0 REQUIRED)
find_package(CUDAToolkit)

file(GLOB SRC src/*.cpp src/*.cu)
if(MSVC)
    add_executable(blur ${SRC})
    # target_link_libraries(blur ${CUDA_LIBRARIES})
    set_target_properties(blur PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
elseif(LINUX)
    add_executable(audioblur ${SRC})
    # target_link_libraries(blur ${CUDA_LIBRARIES})
    set_target_properties(audioblur PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(audioblur PRIVATE AUDIO_ON)
endif()

add_executable(test test.cpp)
