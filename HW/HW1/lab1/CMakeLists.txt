cmake_minimum_required(VERSION 3.0)

if(MSVC)
    set(CMAKE_CUDA_ARCHITECTURES 89) # GForceRTX 4060 Laptop GPU
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")
endif()

project(blur LANGUAGES CXX CUDA)



file(GLOB SRC src/*.cpp src/*.cu)
if(MSVC)
    find_package(CUDAToolkit)
    add_executable(blur ${SRC})
    # target_link_libraries(blur ${CUDA_LIBRARIES})
    set_target_properties(blur PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
elseif(UNIX)
    find_package(CUDA) # GeForce GTX TITAN X
    include_directories(${CUDA_INCLUDE_DIRS})

    add_executable(blur ${SRC})
    target_link_libraries(blur ${CUDA_LIBRARIES})

    add_executable(audioblur ${SRC})
    target_link_libraries(audioblur ${CUDA_LIBRARIES} sndfile)
    target_compile_definitions(audioblur PRIVATE AUDIO_ON=1)
endif()

add_executable(test test.cpp)
