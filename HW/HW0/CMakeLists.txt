cmake_minimum_required(VERSION 2.8)
project(add)

# CUDA
find_package(CUDA REQUIRED)

file(GLOB source src/*.cpp src/*.cu)
cuda_add_executable(add ${source})
target_link_libraries(add ${CUDA_LIBRARIES})

# add_executable(test test.cpp)
# target_link_libraries(test ${OpenCV_LIBRARIES})