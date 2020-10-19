cmake_minimum_required(VERSION 2.8)
project(blur)

option(IMG_ON "Read Image" ON)
if(IMG_ON)
   add_definitions(-DIMG_ON)
endif()

# CUDA
find_package(CUDA REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("-- cv @ ${OpenCV_LIBRARIES}")
endif(OpenCV_FOUND)

file(GLOB source src/*.cpp src/*.cu)
cuda_add_executable(blur ${source})
target_link_libraries(blur ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})