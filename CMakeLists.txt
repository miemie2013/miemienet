cmake_policy(SET CMP0017 NEW)

cmake_minimum_required(VERSION 3.1)

set(APP_NAME main)
project(${APP_NAME})

message(STATUS "    CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "    CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

set(MM2NCNN_ROOT ${CMAKE_SOURCE_DIR}/miemienet)



# ------------------------------- OS -------------------------------
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(WINDOWS TRUE)
  set(SYSTEM_STRING "Windows")
  add_definitions(-DWINDOWS)
  message(STATUS "    platform: WINDOWS")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LINUX TRUE)
  set(SYSTEM_STRING "Linux")
  add_definitions(-DLINUX)
  message(STATUS "    platform: LINUX")
endif()


# ------------------------------- OpenCV CUDA -------------------------------
# https://blog.csdn.net/baobei0112/article/details/123598794

# OpenCV
find_package(OpenCV QUIET COMPONENTS opencv_world)

# for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
if("${OpenCV_LIBS}" STREQUAL "")
    set(OpenCV_FOUND FALSE)
endif()
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
endif()
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
endif()

message(STATUS "    OpenCV_INSTALL_PATH: ${OpenCV_INSTALL_PATH}")
message(STATUS "    OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "    OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "    OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")


find_package(OpenCV QUIET COMPONENTS opencv_world)

#message(STATUS "    CUDA_cudadevrt_LIBRARY: ${CUDA_cudadevrt_LIBRARY}")
#message(STATUS "    CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
#message(STATUS "    CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}")
#message(STATUS "    CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
#message(STATUS "    CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
#message(STATUS "    CUDA_VERSION: ${CUDA_VERSION}")


include_directories(OpenCV_INCLUDE_DIRS)
#include_directories(CUDA_INCLUDE_DIRS)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_21 -std=c++11)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)


# ------------------------------- for -------------------------------
set(FOR_SRCS
    examples/for.cpp
)

add_executable(for ${FOR_SRCS})


# ------------------------------- miemienet -------------------------------
include_directories(miemienet)
include_directories(miemiedet)

#[[
# ------------------------------- test_tanh -------------------------------
file(GLOB_RECURSE Tanh_SRCS
    miemienet/nn/*.cu
    miemienet/nn/*.cpp
    examples/test_cuda.cu
)

message(STATUS "    Tanh_SRCS: ${Tanh_SRCS}")

cuda_add_executable(test_cuda ${Tanh_SRCS})

]]

# ------------------------------- test_tanh -------------------------------
file(GLOB_RECURSE Graph_SRCS
    miemienet/*.cpp
    miemienet/framework/*.cpp
    miemienet/nn/*.cpp
    miemiedet/models/backbones/*.cpp
    miemiedet/models/necks/*.cpp
    miemiedet/models/heads/*.cpp
    test/test2_001_conv_mm_x86.cpp
)

message(STATUS "    Graph_SRCS: ${Graph_SRCS}")

add_executable(mmmmmm ${Graph_SRCS})
target_link_libraries(mmmmmm ${OpenCV_LIBS})

