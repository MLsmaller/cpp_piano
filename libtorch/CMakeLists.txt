cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(dcgan)

set(Torch_DIR /root/libtorch/share/cmake/Torch)  # 我的 libtorch 的路径
find_package(Torch REQUIRED)


set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
#set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)