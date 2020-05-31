cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example-cpp)

#set(Torch_DIR /root/libtorch/share/cmake/Torch)  # libtorch(cpu)的路径
set(Torch_DIR /root/packages/libtorch/share/cmake/Torch)  # libtorch(cpu)的路径

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if (NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

include_directories(${OpenCV_INCLUDE_DIR})
include_directories(./model/headers/ ./utils/headers)

message(STATUS "Pytorch status:")
message(STATUS "     libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#nclude_directories(${OpenCV_INCLUDE_DIRS}
#                    /root/libtorch/include
#                    /root/libtorch/include/torch/csrc/api/include/
#                    )

include_directories(${OpenCV_INCLUDE_DIRS}
                    /root/packages/libtorch/include
                    /root/packages/libtorch/include/torch/csrc/api/include/
                    )

file(GLOB SOURCE_FILES main.cpp /data/nextcloud/dbc2017/files/project/utils/*.cpp /data/nextcloud/dbc2017/files/project/model/*.cpp)
set(CMAKE_CXX_STANDARD 14)
add_executable(example-cpp ${SOURCE_FILES})
target_link_libraries(example-cpp ${TORCH_LIBRARIES} ${OpenCV_LIBS})

