cmake_minimum_required (VERSION 2.8)
project (Autonomous-vision)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

add_executable(cuda GPUcuda.cpp)
target_link_libraries(cuda ${OpenCV_LIBS})
