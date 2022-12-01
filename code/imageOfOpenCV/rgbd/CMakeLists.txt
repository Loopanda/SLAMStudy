option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" ON)
project(joinMap)
cmake_minimum_required(VERSION 3.16)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# eigen
include_directories("/usr/include/eigen3")

# pcl
find_package(PCL 1.9.1 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# pangolin
find_package(Pangolin REQUIRED)

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${Sophus_LIBRARIES} fmt)
target_link_libraries(joinMap ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Pangolin_LIBRARIES}) 