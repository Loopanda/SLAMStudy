cmake_minimum_required(VERSION 2.8)
project(ceres_g2o)
# cmake_policy(SET CMP0074 NEW)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
# export(LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH})
# find_package(CUDA REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
list( APPEND CMAKE_MODULE_PATH /home/loopanda/gitPacage/g2o/cmake_modules ) 
set(G2O_ROOT /usr/local/include/g2o)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES})
# target_link_libraries(ceresCurveFitting PRIVATE
#         CUDA::cublas
#         CUDA::cudart
#         CUDA::cusolver
#         CUDA::cusparse
#         )

add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
