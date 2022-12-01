cmake_minimum_required (VERSION 2.8)
project(EigenTest)

include_directories("/usr/include/eigen3")

add_executable(matrix eigenMatrix.cpp)