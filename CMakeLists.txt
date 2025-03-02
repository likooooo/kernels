cmake_minimum_required (VERSION 3.14)
project(type_traist_notebook 
    DESCRIPTION "type traist notebook"
    HOMEPAGE_URL "https://github.com/likooooo/kernels"
    LANGUAGES CXX
)
find_package(py_visualizer REQUIRED)
find_package(type_traist_notebook REQUIRED)

add_subdirectory(tests)