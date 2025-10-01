cmake_minimum_required (VERSION 3.14)
project(kernels 
    DESCRIPTION "type traist notebook"
    HOMEPAGE_URL "https://github.com/likooooo/kernels"
    LANGUAGES CXX
)
find_package(py_visualizer REQUIRED)
find_package(type_traist_notebook REQUIRED)

add_subdirectory(tests)

# 定义库版本
set(PACKAGE_VERSION "1.0.0")

file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")
set(PACKAGE_INCLUDE_INSTALL_DIR include/kernels)
install(FILES ${HEADERS} DESTINATION ${PACKAGE_INCLUDE_INSTALL_DIR})

# 创建并安装配置文件
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/kernels-config.cmake.in"
    "${PROJECT_BINARY_DIR}/kernels-config.cmake"
    @ONLY
)
install(FILES "${PROJECT_BINARY_DIR}/kernels-config.cmake"
        DESTINATION lib/cmake/kernels)

# 安装版本文件
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/kernels-config-version.cmake.in"
    "${PROJECT_BINARY_DIR}/kernels-config-version.cmake"
    @ONLY
)
install(FILES "${PROJECT_BINARY_DIR}/kernels-config-version.cmake"
        DESTINATION lib/cmake/kernels)