find_package(mekil REQUIRED)

include_directories(../include)
file(GLOB tools ${CMAKE_CURRENT_LIST_DIR}/test_*.cpp)
foreach(tool IN LISTS tools)
    get_filename_component(test_WLE ${tool} NAME_WLE)
    add_executable(${test_WLE} ${tool})
    target_link_libraries(${test_WLE} PUBLIC py_visualizer::py_visualizer)
    
    set(test_WLE _${test_WLE})
    add_library(${test_WLE} SHARED ${tool})
    target_link_libraries(${test_WLE} PUBLIC py_visualizer::py_visualizer)
endforeach()
