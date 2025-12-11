cmake_minimum_required(VERSION 3.18)
project(nn LANGUAGES CXX)

add_library(nn SHARED src/Init.cpp)

target_include_directories(nn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(nn PRIVATE NN_EXPORTS)

target_link_libraries(nn PUBLIC tpx)

install(TARGETS nn
    RUNTIME DESTINATION tensorplay/lib
    LIBRARY DESTINATION tensorplay/lib
    ARCHIVE DESTINATION tensorplay/lib
)
