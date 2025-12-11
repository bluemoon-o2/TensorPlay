cmake_minimum_required(VERSION 3.18)
project(stax LANGUAGES CXX)

add_library(stax SHARED src/Init.cpp)

target_include_directories(stax PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(stax PRIVATE STAX_EXPORTS)

target_link_libraries(stax PUBLIC p10)

install(TARGETS stax
    RUNTIME DESTINATION tensorplay/lib
    LIBRARY DESTINATION tensorplay/lib
    ARCHIVE DESTINATION tensorplay/lib
)
