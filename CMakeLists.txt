cmake_minimum_required(VERSION 3.0)

project (DNN LANGUAGES CXX)
aux_source_directory(. SRC)
add_subdirectory(src)
add_executable(demo ${SRC})
target_link_libraries(demo lib)