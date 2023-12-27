cmake_minimum_required(VERSION 3.0)

#add_definitions(-DUnifiedMem) #use Unified Mem

project (DNN LANGUAGES CUDA CXX)
aux_source_directory(. SRC)
add_subdirectory(src)
add_subdirectory(src/layer_cuda)

add_executable(demo testcuda.cu)
target_link_libraries(demo lib)
target_link_libraries(demo cuda_lib)