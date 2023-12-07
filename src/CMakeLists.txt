aux_source_directory(. SRC_LIB)
aux_source_directory(./layer SRC_LIB)
aux_source_directory(./optimizer SRC_LIB)

add_library(lib ${SRC_LIB})