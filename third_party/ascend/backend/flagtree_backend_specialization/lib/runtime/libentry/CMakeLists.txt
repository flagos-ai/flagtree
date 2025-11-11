add_triton_library(FlagTree_ascend_libentry
  libentry.cpp
)

target_compile_options(FlagTree_ascend_libentry PRIVATE -fexceptions -frtti)

find_package(Python3 REQUIRED COMPONENTS Development)
target_link_libraries(FlagTree_ascend_libentry PRIVATE Python3::Python)
