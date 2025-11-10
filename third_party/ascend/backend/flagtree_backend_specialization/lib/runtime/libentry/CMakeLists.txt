add_triton_library(FlagTree_ascend_libentry
  libentry.cpp
)

target_compile_options(FlagTree_ascend_libentry PRIVATE -fexceptions -frtti)
