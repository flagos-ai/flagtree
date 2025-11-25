add_triton_library(EdslCUDAIR
  Dialect.cpp

  DEPENDS
  EdslCUDATableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
