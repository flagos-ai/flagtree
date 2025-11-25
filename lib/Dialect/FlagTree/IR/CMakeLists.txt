add_triton_library(FlagTreeIR
  Dialect.cpp

  DEPENDS
  FlagTreeTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
