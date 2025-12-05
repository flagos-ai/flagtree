add_triton_library(FlagTreeTransforms
  ConvertArgToMemDesc.cpp
  Eliminate.cpp

  DEPENDS
  TritonIR
  TritonGPUIR
  GluonIR
  FlagTreeIR
  FlagTreeTransformsIncGen
)