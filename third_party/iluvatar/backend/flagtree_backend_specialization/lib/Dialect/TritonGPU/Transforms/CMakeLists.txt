add_triton_library(FlagTree_iluvatar_TritonDialectTritonGPUTransforms
  RemoveLayoutConversions.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
