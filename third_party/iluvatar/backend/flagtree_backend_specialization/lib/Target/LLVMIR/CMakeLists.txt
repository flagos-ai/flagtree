add_triton_library(FlagTree_iluvatar_TritonTargetLLVMIR
  Combine.cpp
  RewriteTensorPointer.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
