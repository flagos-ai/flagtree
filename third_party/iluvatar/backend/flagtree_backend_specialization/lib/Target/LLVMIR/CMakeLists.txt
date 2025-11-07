add_triton_library(FlagTree_iluvatar_TritonTargetLLVMIR
  LLVMDIScope.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
