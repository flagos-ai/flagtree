if (FLAGTREE_BACKEND)
    set(NVGPUIR "")
else()
    set(NVGPUIR "TritonNvidiaGPUIR")
endif()

set(_EXTRA_LINK_LIBS FlagTree_${FLAGTREE_BACKEND}_TritonAnalysis)
if(NOT TARGET ${_EXTRA_LINK_LIBS})
  set(_EXTRA_LINK_LIBS "")
endif()

add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  ${NVGPUIR}
  ${_EXTRA_LINK_LIBS}
)
