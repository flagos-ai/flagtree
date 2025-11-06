set(_EXTRA_LINK_LIBS FlagTree_${FLAGTREE_BACKEND}_TritonDialectTritonGPUIR)
if(NOT TARGET ${_EXTRA_LINK_LIBS})
  set(_EXTRA_LINK_LIBS "")
endif()

add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Types.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
  ${_EXTRA_LINK_LIBS}
)
