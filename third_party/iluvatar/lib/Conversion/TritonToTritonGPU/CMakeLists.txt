set(_EXTRA_LINK_LIBS FlagTree_${FLAGTREE_BACKEND}_TritonToTritonGPU)
if(NOT TARGET ${_EXTRA_LINK_LIBS})
  set(_EXTRA_LINK_LIBS "")
endif()

add_triton_library(TritonToTritonGPU
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    ${_EXTRA_LINK_LIBS}
)
