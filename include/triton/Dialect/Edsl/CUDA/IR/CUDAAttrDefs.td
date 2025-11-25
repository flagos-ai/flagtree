#ifndef EDSL_CUDA_ATTR_DEFS
#define EDSL_CUDA_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Edsl/CUDA/IR/CUDADialect.td"

def CUDA_ThreadDim : I32EnumAttr<
    "ThreadIDDim", "",
    [
        I32EnumAttrCase<"X", 0, "x">,
        I32EnumAttrCase<"Y", 1, "y">,
        I32EnumAttrCase<"Z", 2, "z">,
    ]> {
    let cppNamespace = "::mlir::triton::edsl::CUDA";
}

#endif
