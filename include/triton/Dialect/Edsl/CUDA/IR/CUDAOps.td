#ifndef EDSL_CUDA_OPS
#define EDSL_CUDA_OPS

include "triton/Dialect/Edsl/CUDA/IR/CUDADialect.td"
include "triton/Dialect/Edsl/CUDA/IR/CUDAAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

class CUDA_Op<string mnemonic, list<Trait> traits = []> :
    Op<CUDA_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def CUDA_GetThreadIdOp : CUDA_Op<"get_thread_id", [Pure]> {
  let arguments = (ins CUDA_ThreadDim:$axis);
  let results = (outs I32:$result);
  let assemblyFormat = "$axis attr-dict `:` type($result)";
}

#endif
