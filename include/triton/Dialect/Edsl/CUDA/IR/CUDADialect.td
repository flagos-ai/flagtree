#ifndef EDSL_CUDA_DIALECT
#define EDSL_CUDA_DIALECT

include "mlir/IR/OpBase.td"

def CUDA_Dialect : Dialect {
  let name = "CUDA";
  let cppNamespace = "::mlir::triton::edsl::CUDA";
  let description = [{
    CUDA dialect.
  }];

  let dependentDialects = [
    "mlir::gpu::GPUDialect",
  ];
  let usePropertiesForAttributes = 1;
}

#endif
