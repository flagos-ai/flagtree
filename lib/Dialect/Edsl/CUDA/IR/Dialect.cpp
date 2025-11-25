#include "triton/Dialect/Edsl/CUDA/IR/Dialect.h"

#include "mlir/Support/LLVM.h"

#include "triton/Dialect/Edsl/CUDA/IR/Dialect.cpp.inc"
#include "triton/Dialect/Edsl/CUDA/IR/OpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/Edsl/CUDA/IR/CUDAAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/Edsl/CUDA/IR/Ops.cpp.inc"

namespace mlir::triton::edsl::CUDA {
void CUDADialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/Edsl/CUDA/IR/CUDAAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Edsl/CUDA/IR/Ops.cpp.inc"
      >();
}
}
