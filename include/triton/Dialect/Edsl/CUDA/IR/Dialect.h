#ifndef TRITON_DIALECT_EDSL_CUDA_IR_DIALECT_H_
#define TRITON_DIALECT_EDSL_CUDA_IR_DIALECT_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/Edsl/CUDA/IR/Dialect.h.inc"
#include "triton/Dialect/Edsl/CUDA/IR/OpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/Edsl/CUDA/IR/CUDAAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/Edsl/CUDA/IR/Ops.h.inc"

#endif
