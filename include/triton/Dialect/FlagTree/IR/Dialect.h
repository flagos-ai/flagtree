#ifndef TRITON_DIALECT_FLAGTREE_IR_DIALECT_H_
#define TRITON_DIALECT_FLAGTREE_IR_DIALECT_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "triton/Dialect/FlagTree/IR/Dialect.h.inc"
#include "triton/Dialect/FlagTree/IR/OpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/FlagTree/IR/FlagTreeAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/FlagTree/IR/Ops.h.inc"

#endif
