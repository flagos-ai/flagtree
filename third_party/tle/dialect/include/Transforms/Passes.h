// Copyright (c) 2025 XCoreSigma Inc. All rights reserved.
// flagtree tle

#ifndef TRITON_TLE_PASSES_H
#define TRITON_TLE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton::tle {

#define GEN_PASS_DECL
#include "tle/dialect/include/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "tle/dialect/include/Transforms/Passes.h.inc"

} // namespace mlir::triton::tle

#endif
