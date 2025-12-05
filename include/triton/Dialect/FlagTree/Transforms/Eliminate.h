#ifndef TRITON_DIALECT_FLAGTREE_TRANSFORMS_ELIMINATE_H
#define TRITON_DIALECT_FLAGTREE_TRANSFORMS_ELIMINATE_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::flagtree {
void populateEliminatePatterns(RewritePatternSet &patterns);
}

#endif
