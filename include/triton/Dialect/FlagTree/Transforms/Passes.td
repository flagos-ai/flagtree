#ifndef FLAGTREE_PASSES
#define FLAGTREE_PASSES

include "mlir/Pass/PassBase.td"

def FlagTreeConvertArgToMemDesc : Pass<"flagtree-convert-arg-to-memdesc", "mlir::ModuleOp"> {}
def FlagTreeEliminate : Pass<"flagtree-eliminate", "mlir::ModuleOp"> {}

#endif
