
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUEARLYASSIGNMEMORYSPACE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-early-assign-memory-space"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {}

class EarlyAssignMemorySpacePass
    : public impl::TritonGPUEarlyAssignMemorySpaceBase<
          EarlyAssignMemorySpacePass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m.getContext());
    m.walk([&, this](Operation *op) {
      // if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      auto loadOp = op;
      if (loadOp->getNumResults() == 1) {
        auto loadValue = loadOp->getResult(0);
        auto memorySpaceAttr = llvm::cast_if_present<StringAttr>(
            loadOp->getAttr("tt.memory_space"));
        if (isa<RankedTensorType>(loadValue.getType()) && memorySpaceAttr &&
            memorySpaceAttr.getValue() == "shared_memory") {
          // Replace the load with a local alloc + local load
          builder.setInsertionPointAfter(loadOp);
          auto localAlloc = createLocalAllocForLoad(builder, loadValue);
          auto localLoad = createLocalLoad(builder, loadValue, localAlloc);
          loadOp->replaceUsesWithIf(localLoad, [&](OpOperand &use) {
            return use.getOwner() != localAlloc;
          });
          loadOp->removeAttr("tt.memory_space");
        }
      }
    });
  }

  triton::gpu::LocalAllocOp createLocalAllocForLoad(OpBuilder &builder,
                                                    Value loadOp) {
    auto loc = loadOp.getLoc();
    auto type = llvm::cast<RankedTensorType>(loadOp.getType());
    auto order = triton::gpu::getOrder(type);
    auto ctaLayout = triton::gpu::getCTALayout(type.getEncoding());
    auto sharedEncoding = triton::gpu::SwizzledSharedEncodingAttr::get(
        builder.getContext(), 1, 1, 1, order, ctaLayout);
    auto sharedMemSpace =
        triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
    auto memDescType = triton::gpu::MemDescType::get(
        type.getShape(), type.getElementType(), sharedEncoding, sharedMemSpace);

    auto allocOp =
        builder.create<triton::gpu::LocalAllocOp>(loc, memDescType, loadOp);
    return allocOp;
  }

  triton::gpu::LocalLoadOp createLocalLoad(OpBuilder &builder, Value loadOp,
                                           Value localAllocOp) {
    auto loc = loadOp.getLoc();
    auto type = llvm::cast<RankedTensorType>(loadOp.getType());

    auto localLoadOp =
        builder.create<triton::gpu::LocalLoadOp>(loc, type, localAllocOp);
    return localLoadOp;
  }
};
} // namespace mlir::triton::gpu
