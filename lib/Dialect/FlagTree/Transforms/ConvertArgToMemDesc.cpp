#include "triton/Dialect/FlagTree/Transforms/ConvertArgToMemDesc.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"
#include "triton/Dialect/FlagTree/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

namespace mlir::triton::flagtree {
#define GEN_PASS_DEF_FLAGTREECONVERTARGTOMEMDESC
#include "triton/Dialect/FlagTree/Transforms/Passes.h.inc"
} // namespace mlir::triton::flagtree

namespace {

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace fl = mlir::triton::flagtree;

struct FlagTreeArgConversion : public OpRewritePattern<fl::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

FlagTreeArgConversion::FlagTreeArgConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
FlagTreeArgConversion::matchAndRewrite(fl::DSLRegionOp op,
                                       PatternRewriter &rewriter) const {
  SmallVector<Value> newOperands;
  bool hasConversion = false;
  for (const auto &operand : op->getOperands()) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(operand.getType())) {
      Operation *defOp = operand.getDefiningOp();
      ttg::SwizzledSharedEncodingAttr encoding =
          ttg::SwizzledSharedEncodingAttr::get(
              getContext(), 1, 1, 1, ttg::getOrder(tensorTy),
              ttg::getCTALayout(tensorTy.getEncoding()));
      ttg::MemDescType memDescTy = ttg::MemDescType::get(
          tensorTy.getShape(), tensorTy.getElementType(), encoding,
          ttg::SharedMemorySpaceAttr::get(getContext()), true);
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      ttg::LocalAllocOp allocOp =
          rewriter.create<ttg::LocalAllocOp>(op->getLoc(), memDescTy);
      rewriter.setInsertionPointAfter(op);
      rewriter.create<ttg::LocalDeallocOp>(op->getLoc(), allocOp);
      newOperands.push_back(allocOp);
      hasConversion = true;
    } else {
      newOperands.push_back(operand);
    }
  }
  if (!hasConversion) {
    return failure();
  }
  rewriter.modifyOpInPlace(op, [&]() {
    op->setOperands(newOperands);
    for (auto [blockArg, newOperand] :
         llvm::zip(op.getBody().getArguments(), newOperands)) {
      blockArg.setType(newOperand.getType());
    }
  });
  return success();
}

struct FlagTreeConvertArgToMemDesc
    : public fl::impl::FlagTreeConvertArgToMemDescBase<
          FlagTreeConvertArgToMemDesc> {
  void runOnOperation() override;
};

} // namespace

void mlir::triton::flagtree::populateConvertArgToMemDescPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FlagTreeArgConversion>(patterns.getContext());
}

void FlagTreeConvertArgToMemDesc::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  fl::populateConvertArgToMemDescPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
