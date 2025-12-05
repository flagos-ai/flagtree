#include "triton/Dialect/FlagTree/Transforms/Eliminate.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"
#include "triton/Dialect/FlagTree/Transforms/Eliminate.h"
#include "triton/Dialect/FlagTree/Transforms/Passes.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::triton::flagtree {
#define GEN_PASS_DEF_FLAGTREEELIMINATE
#include "triton/Dialect/FlagTree/Transforms/Passes.h.inc"
} // namespace mlir::triton::flagtree

using namespace mlir;
namespace fl = mlir::triton::flagtree;

namespace {

struct FlagTreeEliminateDSLRegionOp : public OpRewritePattern<fl::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeEliminateDSLRegionOp(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct FlagTreeEliminateExtractAllocatedPtr
    : public OpRewritePattern<fl::ExtractAllocatedPtrOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeEliminateExtractAllocatedPtr(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::ExtractAllocatedPtrOp op,
                                PatternRewriter &rewriter) const override;
};

struct FlagTreeEliminateExtractAlignedPtr
    : public OpRewritePattern<fl::ExtractAlignedPtrOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeEliminateExtractAlignedPtr(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::ExtractAlignedPtrOp op,
                                PatternRewriter &rewriter) const override;
};

struct FlagTreeEliminateExtractOffset
    : public OpRewritePattern<fl::ExtractOffsetOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeEliminateExtractOffset(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::ExtractOffsetOp op,
                                PatternRewriter &rewriter) const override;
};

struct FlagTreeEliminateExtractSizes
    : public OpRewritePattern<fl::ExtractSizesOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeEliminateExtractSizes(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::ExtractSizesOp op,
                                PatternRewriter &rewriter) const override;
};

struct FlagTreeEliminateExtractStrides
    : public OpRewritePattern<fl::ExtractStridesOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeEliminateExtractStrides(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::ExtractStridesOp op,
                                PatternRewriter &rewriter) const override;
};

struct FlagTreeEliminate
    : public fl::impl::FlagTreeEliminateBase<FlagTreeEliminate> {
  void runOnOperation() override;
};

} // namespace

FlagTreeEliminateDSLRegionOp::FlagTreeEliminateDSLRegionOp(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
FlagTreeEliminateDSLRegionOp::matchAndRewrite(fl::DSLRegionOp op,
                                              PatternRewriter &rewriter) const {
  return failure();
}

FlagTreeEliminateExtractAllocatedPtr::FlagTreeEliminateExtractAllocatedPtr(
    MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult FlagTreeEliminateExtractAllocatedPtr::matchAndRewrite(
    fl::ExtractAllocatedPtrOp op, PatternRewriter &rewriter) const {
  if (isa<LLVM::LLVMStructType>(op.getInput().getType())) {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, op.getInput(),
                                                      SmallVector<int64_t>{0});
    return success();
  } else {
    return failure();
  }
}

FlagTreeEliminateExtractAlignedPtr::FlagTreeEliminateExtractAlignedPtr(
    MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult FlagTreeEliminateExtractAlignedPtr::matchAndRewrite(
    fl::ExtractAlignedPtrOp op, PatternRewriter &rewriter) const {
  if (isa<LLVM::LLVMStructType>(op.getInput().getType())) {
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, op.getInput(),
                                                      SmallVector<int64_t>{0});
    return success();
  } else {
    return failure();
  }
}

FlagTreeEliminateExtractOffset::FlagTreeEliminateExtractOffset(
    MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult FlagTreeEliminateExtractOffset::matchAndRewrite(
    fl::ExtractOffsetOp op, PatternRewriter &rewriter) const {
  return failure();
}

FlagTreeEliminateExtractSizes::FlagTreeEliminateExtractSizes(
    MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult FlagTreeEliminateExtractSizes::matchAndRewrite(
    fl::ExtractSizesOp op, PatternRewriter &rewriter) const {
  return failure();
}

FlagTreeEliminateExtractStrides::FlagTreeEliminateExtractStrides(
    MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult FlagTreeEliminateExtractStrides::matchAndRewrite(
    fl::ExtractStridesOp op, PatternRewriter &rewriter) const {
  return failure();
}

void FlagTreeEliminate::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns
      .add<FlagTreeEliminateDSLRegionOp, FlagTreeEliminateExtractAllocatedPtr,
           FlagTreeEliminateExtractAlignedPtr, FlagTreeEliminateExtractSizes,
           FlagTreeEliminateExtractStrides>(&getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
