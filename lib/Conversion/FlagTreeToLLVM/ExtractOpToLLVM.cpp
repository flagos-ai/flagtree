#include "triton/Conversion/FlagTreeToLLVM/ExtractOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/LogicalResult.h"

namespace {

using namespace mlir;
namespace fl = mlir::triton::flagtree;

struct ExtractAllocatedPtrOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractAllocatedPtrOp> {
  ExtractAllocatedPtrOpConversion(LLVMTypeConverter &typeConverter,
                                  PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractAllocatedPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractAlignedPtrOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractAlignedPtrOp> {
  ExtractAlignedPtrOpConversion(LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractAlignedPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractOffsetOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractOffsetOp> {
  ExtractOffsetOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractSizesOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractSizesOp> {
  ExtractSizesOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractSizesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ExtractStridesOpConversion
    : public ConvertOpToLLVMPattern<fl::ExtractStridesOp> {
  ExtractStridesOpConversion(LLVMTypeConverter &typeConverter,
                             PatternBenefit benefit);
  LogicalResult
  matchAndRewrite(fl::ExtractStridesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
}; // namespace

ExtractAllocatedPtrOpConversion::ExtractAllocatedPtrOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractAllocatedPtrOpConversion::matchAndRewrite(
    fl::ExtractAllocatedPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<fl::ExtractAllocatedPtrOp>(
      op, op->getResultTypes(), adaptor.getOperands());
  return success();
}

ExtractAlignedPtrOpConversion::ExtractAlignedPtrOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractAlignedPtrOpConversion::matchAndRewrite(
    fl::ExtractAlignedPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<fl::ExtractAlignedPtrOp>(op, op->getResultTypes(),
                                                       adaptor.getOperands());
  return success();
}

ExtractOffsetOpConversion::ExtractOffsetOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractOffsetOpConversion::matchAndRewrite(
    fl::ExtractOffsetOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<fl::ExtractOffsetOp>(op, op->getResultTypes(),
                                                   adaptor.getOperands());
  return success();
}

ExtractSizesOpConversion::ExtractSizesOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractSizesOpConversion::matchAndRewrite(
    fl::ExtractSizesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<fl::ExtractSizesOp>(op, op->getResultTypes(),
                                                  adaptor.getOperands());
  return success();
}

ExtractStridesOpConversion::ExtractStridesOpConversion(
    LLVMTypeConverter &typeConverter, PatternBenefit benefit)
    : ConvertOpToLLVMPattern(typeConverter, benefit) {}

LogicalResult ExtractStridesOpConversion::matchAndRewrite(
    fl::ExtractStridesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<fl::ExtractStridesOp>(op, op->getResultTypes(),
                                                    adaptor.getOperands());
  return success();
}

void fl::populateExtractOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit) {
  patterns.add<ExtractAllocatedPtrOpConversion, ExtractAlignedPtrOpConversion,
               ExtractOffsetOpConversion, ExtractSizesOpConversion,
               ExtractStridesOpConversion>(typeConverter, benefit);
}
