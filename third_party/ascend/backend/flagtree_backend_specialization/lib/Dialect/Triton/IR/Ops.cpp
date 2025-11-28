#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {

//-- SortOp --
LogicalResult SortOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
    {
    if (operands.size() != 1) {
        return emitOptionalError(location, "expected exactly one operand for SortOp");
    }

    if (!isa<RankedTensorType>(operands[0].getType())) {
        return emitOptionalError(location, "operand must be a ranked tensor type for SortOp");
    }

    Value src = operands[0];
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcShape = srcTy.getShape();
    auto srcEnc = srcTy.getEncoding();

    if (srcShape.empty()) {
    return emitOptionalError(location, "input tensor must have rank >= 1");
    }

    Type sortedTy = RankedTensorType::get(srcShape, srcTy.getElementType(), srcEnc);

    inferredReturnTypes.push_back(sortedTy);

    return success();
}

//-- MakeTensorDescOp --
void MakeTensorDescOp::build(OpBuilder &builder, OperationState &state,
                             Value base, ValueRange shape, ValueRange strides,
                             ArrayRef<int32_t> blockShape,
                             bool isSignedInteger)
{
  auto ptrTy = dyn_cast<triton::PointerType>(base.getType());
  if (!ptrTy) {
    llvm::report_fatal_error("Expected pointer type");
  }
  auto elemTy = ptrTy.getPointeeType();
  SmallVector<int64_t> blockShape64(blockShape);
  auto blockTy = RankedTensorType::get(blockShape64, elemTy);
  auto descTy =
      TensorDescType::get(builder.getContext(), blockTy, isSignedInteger);
  return build(builder, state, descTy, base, shape, strides);
}

// -- DescriptorLoadOp --
static LogicalResult verifyDescriptorLoadStoreType(Operation *op,
                                                   TensorDescType desc,
                                                   RankedTensorType tensor)
{
  RankedTensorType block = desc.getSignlessBlockType();
  ArrayRef<int64_t> blockShape = block.getShape();
  ArrayRef<int64_t> tensorShape = tensor.getShape();
  if (blockShape.size() > tensorShape.size()) {
    // Allow ranked reduced load if the leading dimensions are all 1s.
    for (int i = 0; i < blockShape.size() - tensorShape.size(); ++i) {
      if (blockShape[i] != 1)
        return op->emitOpError(
            "ranked reduce load only allowed for unit dimension leading dim.");
    }
    blockShape = blockShape.take_back(tensorShape.size());
  }

  if (blockShape == tensorShape &&
      block.getElementType() == tensor.getElementType()) {
        return success();
      }
  return op->emitOpError("tensor descriptor block and tensor types must match");
}

LogicalResult DescriptorLoadOp::verify()
{
  return verifyDescriptorLoadStoreType(*this, getDesc().getType(), getType());
}

// -- DescriptorStoreOp --
LogicalResult DescriptorStoreOp::verify()
{
  return verifyDescriptorLoadStoreType(*this, getDesc().getType(),
                                       getSrc().getType());
}

// The following ops, including `call`, `func`, and `return` are copied and
// modified from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Func/IR/FuncOps.cpp
// We could revert it back once MLIR has a better inliner interface.
//-- FuncOp --
void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
#if LLVM_VERSION_MAJOR < 21
  function_interface_impl::addArgAndResultAttrs(
#else  // triton_v3.3.x
  call_interface_impl::addArgAndResultAttrs(
#endif
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

// -- JoinOp --
LogicalResult
JoinOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  // These should have been checked by tablegen-generated code.
  assert(operands.size() == 2);
  assert(operands[0].getType() == operands[1].getType());
  assert(isa<RankedTensorType>(operands[0].getType()));
  assert(isa<RankedTensorType>(operands[1].getType()));

  Value lhs = operands[0];
  // Value rhs = operands[1];
  auto srcTy = cast<RankedTensorType>(lhs.getType());

  SmallVector<int64_t> retShape(srcTy.getShape());
  retShape.push_back(2);

  Attribute srcEnc = srcTy.getEncoding();
  Attribute retEnc;
  if (srcEnc) {
    if (dyn_cast<DialectInferLayoutInterface>(&srcEnc.getDialect())
            ->inferJoinOpEncoding(srcEnc, retEnc, location)
            .failed()) {
      return failure();
    }
  }
  inferredReturnTypes.push_back(
      RankedTensorType::get(retShape, srcTy.getElementType(), retEnc));
  return success();
}

// -- GatherOp --
LogicalResult GatherOp::verify() {
  RankedTensorType indicesTy = getIndices().getType();
  RankedTensorType srcTy = getSrc().getType();
  RankedTensorType resTy = getResult().getType();

  if (indicesTy.getShape() != resTy.getShape()) {
    return emitOpError("indices and output shapes must match");
  }
  if (indicesTy.getEncoding() != resTy.getEncoding()) {
    return emitOpError("indices and output encodings must match");
  }
  if (srcTy.getElementType() != resTy.getElementType()) {
    return emitOpError("input and output element types must match");
  }
  if (srcTy.getRank() != indicesTy.getRank()) {
    return emitOpError("input and indices ranks must match");
  }
  if (getAxis() >= srcTy.getRank()) {
    return emitOpError("gather dimension must be less than the input rank");
  }
  for (int dim = 0; dim < indicesTy.getRank(); ++dim) {
    if (dim == getAxis())
      continue;
    if (indicesTy.getShape()[dim] != srcTy.getShape()[dim]) {
      return emitOpError("indices dimension ")
             << dim << " must match the corresponding input dimension";
    }
  }

  return success();
}

LogicalResult GatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  GatherOpAdaptor adaptor(operands, attributes, properties, regions);
  auto indicesType = cast<RankedTensorType>(adaptor.getIndices().getType());
  auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());

  // Shape and encoding of the indices with the element type of the src.
  inferredReturnTypes.push_back(
      RankedTensorType::get(indicesType.getShape(), srcType.getElementType(),
                            indicesType.getEncoding()));
  return success();
}

} // namespace triton
} // namespace mlir
