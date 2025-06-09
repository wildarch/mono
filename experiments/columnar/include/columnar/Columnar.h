#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "columnar/Dialect.h.inc"

#include "columnar/Enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "columnar/Attrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "columnar/Types.h.inc"

namespace mlir {

class RewriterBase;

} // namespace mlir

namespace columnar {

template <typename ConcreteType>
class IsProjection
    : public mlir::OpTrait::TraitBase<ConcreteType, IsProjection> {};

template <typename ConcreteType>
class Source : public mlir::OpTrait::TraitBase<ConcreteType, Source> {};

template <typename ConcreteType>
class Sink : public mlir::OpTrait::TraitBase<ConcreteType, Sink> {};

struct LowerBodyCtx {
  const mlir::TypeConverter &typeConverter;
  mlir::Value pipelineCtx;
  mlir::ValueRange globals;
  mlir::ValueRange operands;
  llvm::SmallVector<mlir::Value> results;
  llvm::SmallVector<mlir::Value> haveMore;
};

#include "columnar/Interfaces.h.inc"

} // namespace columnar

#define GET_OP_CLASSES
#include "columnar/Ops.h.inc"

namespace columnar {

#define GEN_PASS_DECL
#include "columnar/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "columnar/Passes.h.inc"

/** Element type for columns produced by a COUNT aggregator. */
mlir::Type getCountElementType(mlir::MLIRContext *ctx);

void registerLLVMTranslation(mlir::DialectRegistry &registry);

} // namespace columnar
