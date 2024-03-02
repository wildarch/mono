#ifndef MINI_OPS
#define MINI_OPS

include "MiniDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def ConstantOp : Mini_Op<"constant", [Pure]> {
    let summary = "constant";
    let arguments = (ins I32Attr:$value);
    let results = (outs I32);
}

def FuncOp : Mini_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove, Symbol
]> {
    let summary = "user-defined function";

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
      "mlir::StringRef":$name, "mlir::FunctionType":$type,
      CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)
    >];
    let extraClassDeclaration = [{
      /// Returns the argument types of this function.
      mlir::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }
  
      /// Returns the result types of this function.
      mlir::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
    }];
    let skipDefaultBuilders = 1;
    let hasCustomAssemblyFormat = 1;
}

def ReturnOp : Mini_Op<"return", [
    Pure, HasParent<"FuncOp">, Terminator
]> {
    let summary = "returns from a function";

    let arguments = (ins I32:$input);
}

#endif // MINI_OPS