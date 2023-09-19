#ifndef EXPRESSION_OPS
#define EXPRESSION_OPS

include "execution/expression/IR/ExpressionDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ColumnOp : Expression_Op<"column", [Pure]> {
    let summary = "Reference to a column";
    let description = [{
        A reference to a column in a table.
    }];

    let arguments = (ins I32Attr:$column);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $column attr-dict `:` type($result)
    }];
}

#endif // EXPRESSION_OPS