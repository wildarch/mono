#ifndef OPERATOR_OPS
#define OPERATOR_OPS

include "execution/operator/IR/OperatorDialect.td"
include "execution/operator/IR/OperatorTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ScanParquetOp : Operator_Op<"scan_parquet", [Pure]> {
    let summary = "Scan a parquet file";
    let description = [{
        Scan a table stored in a parquet file
    }];

    let arguments = (ins StrAttr:$path);
    let results = (outs Operator_Table:$result);

    let assemblyFormat = [{
        $path attr-dict `:` type($result)
    }];
}

#endif // OPERATOR_OPS