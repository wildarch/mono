#ifndef OPERATOR_TYPES
#define OPERATOR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "execution/operator/IR/OperatorDialect.td"

//===----------------------------------------------------------------------===//
// Standalone type definitions
//===----------------------------------------------------------------------===//

class Operator_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Operator_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Operator_Table : Operator_Type<"Table", "table"> {
    let summary = "Table with a fixed schema";
    let description = "A table (persistent or intermediate) with a fixed schema and a variable number of columns.";
    let parameters = (ins ArrayRefParameter<"mlir::Type">:$columns);
    let assemblyFormat = "`table` `<` $columns `>`";
}

#endif // OPERATOR_TYPES
