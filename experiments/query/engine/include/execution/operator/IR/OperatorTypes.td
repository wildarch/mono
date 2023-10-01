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

def Aggregator : Operator_Type<"Aggregator", "aggregator"> {
    let summary = "Aggregation function";
    let description = "Output column configuration for an aggregation operators. Either a key column or an monoid";
    let parameters = (ins "mlir::Type":$elementType);
    let assemblyFormat = "`aggregator` `<` $elementType `>`";
}

def Varchar : Operator_Type<"Varchar", "varchar"> {
    let summary = "string";
}

#endif // OPERATOR_TYPES
