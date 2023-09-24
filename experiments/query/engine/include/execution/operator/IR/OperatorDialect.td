#ifndef OPERATOR_DIALECT
#define OPERATOR_DIALECT

include "mlir/IR/OpBase.td"

def Operator_Dialect : Dialect {
    let name = "operator";
    let summary = "Dialect for query operators.";
    let description = [{
        This dialect is used to encode physical query plans.
    }];
    let cppNamespace = "::execution::qoperator";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

class Operator_Op<string mnemonic, list<Trait> traits = []> :
        Op<Operator_Dialect, mnemonic, traits>;

#endif // OPERATOR_DIALECT