#ifndef EXPRESSION_DIALECT
#define EXPRESSION_DIALECT

include "mlir/IR/OpBase.td"

def Expression_Dialect : Dialect {
    let name = "expression";
    let summary = "Dialect for scalar expressions in queries.";
    let description = [{
        This dialect is used to encode scalar expressions inside of queries.
    }];
    let cppNamespace = "::execution::expression";

    let useDefaultTypePrinterParser = 1;
}

class Expression_Op<string mnemonic, list<Trait> traits = []> :
        Op<Expression_Dialect, mnemonic, traits>;

#endif // EXPRESSION_DIALECT