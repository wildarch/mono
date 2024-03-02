// Mini dialect

#ifndef MINI_DIALECT
#define MINI_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Mini_Dialect : Dialect {
    let name = "mini";
    let cppNamespace = "::experiments_mlir::mini";
    let useFoldAPI = kEmitFoldAdaptorFolder;
}

class Mini_Op<string mnemonic, list<Trait> traits = []> :
        Op<Mini_Dialect, mnemonic, traits>;

#endif // MINI_DIALECT