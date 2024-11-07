#ifndef COLUMNAR
#define COLUMNAR

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"

def Columnar_Dialect : Dialect {
  let name = "columnar";
  let cppNamespace = "::columnar";

  let useDefaultTypePrinterParser = 1;

}

class Columnar_Op<string mnemonic, list<Trait> traits = []> :
    Op<Columnar_Dialect, mnemonic, traits>;

class Columnar_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Columnar_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Column : Columnar_Type<"Column", "col"> {
    let parameters = (ins TypeParameter<"::mlir::Type", "element type">:$elementType);

    let assemblyFormat = [{
        `<` $elementType `>`
    }];
}

def ReadLinesOp : Columnar_Op<"read_lines"> {
    let summary = "Reads lines from a file";

    let arguments = (ins StrAttr:$path);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $path `:` type($result) attr-dict
    }];
}

#endif // COLUMNAR