#ifndef COLUMNAR
#define COLUMNAR

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"

def Columnar_Dialect : Dialect {
    let name = "columnar";
    let cppNamespace = "::columnar";

    let useDefaultTypePrinterParser = 1;
    let usePropertiesForAttributes = 1;
    let useDefaultAttributePrinterParser = 1;
}

class Columnar_Op<string mnemonic, list<Trait> traits = []> :
    Op<Columnar_Dialect, mnemonic, traits>;

class Columnar_Type<string name, string typeMnemonic, list<Trait> traits = []> : 
    TypeDef<Columnar_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

class Columnar_Attr<string name, string attrMnemonic, list<Trait> traits = []>
        : AttrDef<Columnar_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def Column : Columnar_Type<"Column", "col"> {
    let parameters = (ins TypeParameter<"::mlir::Type", "element type">:$elementType);

    let assemblyFormat = [{
        `<` $elementType `>`
    }];
}

// Attributes
def StringArray : ArrayOfAttr<
        Columnar_Dialect,
        "StringArray",
        "strings",
        "::mlir::StringAttr">;

def Aggregator : I64EnumAttr<
    "Aggregator", "",
    [
        I64EnumAttrCase<"SUM", 0>,
        I64EnumAttrCase<"AVG", 1>,
        I64EnumAttrCase<"COUNT", 2>,
        I64EnumAttrCase<"COUNT_DISTINCT", 3>,
        I64EnumAttrCase<"MIN", 4>,
    ]> {
    let cppNamespace = "::columnar";
}

def AggregatorArray : ArrayOfAttr<
        Columnar_Dialect,
        "AggregatorArray",
        "aggregators",
        "Aggregator">;

def QueryOp : Columnar_Op<"query", [IsolatedFromAbove]> {
    let summary = "Represents a full query";

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $body attr-dict
    }];
}

def QueryOutputOp : Columnar_Op<"query.output", [
        Terminator,
        HasParent<"QueryOp">]> {
    let summary = "Outputs the final query results";

    let arguments = (ins
        Variadic<Column>:$columns,
        StringArray:$names);

    let assemblyFormat = [{
        $columns `:` type($columns)
        $names
        attr-dict
    }];

    let hasVerifier = 1;
}

// ============================================================================
// ================================= CORE OPS =================================
// ============================================================================

// TODO: update to read columns from a table format instead.
def ReadLinesOp : Columnar_Op<"read_lines", [Pure]> {
    let summary = "Reads lines from a file";

    let arguments = (ins StrAttr:$path);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $path `:` type($result) attr-dict
    }];
}

def AggregateOp : Columnar_Op<"aggregate", [
        Pure, 
        AttrSizedOperandSegments, 
        InferTypeOpAdaptor]> {
    let arguments = (ins 
        Variadic<Column>:$groupBy, 
        Variadic<Column>:$aggregate,
        AggregatorArray:$aggregators);
    
    let results = (outs Variadic<Column>:$results);

    let assemblyFormat = [{
        (`group` $groupBy^ `:` type($groupBy))?
        (`aggregate` $aggregate^ `:` type($aggregate))?
        $aggregators
        attr-dict
    }];

    let hasVerifier = 1;
} 

// TODO: ConstantOp

def JoinOp : Columnar_Op<"join", [
        Pure, 
        AttrSizedOperandSegments, 
        InferTypeOpAdaptor]> {
    let summary = "Cartesian product of two sets of columns";

    let arguments = (ins
        Variadic<Column>:$lhs,
        Variadic<Column>:$rhs);
    
    let results = (outs Variadic<Column>:$results);

    let assemblyFormat = [{
        ` ` `(` $lhs `)` `(` $rhs `)`
        `:` `(` type($lhs) `)` `(` type($rhs) `)`
        attr-dict
    }];
}

#endif // COLUMNAR