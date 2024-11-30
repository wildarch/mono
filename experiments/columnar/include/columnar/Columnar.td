#ifndef COLUMNAR
#define COLUMNAR

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"
include "mlir/Pass/PassBase.td"

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

// TODO: define as a ContainerType<..>
def Column : Columnar_Type<"Column", "col"> {
    let parameters = (ins TypeParameter<"::mlir::Type", "element type">:$elementType);

    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$type), [{
            return Base::get(type.getContext(), type);
        }]>,
    ];

    let assemblyFormat = [{
        `<` $elementType `>`
    }];
}

class ColumnOf<Type type> 
        : ConfinedType<Column, [
            SubstLeaves<"$_self", "::llvm::cast<ColumnType>($_self).getElementType()", type.predicate>],
            "column type with " # type.summary # " elements",
            "ColumnType">
        , SameBuildabilityAs<type, "ColumnType::get($_builder.get" # type # "Type())"> {
    Type elementType = type;
}

def String : Columnar_Type<"String", "str"> {
    let summary = "Variable-length string";
}

def Decimal : Columnar_Type<"Decimal", "dec"> {
    let summary = "Represents values between -9,999,999,999.99 and +9,999,999,999.99 in increments of 0.01";
}

def Date : Columnar_Type<"Date", "date"> {
    let summary = "YYYY-MM-DD style date";
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

def CmpPredicate : I64EnumAttr<
    "CmpPredicate", "",
    [
        I64EnumAttrCase<"EQ", 0>,
        I64EnumAttrCase<"LT", 1>,
    ]> {
    let cppNamespace = "::columnar";
}

def TableColumn : Columnar_Attr<"TableColumn", "table_col"> {
    let parameters = (ins StringRefParameter<>:$name, "mlir::Type":$type);

    let assemblyFormat = [{
        `<`  $name `:` $type `>`
    }];
}

def Table : Columnar_Attr<"Table", "table"> {
    let parameters = (ins 
            StringRefParameter<>:$name, 
            OptionalArrayRefParameter<"TableColumnAttr">:$columns);
    
    let assemblyFormat = [{
        `<` 
        `name` `` `=` `` $name
        `columns` `` `=` `` `[` (`]`) : ($columns^ `]`)?
        `>`
    }];
}

// Interfaces and Traits
def IsProjection : NativeOpTrait<"IsProjection"> {
    let cppNamespace = "::columnar";
}

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

def ReadTableOp : Columnar_Op<"read_table", [Pure]> {
    let summary = "Reads a column from a table";

    let arguments = (ins StrAttr:$table, StrAttr:$column);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $table $column `:` type($result) attr-dict
    }];
}

def ConstantOp : Columnar_Op<"constant", [Pure, ConstantLike, InferTypeOpAdaptor]> {
    let arguments = (ins TypedAttrInterface:$value);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $value attr-dict
    }];

    let hasFolder = 1;
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

def SelectOp : Columnar_Op<"select", [
        Pure, 
        IsolatedFromAbove,
        AllTypesMatch<["inputs", "results"]>]> {
    let arguments = (ins Variadic<Column>:$inputs);

    let builders = [OpBuilder<(ins "mlir::ValueRange":$inputs)>];

    let regions = (region AnyRegion:$predicates);

    let results = (outs Variadic<Column>:$results);

    let assemblyFormat = [{
        $inputs `:` type($inputs)
        $predicates 
        attr-dict
    }];

    let extraClassDeclaration = [{
        mlir::Block& addPredicate();
    }];

    let hasFolder = 1;
    // TODO: region verify
}

def SelectReturnOp : Columnar_Op<"select.return", [Pure, Terminator]> {
    let arguments = (ins ColumnOf<I1>:$cond);

    let assemblyFormat = [{
        $cond attr-dict
    }];
}

def UnionOp : Columnar_Op<"union", [
        Pure, 
        SameVariadicOperandSize,
        AllTypesMatch<["lhs", "rhs", "results"]>]> {
    let arguments = (ins Variadic<Column>:$lhs, Variadic<Column>:$rhs);

    let results = (outs Variadic<Column>:$results);

    let assemblyFormat = [{
        ` ` `(` $lhs `)` `(` $rhs `)` `:` type($results) attr-dict
    }];
}

// ============================================================================
// ================================ PROJECTIONS ===============================
// ============================================================================

def CmpOp : Columnar_Op<"cmp", [
        Pure, 
        IsProjection, 
        AllTypesMatch<["lhs", "rhs"]>]> {
    let arguments = (ins CmpPredicate:$pred, Column:$lhs, Column:$rhs);

    let results = (outs ColumnOf<I1>:$result);

    let assemblyFormat = [{
        $pred $lhs `,` $rhs `:` type($lhs) attr-dict
    }];
}

def CastOp : Columnar_Op<"cast", [Pure]> {
    let arguments = (ins Column:$input);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];

    // TODO: verify cast is valid
    let hasFolder = 1;
}

// ============================================================================
// =================================== PASSES =================================
// ============================================================================
def PushDownPredicates: Pass<"push-down-predicates", "::columnar::QueryOp"> {
    let summary = "Push down selection predicates";

    let dependentDialects = [
        "columnar::ColumnarDialect",
    ];
}


#endif // COLUMNAR