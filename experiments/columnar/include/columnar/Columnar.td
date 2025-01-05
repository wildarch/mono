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
    let hasConstantMaterializer = 1;
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
        , SameBuildabilityAs<type, "ColumnType::get(" # type.builderCall # ")"> {
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

def Select : Columnar_Type<"Select", "sel"> {
    let summary = "Index type for selection vectors";
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
        I64EnumAttrCase<"COUNT_ALL", 4>,
        I64EnumAttrCase<"MIN", 5>,
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
        I64EnumAttrCase<"NE", 1>,
        I64EnumAttrCase<"LT", 2>,
        I64EnumAttrCase<"LE", 3>,
        I64EnumAttrCase<"GT", 4>,
        I64EnumAttrCase<"GE", 5>,
    ]> {
    let cppNamespace = "::columnar";
}

def SortDirection : I64EnumAttr<
    "SortDirection", "",
    [
        I64EnumAttrCase<"ASC", 0>,
        I64EnumAttrCase<"DESC", 1>,
    ]> {
    let cppNamespace = "::columnar";
}

def SortDirectionArray : ArrayOfAttr<
        Columnar_Dialect,
        "SortDirectionArray",
        "sort_directions",
        "SortDirection">;

def Table : Columnar_Attr<"Table", "table"> {
    let parameters = (ins StringRefParameter<>:$name);
    
    let assemblyFormat = [{
        `<` $name `>`
    }];
}

def TableColumn : Columnar_Attr<"TableColumn", "table_col"> {
    let parameters = (ins 
        "TableAttr":$table,
        StringRefParameter<>:$name, 
        "mlir::Type":$type);

    let assemblyFormat = [{
        `<` $table $name `:` $type `>`
    }];
}

def DecimalAttr : Columnar_Attr<"Decimal", "dec", [TypedAttrInterface]> {
    let parameters = (ins "std::int64_t":$value);

    let assemblyFormat = [{
        $value
    }];

    let extraClassDeclaration = [{
        mlir::Type getType();
    }];
}

def StringAttr : Columnar_Attr<"String", "str", [TypedAttrInterface]> {
    let parameters = (ins "mlir::StringAttr":$value);

    let assemblyFormat = "$value";

    let extraClassDeclaration = [{
        mlir::Type getType();
    }];
}

def DateAttr : Columnar_Attr<"Date", "date", [TypedAttrInterface]> {
    let parameters = (ins "int":$year, "int":$month, "int":$day);

    let assemblyFormat = "$year $month $day";

    let extraClassDeclaration = [{
        mlir::Type getType();
    }];

    // TODO: verify
}

def SelIdAttr : Columnar_Attr<"SelId", "sel_id", [TypedAttrInterface]> {
    let extraClassDeclaration = [{
        mlir::Type getType();
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
        AttrSizedOperandSegments, 
        ParentOneOf<["QueryOp", "SubQueryOp"]>]> {
    let summary = "Outputs the final query results";

    let arguments = (ins
        Variadic<Column>:$columns,
        StringArray:$names,
        Optional<ColumnOf<Select>>:$sel);
    let builders = [
        OpBuilder<(ins 
            "mlir::ValueRange":$columns, 
            "llvm::ArrayRef<mlir::StringAttr>":$names), 
            [{
                build($_builder, $_state, columns, names, /*sel=*/nullptr);
            }]>,
    ];

    let assemblyFormat = [{
        $columns `:` type($columns)
        $names
        (`sel` `` `=` `` $sel^)?
        attr-dict
    }];

    let hasVerifier = 1;
}

// ============================================================================
// ================================= CORE OPS =================================
// ============================================================================

def ReadTableOp : Columnar_Op<"read_table", [Pure]> {
    let summary = "Reads a column from a table";

    let arguments = (ins TableColumn:$column);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $column `:` type($result) attr-dict
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

    let extraClassDeclaration = [{
        mlir::ValueRange getGroupByResults();
        mlir::ValueRange getAggregationResults();
    }];
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

    let extraClassDeclaration = [{
        mlir::ValueRange getLhsResults();
        mlir::ValueRange getRhsResults();
    }];

    let hasCanonicalizer = 1;
}

def SelectOp : Columnar_Op<"select", [
        Pure, 
        IsolatedFromAbove,
        NoTerminator,
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

    let hasFolder = 1;
    let hasCanonicalizer = 1;
    // TODO: region verify
}

def PredicateOp : Columnar_Op<"pred", [IsolatedFromAbove, HasParent<"SelectOp">]> {
    let arguments = (ins Variadic<Column>:$inputs);

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{ 
        $inputs `:` type($inputs)
        $body  
        attr-dict 
    }];

    let hasFolder = 1;
}

def PredicateEvalOp : Columnar_Op<"pred.eval", [
        Pure, 
        Terminator,
        HasParent<"PredicateOp">]> {
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

def OrderByOp : Columnar_Op<"order_by", [
        Pure, 
        AttrSizedOperandSegments, 
        InferTypeOpAdaptor]> {
    let arguments = (ins 
        Variadic<Column>:$keys, 
        SortDirectionArray:$dirs,
        Variadic<Column>:$values);
    
    let results = (outs Variadic<Column>:$results);

    let assemblyFormat = [{
        $keys `:` type($keys) $dirs $values `:` type($values) attr-dict
    }];

    let extraClassDeclaration = [{
        mlir::ValueRange getKeyResults();
        mlir::ValueRange getValueResults();
    }];
}

def LimitOp : Columnar_Op<"limit", [
        Pure, 
        AllTypesMatch<["inputs", "results"]>, 
        InferTypeOpAdaptor]> {
    let arguments = (ins 
        I64Attr:$limit,
        Variadic<Column>:$inputs);

    let results = (outs Variadic<Column>:$results);

    let assemblyFormat = [{
        $limit $inputs `:` type($inputs) attr-dict
    }];
}

// ============================================================================
// ================================ PROJECTIONS ===============================
// ============================================================================

def CmpOp : Columnar_Op<"cmp", [
        Pure, 
        IsProjection, 
        AllTypesMatch<["lhs", "rhs"]>]> {
    let arguments = (ins 
        CmpPredicate:$pred, 
        Column:$lhs, 
        Column:$rhs,
        Optional<ColumnOf<Select>>:$sel);
    let builders = [
        OpBuilder<(ins 
            "CmpPredicate":$pred, 
            "mlir::Value":$lhs, 
            "mlir::Value":$rhs), [{
                build($_builder, $_state, pred, lhs, rhs, /*sel=*/nullptr);
            }]>,
    ];

    let results = (outs ColumnOf<I1>:$result);

    let assemblyFormat = [{
        $pred $lhs `,` $rhs `:` type($lhs) 
        (`sel` `` `=` `` $sel^)?
        attr-dict
    }];
}

def CastOp : Columnar_Op<"cast", [Pure, IsProjection]> {
    let arguments = (ins Column:$input);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) attr-dict
    }];

    // TODO: verify cast is valid
    let hasFolder = 1;
}

def AndOp : Columnar_Op<"and", [Pure, IsProjection]> {
    let arguments = (ins Variadic<ColumnOf<I1>>:$inputs);

    let results = (outs ColumnOf<I1>:$result);

    let assemblyFormat = [{
        $inputs attr-dict
    }];
}

def NotOp : Columnar_Op<"not", [Pure, IsProjection]> {
    let arguments = (ins ColumnOf<I1>:$input);

    let results = (outs ColumnOf<I1>:$result);

    let assemblyFormat = [{
        $input attr-dict
    }];
}

def Numeric : AnyTypeOf<[
    ColumnOf<SI64>,
    ColumnOf<Decimal>,
    ColumnOf<F64>,
]>;

class NumericBinOp<string mnemonic> : Columnar_Op<mnemonic, [
        Pure,
        IsProjection,
        AllTypesMatch<["lhs", "rhs", "result"]>]> {
    let arguments = (ins 
        Numeric:$lhs, 
        Numeric:$rhs,
        Optional<ColumnOf<Select>>:$sel);
    let builders = [
        OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs), [{
            build($_builder, $_state, lhs, rhs, /*sel=*/nullptr);
        }]>,
    ];

    let results = (outs Numeric:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs `:` type($result) 
        (`sel` `` `=` `` $sel^)? 
        attr-dict
    }];
}

def AddOp : NumericBinOp<"add">;
def SubOp : NumericBinOp<"sub">;
def MulOp : NumericBinOp<"mul">;
def DivOp : NumericBinOp<"div">;

def LikeOp : Columnar_Op<"like", [Pure, IsProjection]> {
    let arguments = (ins
        ColumnOf<String>:$input,
        ColumnOf<String>:$pattern);

    let results = (outs ColumnOf<I1>:$result);

    let assemblyFormat = [{
        $input `like` $pattern attr-dict
    }];
}

def SubQueryOp : Columnar_Op<"sub_query", [Pure, IsolatedFromAbove]> {
    let arguments = (ins Variadic<Column>:$inputs);

    let regions = (region SizedRegion<1>:$body);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $inputs `:` type($inputs) `->` type($result) $body attr-dict
    }];

    let hasCanonicalizer = 1;
}

// ============================================================================
// ======================== EXPLICIT SELECTION VECTORS ========================
// ============================================================================

def SelAddOp : Columnar_Op<"sel.add", [
        Pure, 
        AllTypesMatch<["inputs", "results"]>,
        InferTypeOpAdaptor]> {
    let arguments = (ins Variadic<Column>:$inputs);

    let results = (outs Variadic<Column>:$results, ColumnOf<Select>:$sel);
}

def SelApplyOp : Columnar_Op<"sel.apply", [
        Pure,
        AllTypesMatch<["input", "result"]>]> {
    let summary = "Removes all input rows not in the selection vector";

    let arguments = (ins Column:$input, ColumnOf<Select>:$sel);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $input `:` type($input) `sel` `` `=` `` $sel attr-dict
    }];
}

def SelFilterOp : Columnar_Op<"sel.filter", [Pure]> {
    let summary = "Apply a filter to a selection vector";

    let arguments = (ins 
        ColumnOf<Select>:$inSel, 
        ColumnOf<I1>:$filter,
        ColumnOf<Select>:$filterSel);

    let results = (outs ColumnOf<Select>:$outSel);

    let assemblyFormat = [{
        $inSel `by` $filter `filter_sel` `` `=` `` $filterSel attr-dict
    }];
}

def SelTableOp : Columnar_Op<"sel.table", [Pure, ConstantLike]> {
    let summary = "The selection vector a base table";

    let arguments = (ins Table:$table);

    let results = (outs ColumnOf<Select>:$result);

    let assemblyFormat = [{
        $table attr-dict
    }];

    let hasFolder = 1;
}

// ============================================================================
// =========================== LOWER PIPELINES OPS ============================
// ============================================================================
def PipelineOp : Columnar_Op<"pipeline", [NoTerminator, IsolatedFromAbove]> {
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $body attr-dict
    }];
}

def PrintOp : Columnar_Op<"print"> {
    let arguments = (ins StrAttr:$name, Column:$input, ColumnOf<Select>:$sel);

    let assemblyFormat = [{
        $name 
        $input `:` type($input) 
        `sel` `` `=` `` $sel
        attr-dict
    }];
}

def TensorReadColumnOp : Columnar_Op<"tensor.read_column", [Pure]> {
    let summary = "Reads a column as a tensor";

    let arguments = (ins TableColumn:$column);

    let results = (outs 1DTensorOf<[AnyType]>:$result);

    let assemblyFormat = [{
        $column `:` type($result) attr-dict
    }];
}

def TensorPrintOp : Columnar_Op<"tensor.print"> {
    let arguments = (ins 
        StrAttr:$name, 
        1DTensorOf<[AnyType]>:$input, 
        1DTensorOf<[Index]>:$sel);

    let assemblyFormat = [{
        $name 
        $input `:` type($input) 
        `sel` `` `=` `` $sel `:` type($sel)
        attr-dict
    }];
}

// ============================================================================
// =================================== PASSES =================================
// ============================================================================
def PushDownPredicates: Pass<"push-down-predicates", "::columnar::QueryOp"> {
    let summary = "Push down selection predicates";
}

def AddSelectionVectors: Pass<"add-selection-vectors", "::columnar::QueryOp"> {
    let summary = "Adds explicit selection vectors";
}

def LowerPipelines : Pass<"lower-pipelines", "::columnar::PipelineOp"> {
    let summary = "Lower pipelines to operations over tensors";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",
    ];
}

#endif // COLUMNAR