#ifndef COLUMNAR
#define COLUMNAR

include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
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

def ByteArray : Columnar_Type<"ByteArray", "byte_array", [
        MemRefElementTypeInterface]> {
    let summary = "Variable-length byte array";
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

def ScannerHandle : Columnar_Type<"ScannerHandle", "scanner_handle"> {
    let summary = "Table scanner handle";
}

def ColumnHandle : Columnar_Type<"ColumnHandle", "column_handle"> {
    let summary = "Table column handle";
}

def PrintChunk : Columnar_Type<"PrintChunk", "print_chunk"> {
    let summary = "Print chunk";
}

def PrintHandle : Columnar_Type<"PrintHandle", "print_handle"> {
    let summary = "Print handle";
}

def PipelineContext : Columnar_Type<"PipelineContext", "pipe_ctx"> {
    let summary = "Pipeline context handle";
}

def Struct : Columnar_Type<"Struct", "struct"> {
    let summary = "Compound type analogous to a C struct";

    let parameters = (ins ArrayRefParameter<"mlir::Type">:$fieldTypes);

    let assemblyFormat = [{
        `<` $fieldTypes `>`
    }];
}

def Ptr : Columnar_Type<"Pointer", "ptr", [MemRefElementTypeInterface]> {
    let parameters = (ins TypeParameter<"::mlir::Type", "pointee type">:$pointee);

    let builders = [
        TypeBuilderWithInferredContext<(ins "Type":$type), [{
            return Base::get(type.getContext(), type);
        }]>,
    ];

    let assemblyFormat = [{
        `<` $pointee `>`
    }];
}

class PtrOf<Type type>
        : ConfinedType<Ptr, [
            SubstLeaves<"$_self", "::llvm::cast<PointerType>($_self).getPointee()", type.predicate>],
            "pointer to " # type.summary,
            "PointerType">
        , SameBuildabilityAs<type, "PointerType::get(" # type.builderCall # ")"> {
    Type elementType = type;
}

def StringLiteral : Columnar_Type<"StringLiteral", "str_lit"> {
    let summary = "String literal";
}

def TupleBuffer : Columnar_Type<"TupleBuffer", "tuple_buffer"> {
    let summary = "Row-oriented buffer for tuple data";

    let parameters = (ins Struct:$tupleType);

    let assemblyFormat = [{ `<` $tupleType `>` }];
}

def TupleBufferLocal : Columnar_Type<"TupleBufferLocal", "tuple_buffer.local"> {
    let summary = "Thread-local tuple buffer";

    let parameters = (ins
        Struct:$tupleType);

    let assemblyFormat = [{
        `<` $tupleType `>`
    }];
}

def HashTable : Columnar_Type<"HashTable", "hash_table"> {
    let summary = "A hash table";

    let parameters = (ins
        ArrayRefParameter<"mlir::Type">:$keyTypes,
        ArrayRefParameter<"mlir::Type">:$valueTypes);

    let assemblyFormat = [{
        `<` `(` $keyTypes `)` `->` `(` $valueTypes `)` `>`
    }];
}

def Allocator : Columnar_Type<"Allocator", "allocator"> {}

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
    let parameters = (ins StringRefParameter<>:$name, StringRefParameter<>:$path);

    let assemblyFormat = [{
        `<` $name `path` `` `=` `` $path `>`
    }];
}

def TableColumn : Columnar_Attr<"TableColumn", "table_col"> {
    let parameters = (ins
        "TableAttr":$table,
        "int":$idx,
        StringRefParameter<>:$name,
        "mlir::Type":$logicalType,
        "mlir::Type":$physicalType);

    let assemblyFormat = [{
        `<` $table $idx $name `:` $logicalType `[` $physicalType `]` `>`
    }];
}

def TableColumnArray : ArrayOfAttr<
        Columnar_Dialect,
        "TableColumnArray",
        "table_columns",
        "TableColumnAttr">;

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

def Source : NativeOpTrait<"Source"> {
    let cppNamespace = "::columnar";
}

def Sink : NativeOpTrait<"Sink"> {
    let cppNamespace = "::columnar";
}

def LowerPipelineOpInterface : OpInterface<"LowerPipelineOpInterface"> {
  let methods = [
    // Global state
    InterfaceMethod<
      "Initialize global state",
      "mlir::LogicalResult",
      "lowerGlobalOpen",
      (ins
        "mlir::OpBuilder&":$builder,
        "llvm::SmallVectorImpl<mlir::Value>&":$newGlobals),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            return mlir::success();
        }]
    >,
    InterfaceMethod<
      "Free global state",
      "mlir::LogicalResult",
      "lowerGlobalClose",
      (ins
        "mlir::OpBuilder&":$builder,
        "mlir::ValueRange":$globals),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            assert (globals.empty() && "leaking globals");
            return mlir::success();
        }]
    >,
    // Local state
    InterfaceMethod<
      "Initialize local state",
      "mlir::LogicalResult",
      "lowerLocalOpen",
      (ins
        "mlir::OpBuilder&":$builder,
        "mlir::ValueRange":$globals,
        "llvm::SmallVectorImpl<mlir::Value>&":$newLocals),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            return mlir::success();
        }]
    >,
    InterfaceMethod<
      "Free local state",
      "mlir::LogicalResult",
      "lowerLocalClose",
      (ins
        "mlir::OpBuilder&":$builder,
        "mlir::ValueRange":$globals,
        "mlir::ValueRange":$locals),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            assert (locals.empty() && "leaking locals");
            return mlir::success();
        }]
    >,
    InterfaceMethod<
      "Lower the body",
      "mlir::LogicalResult",
      "lowerBody",
      (ins "LowerBodyCtx&":$ctx, "mlir::OpBuilder&":$builder)
    >,
  ];
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
        ParentOneOf<["QueryOp", "SubQueryOp", "PipelineOp"]>,
        Sink,
        DeclareOpInterfaceMethods<
            LowerPipelineOpInterface,
            ["lowerGlobalOpen", "lowerGlobalClose"]>]> {
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

def ReadColumnOp : Columnar_Op<"read_column", [Pure, Source]> {
    let summary = "Reads a column from a table";

    let arguments = (ins TableColumn:$column);

    let results = (outs Column:$result);

    let assemblyFormat = [{
        $column `:` type($result) attr-dict
    }];
}

def ConstantOp : Columnar_Op<"constant", [
        Pure,
        ConstantLike,
        InferTypeOpAdaptor,
        Source]> {
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
        InferTypeOpAdaptor,
        Source,
        Sink]> {
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
        InferTypeOpAdaptor,
        Source,
        Sink]> {
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
        AllTypesMatch<["lhs", "rhs", "results"]>,
        Source,
        Sink]> {
    let arguments = (ins Variadic<Column>:$lhs, Variadic<Column>:$rhs);

    let results = (outs Variadic<Column>:$results);

    let assemblyFormat = [{
        ` ` `(` $lhs `)` `(` $rhs `)` `:` type($results) attr-dict
    }];
}

def OrderByOp : Columnar_Op<"order_by", [
        Pure,
        AttrSizedOperandSegments,
        InferTypeOpAdaptor,
        Source,
        Sink]> {
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
        InferTypeOpAdaptor,
        Source,
        Sink]> {
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

def GlobalState : AnyTypeOf<[TupleBuffer, HashTable]>;

def GlobalOp : Columnar_Op<"global", [Symbol]> {
    let summary = "Declare shared state accessible from multiple operators";

    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$globalType);

    let assemblyFormat = [{
        $sym_name $globalType attr-dict
    }];
}

// TODO categorize
def HashJoinCollectOp : Columnar_Op<"hj.collect", [
        AttrSizedOperandSegments,
        DeclareOpInterfaceMethods<
            LowerPipelineOpInterface,
            ["lowerLocalOpen", "lowerLocalClose"]>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Writes tuples into a buffer";
    // TODO: selection vectors
    let arguments = (ins
        Variadic<Column>:$keys,
        Variadic<ColumnOf<Select>>:$keySel,
        Variadic<Column>:$values,
        Variadic<ColumnOf<Select>>:$valueSel,
        FlatSymbolRefAttr:$buffer);

    let assemblyFormat = [{
        `keys` `` `=` `` `[` $keys `]` `:` type($keys)
        `key_sel` `` `=` `` `[` $keySel `]`
        `values` `` `=` `` `[` $values `]` `:` type($values)
        `value_sel` `` `=` `` `[` $valueSel `]`
        `->` $buffer
        attr-dict
    }];

    // TODO: verify types
    // TODO: verify buffer has hash field

    let extraClassDeclaration = [{
        TupleBufferType getBufferType();
    }];
}

def HashJoinBuildOp : Columnar_Op<"hj.build", [Pure]> {
    let summary = "Write entries from a tuple buffer into a hash table";

    let arguments = (ins
        TupleBuffer:$buffer,
        HashTable:$table);

    let assemblyFormat = [{
        $buffer `:` type($buffer)
        `->` $table `:` type($table) attr-dict
    }];
}

def HashJoinProbeOp : Columnar_Op<"hj.probe", [Pure, InferTypeOpAdaptor]> {
    let arguments = (ins
        Variadic<Column>:$keys,
        HashTable:$table);

    let results = (outs
        Variadic<Column>:$tableValues,
        ColumnOf<Select>);

    // TODO: Infer types
    let assemblyFormat = [{
        $keys `:` type($keys)
        `probe` $table `:` type($table)
        attr-dict
    }];
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

def SelTableOp : Columnar_Op<"sel.table", [
        Pure,
        ConstantLike]> {
    let summary = "The selection vector of a base table";

    let arguments = (ins Table:$table);

    let results = (outs ColumnOf<Select>:$result);

    let assemblyFormat = [{
        $table attr-dict
    }];

    let hasFolder = 1;
}

// ============================================================================
// =========================== MAKE PIPELINES OPS =============================
// ============================================================================
def PipelineOp : Columnar_Op<"pipeline", [NoTerminator]> {
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $body attr-dict
    }];
}

// ============================================================================
// ========================== GROUP TABLE READS OPS ===========================
// ============================================================================
def ReadTableOp : Columnar_Op<"read_table", [
        Pure,
        DeclareOpInterfaceMethods<
            LowerPipelineOpInterface,
            ["lowerGlobalOpen", "lowerGlobalClose"]>]> {
    let summary = "Reads a table";

    let arguments = (ins Table:$table, TableColumnArray:$columnsToRead);

    let results = (outs ColumnOf<Select>:$sel, Variadic<Column>:$col);

    let assemblyFormat = [{
        $table $columnsToRead `:` type($col) attr-dict
    }];
}

// ============================================================================
// =========================== LOWER PIPELINES OPS ============================
// ============================================================================

def TensorCol : 1DTensorOf<[AnyType]>;

class TensorColOf<Type elType> :
    1DTensorOf<[elType]>,
    SameBuildabilityAs<
        elType,
        "::mlir::RankedTensorType::get({::mlir::ShapedType::kDynamic}, "
        "$_builder.get" # elType # "Type())"> {
}

def PipelineLowOp : Columnar_Op<"pipeline_low", [IsolatedFromAbove]> {
    let regions = (region
        MaxSizedRegion<1>:$globalOpen,
        MaxSizedRegion<1>:$localOpen,
        SizedRegion<1>:$body,
        MaxSizedRegion<1>:$localClose,
        MaxSizedRegion<1>:$globalClose);

    let assemblyFormat = [{
        (`global_open` $globalOpen^)?
        (`local_open` $localOpen^)?
        `body` $body
        (`local_close` $localClose^)?
        (`global_close` $globalClose^)?
        attr-dict
    }];
}

def PipelineLowYieldOp : Columnar_Op<"pipeline_low.yield", [
        Terminator,
        HasParent<"PipelineLowOp">]> {
    let arguments = (ins Variadic<AnyType>:$values);

    let assemblyFormat = [{
        ($values^ `:` type($values))? attr-dict
    }];
}

def GlobalReadOp : Columnar_Op<"global.read", [
        Pure,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$globalName);

    let results = (outs GlobalState:$result);

    let assemblyFormat = [{
        $globalName `->` type($result) attr-dict
    }];
}

def RuntimeCallOp : Columnar_Op<"runtime_call"> {
    let arguments = (ins StrAttr:$func, Variadic<AnyType>:$inputs);

    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $func `(` $inputs `)` attr-dict `:` functional-type($inputs, $results)
    }];
}

def TableColumnReadOp : Columnar_Op<"table.column.read", [
        DeclareOpInterfaceMethods<
            BufferizableOpInterface,
            ["bufferize", "bufferizesToAllocation"]>]> {
    let summary = "Reads a column chunk as a tensor";

    let arguments = (ins
        PipelineContext:$ctx,
        ColumnHandle:$handle,
        I32:$rowGroup,
        I32:$skip,
        Index:$size);

    let results = (outs TensorCol:$result);

    let assemblyFormat = [{
        $handle `:` type($result)
        `row_group` `` `=` `` $rowGroup
        `skip` `` `=` `` $skip
        `size` `` `=` `` $size
        `ctx` `` `=` `` $ctx
        attr-dict
    }];
}

def PrintChunkAppendOp : Columnar_Op<"print.chunk.append", [
        DeclareOpInterfaceMethods<BufferizableOpInterface, [
            "bufferize",
            "bufferizesToMemoryRead",
            "bufferizesToMemoryWrite",
            "getAliasingValues"]>]> {
    let summary = "Append a column to a chunk";

    let arguments = (ins
        PrintChunk:$chunk,
        TensorCol:$col,
        TensorColOf<Index>:$sel);

    let assemblyFormat = [{
        $chunk
        $col `:` type($col)
        `sel` `` `=` `` $sel
        attr-dict
    }];
}

def HashOp : Columnar_Op<"hash", [
        Pure,
        AllTypesMatch<["base", "result"]>,
        DeclareOpInterfaceMethods<BufferizableOpInterface, [
            "bufferize",
            "bufferizesToAllocation",
            "bufferizesToMemoryRead",
            "bufferizesToMemoryWrite",
            "getAliasingValues"]>]> {
    let arguments = (ins
        TensorColOf<I64>:$base,
        TensorColOf<Index>:$sel,
        TensorCol:$value);

    let results = (outs TensorColOf<I64>:$result);

    let assemblyFormat = [{
        $base `,` $value `[` $sel `]` `:` type($value)
        `->` type($result)
        attr-dict
    }];
}

def TupleBufferInsertOp : Columnar_Op<"tuple_buffer.insert", [
        InferTypeOpAdaptor,
        DeclareOpInterfaceMethods<BufferizableOpInterface, [
            "bufferize",
            "bufferizesToAllocation",
            "bufferizesToMemoryRead",
            "bufferizesToMemoryWrite",
            "getAliasingValues"]>]> {
    let summary = "Insert tuples into a tuple buffer";

    let description = [{
        Reserves memory locations in the buffer for the elements.
        The returned values are indices to the allocated memory locations.
        This op does not write any tuple data to the allocated memory beyond the
        leading hash value.
    }];

    let arguments = (ins
        TupleBufferLocal:$buffer,
        TensorColOf<I64>:$hashes);

    let results = (outs TensorColOf<Ptr>:$result);

    let assemblyFormat = [{
        $buffer `,` $hashes `:` type($buffer) attr-dict
    }];
}

def GetFieldPtrOp : Columnar_Op<"gfp", [Pure, InferTypeOpAdaptor]> {
    let arguments = (ins
        PtrOf<Struct>:$base,
        I64Attr:$field);

    let results = (outs Ptr:$result);

    let assemblyFormat = [{
        $base $field `:` type($base) attr-dict
    }];
}

def TypeSizeOp : Columnar_Op<"type.size", [Pure]> {
    let arguments = (ins TypeAttr:$type);

    let results = (outs I64:$result);

    let assemblyFormat = [{
        $type attr-dict
    }];
}

def TypeAlignOp : Columnar_Op<"type.align", [Pure]> {
    let arguments = (ins TypeAttr:$type);

    let results = (outs I64:$result);

    let assemblyFormat = [{
        $type attr-dict
    }];
}

def ScatterOp : Columnar_Op<"scatter", [
        DeclareOpInterfaceMethods<BufferizableOpInterface, [
            "bufferize",
            "bufferizesToMemoryRead",
            "bufferizesToMemoryWrite",
            "getAliasingValues"]>]> {
    let arguments = (ins
        TensorColOf<Index>:$sel,
        TensorCol:$value,
        TensorColOf<Ptr>:$dest,
        Allocator:$allocator);

    let assemblyFormat = [{
        $value `[` $sel `]` `->` $dest `:` type($value) `->` type($dest)
        `allocator` `` `=` `` $allocator
        attr-dict
    }];
}

// ============================================================================
// ============================ LOWER TO LLVM OPS =============================
// ============================================================================
def AllocStructOp : Columnar_Op<"struct.alloc", [Pure, InferTypeOpAdaptor]> {
    let arguments = (ins Variadic<AnyType>:$values);

    let results = (outs PtrOf<Struct>:$result);

    let assemblyFormat = [{
        $values `:` type($values) attr-dict
    }];
}

def GetStructElementOp : Columnar_Op<"struct.get", [Pure, InferTypeOpAdaptor]> {
    let arguments = (ins PtrOf<Struct>:$value, I64Attr:$index);

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $index $value `:` type($value) attr-dict
    }];
}

def PipelineRefOp : Columnar_Op<"pipeline_ref", [IsolatedFromAbove]> {
    let arguments = (ins
        OptionalAttr<SymbolRefAttr>:$globalOpen,
        OptionalAttr<SymbolRefAttr>:$localOpen,
        SymbolRefAttr:$body,
        OptionalAttr<SymbolRefAttr>:$localClose,
        OptionalAttr<SymbolRefAttr>:$globalClose);

    let assemblyFormat = [{
        (`global_open` $globalOpen^)?
        (`local_open` $localOpen^)?
        `body` $body
        (`local_close` $localClose^)?
        (`global_close` $globalClose^)?
        attr-dict
    }];
}

def ConstantStringOp : Columnar_Op<"constant_string", [Pure, ConstantLike]> {
    let arguments = (ins StrAttr:$value);

    let results = (outs StringLiteral:$result);

    let assemblyFormat = [{
        $value attr-dict
    }];

    let hasFolder = 1;
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

def MakePipelines: Pass<"make-pipelines", "mlir::ModuleOp"> {
    let summary = "Groups and splits operators into pipelines";
}

def GroupTableReads: Pass<"group-table-reads", "mlir::ModuleOp"> {
    let summary = "Groups column read ops into a single table read op";
}

def LowerPipelines : Pass<"lower-pipelines", "mlir::ModuleOp"> {
    let summary = "Lower pipelines to operations over tensors";

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",
    ];
}

def LowerToLLVM : Pass<"lower-to-llvm", "mlir::ModuleOp"> {
    let summary = "Lower pipelines to LLVM functions";

    let dependentDialects = [
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect",
    ];
}

#endif // COLUMNAR
