#ifndef OPERATOR_OPS
#define OPERATOR_OPS

include "execution/operator/IR/OperatorDialect.td"
include "execution/operator/IR/OperatorTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ScanParquetOp : Operator_Op<"scan_parquet", [Pure]> {
    let summary = "Scan a parquet file";
    let description = [{
        Scan a table stored in a parquet file
    }];

    let arguments = (ins StrAttr:$path, ArrayAttr:$columns);
    let results = (outs Operator_Table:$result);

    let assemblyFormat = [{
        $path $columns attr-dict `:` type($result)
    }];
    let hasVerifier = 1;
}

def FilterOp : Operator_Op<"filter", [Pure, SameOperandsAndResultType]> {
    let summary = "Filter rows from the child based on a predicate";
    let description = [{
        Applies a predicate to all rows produced by the child operator.
        Only rows that satisfy the predicate are forwarded to the parent.
    }];

    let arguments = (ins Operator_Table:$child);
    // TODO: validate that the block args match the columns
    let regions = (region SizedRegion<1>:$predicate);
    let results = (outs Operator_Table:$result);

    let assemblyFormat = [{
        $child `:` type($result) attr-dict-with-keyword $predicate
    }];
    let hasVerifier = 1;
}

def FilterReturnOp : Operator_Op<"filter.return", [Terminator, Pure, HasParent<"FilterOp">]> {
    let summary = "return from a filter predicate block";
    let description = [{
        Return the evaluated predicate condition value.
    }];

    let arguments = (ins I1:$condition);

    let assemblyFormat = [{
        $condition attr-dict
    }];
}

def AggregateOp : Operator_Op<"aggregate", [Pure]> {
    let summary = "Aggregates rows by a set of keys, merge values of other rows using aggregation functions.";

    let arguments = (ins Operator_Table:$child);
    let regions = (region SizedRegion<1>:$aggregators);
    let results = (outs Operator_Table:$result);

    let assemblyFormat = [{
        $child `:` type($child) `->` type($result) attr-dict-with-keyword $aggregators
    }];
}

def AggregateReturnOp : Operator_Op<"aggregate.return", [Terminator, Pure, HasParent<"AggregateOp">]> {
    let summary = "return from an aggregators block";
    let description = [{
        Return the output column configuration for the aggregation.
    }];

    let arguments = (ins Variadic<Aggregator>:$aggregators);

    let assemblyFormat = [{
        ( `(` $aggregators^ `:` type($aggregators) `)` )? attr-dict
    }];
}

def AggregateKeyOp : Operator_Op<"aggregate.key", [Pure]> {
    let summary = "a key column for an aggregation";

    let arguments = (ins AnyType:$column);
    let results = (outs Aggregator:$result);

    let assemblyFormat = [{
        $column `:` type($column) `->` type($result) attr-dict
    }];
}

def AggregateSumOp : Operator_Op<"aggregate.sum", [Pure]> {
    let summary = "a sum column for an aggregation";

    let arguments = (ins AnyType:$column);
    let results = (outs Aggregator:$result);

    let assemblyFormat = [{
        $column `:` type($column) `->` type($result) attr-dict
    }];
}

def AggregateAvgOp : Operator_Op<"aggregate.avg", [Pure]> {
    let summary = "an average column for an aggregation";

    let arguments = (ins AnyType:$column);
    let results = (outs Aggregator:$result);

    let assemblyFormat = [{
        $column `:` type($column) `->` type($result) attr-dict
    }];
}

def AggregateCountOp : Operator_Op<"aggregate.count", [Pure]> {
    let summary = "a tuple count column for an aggregation";

    // TODO: builder so we don't have to set the type for the aggregator.
    let results = (outs Aggregator:$result);

    let assemblyFormat = [{
        type($result) attr-dict
    }];
}

def ProjectOp : Operator_Op<"project", [Pure]> {
    let summary = "restricts the set of columns returned, and applies arithmetic to tuples";

    let arguments = (ins Operator_Table:$child);
    // TODO: validate that the block args match the columns
    let regions = (region SizedRegion<1>:$projections);
    let results = (outs Operator_Table:$result);

    let assemblyFormat = [{
        $child `:` type($child) `->` type($result) attr-dict-with-keyword $projections
    }];
}

def ProjectReturnOp : Operator_Op<"project.return", [Terminator, Pure, HasParent<"ProjectOp">]> {
    let summary = "return a set of projections";

    let arguments = (ins Variadic<AnyType>:$projections);

    let assemblyFormat = [{
        ( `(` $projections^ `:` type($projections) `)` )? attr-dict
    }];
}

#endif // OPERATOR_OPS