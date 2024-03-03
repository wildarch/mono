#ifndef PHYSICALPLAN_OPS
#define PHYSICALPLAN_OPS

include "PhysicalPlanDialect.td"
include "PhysicalPlanTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PipelineOp : PhysicalPlan_Op<"pipeline"> {
    let summary = "A pipeline";

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $body attr-dict
    }];
}

def ScanOp : PhysicalPlan_Op<"scan", [Pure]> {
    let summary = "fake scan op";

    let arguments = (ins 
        I64Attr:$numberOfTuples, 
        DenseI64ArrayAttr:$columnPointers);
    let results = (outs BlockType:$result);

    let assemblyFormat = [{
        $numberOfTuples `tuples` `at` $columnPointers `:` type($result) attr-dict
    }];

    // TODO: verify one column pointer per column in block type
}

def ComputeOp : PhysicalPlan_Op<"compute", [Pure, IsolatedFromAbove]> {
    let summary = "Compute a column derived from other columns";

    let arguments = (ins BlockType:$input);
    let results = (outs BlockType:$result);

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $input `:` type($input) `->` type($result) $body attr-dict
    }];

    // TODO: verify the compute column type
}

def ComputeReturnOp : PhysicalPlan_Op<"compute.return", [
        Pure,
        HasParent<"ComputeOp">,
        Terminator]> {
    let summary = "Return the computed column value";

    let arguments = (ins AnyType:$input);

    let assemblyFormat = [{
        $input `:` type($input) attr-dict
    }];
}

def FilterOp : PhysicalPlan_Op<"filter", [
        Pure, 
        AllTypesMatch<["input", "result"]>]> {
    let summary = "Filter tuples based on the value of a (boolean) column";

    let arguments = (ins BlockType:$input, I64Attr:$column);
    let results = (outs BlockType:$result);

    let assemblyFormat = [{
        $input `:` type($input) $column attr-dict
    }];
}

def WriteArrayOp : PhysicalPlan_Op<"write_array", [Pure, Terminator]> {
    let summary = "Write results to in-memory buffers";

    let arguments = (ins 
        BlockType:$input, 
        I64ArrayAttr:$selectedColumns,
        DenseI64ArrayAttr:$outputColumnPointers,
        I64Attr:$offsetPointer,
        I64Attr:$capacity
    );

    let assemblyFormat = [{
        $input `:` type($input) 
        `columns` $selectedColumns
        `to` $outputColumnPointers 
        `offptr` $offsetPointer 
        `capacity` $capacity
        attr-dict
    }];

    // TODO: verify 
    // - selected columns valid
    // - same number output pointers as selected columns
}

#endif // PHYSICALPLAN_OPS