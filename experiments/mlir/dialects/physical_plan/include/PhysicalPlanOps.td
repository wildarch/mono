#ifndef PHYSICALPLAN_OPS
#define PHYSICALPLAN_OPS

include "PhysicalPlanDialect.td"
include "PhysicalPlanTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ScanOp : PhysicalPlan_Op<"scan", [IsolatedFromAbove]> {
    let summary = "Scan in-memory columns";

    let arguments = (ins 
        I64Attr:$numberOfTuples, 
        DenseI64ArrayAttr:$columnPointers);
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        `ntuples` $numberOfTuples
        `at` $columnPointers 
        $body
        attr-dict
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

// ============================================================================
// ============================ Vectorized Ops ================================
// ============================================================================
def VectorizedScanOp : PhysicalPlan_Op<"scan_vec", [IsolatedFromAbove]> {
    let summary = "Scan in-memory columns using vectorized ops";

    let arguments = (ins 
        I64Attr:$numberOfTuples, 
        DenseI64ArrayAttr:$columnPointers);
    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $body attr-dict
    }];
}

def VectorizedWriteArrayOp : PhysicalPlan_Op<"write_array_vec", [Terminator, AttrSizedOperandSegments]> {
    let summary = "Write to in-memory buffers";

    let arguments = (ins 
        Variadic<AnyVector>:$inputs,
        Optional<VectorOf<[I1]>>:$mask,
        DenseI64ArrayAttr:$outputColumnPointers,
        I64Attr:$offsetPointer,
        I64Attr:$capacity
    );

    let assemblyFormat = [{
        $inputs `:` type($inputs) 
        (`mask` $mask^ `:` type($mask))?
        `to` $outputColumnPointers 
        `offptr` $offsetPointer 
        `capacity` $capacity
        attr-dict
    }];

    // TODO: verify 
    // - selected columns valid
    // - same number output pointers as selected columns
}

def PackVectorsOp : PhysicalPlan_Op<"pack_vec", [Pure, AttrSizedOperandSegments]> {
    let summary = "pack vectors into one object";

    let arguments = (ins Variadic<AnyVector>:$inputs, Optional<VectorOf<[I1]>>:$mask);
    let builders = [OpBuilder<(ins "mlir::ValueRange":$inputs, "mlir::Value":$mask)>];
    let results = (outs AnyType:$result);
}

def UnpackVectorsOp : PhysicalPlan_Op<"unpack_vec", [Pure]> {
    let summary = "unpack vectors";

    let arguments = (ins AnyType:$inputs);
    let results = (outs Variadic<AnyVector>:$result);
}

#endif // PHYSICALPLAN_OPS