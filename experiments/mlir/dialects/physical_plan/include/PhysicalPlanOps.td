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

def DeclMemRefOp : PhysicalPlan_Op<"decl_memref", [Pure]> {
    let summary = "Declare a memref at the given address";

    let arguments = (ins I64:$basePtr);
    let results = (outs AnyMemRef:$result);

    let assemblyFormat = [{
        type($result) `at` $basePtr attr-dict
    }];
}

def ClaimSliceOp : PhysicalPlan_Op<"claim_slice"> {
    let summary = "Claim a slice of a target array";

    let description = [{
        If enough of the total capacity of the array is left, allocates $claimSize elements in the target array by incrementing the value at $offsetPtr.
        On success, the base offset at which to start writing to the array is returned as $offset.
        $errorNoCapacity is set to true if for the offset (dereferenced from $offsetPtr) we have offset + $claimSize >= $capacity.
    }];

    let arguments = (ins I64:$claimSize, I64:$offsetPtr, I64:$capacity);

    let results = (outs Index:$offset, I1:$errorNoCapacity);

    let assemblyFormat = [{
        `claim` $claimSize `at` $offsetPtr `capacity` $capacity attr-dict
    }];
}

#endif // PHYSICALPLAN_OPS