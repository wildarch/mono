#ifndef PHYSICALPLAN_OPS
#define PHYSICALPLAN_OPS

include "PhysicalPlanDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TestOp : PhysicalPlan_Op<"test", [Pure]> {
    let summary = "sample op";

    let arguments = (ins I64:$input);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $input attr-dict `:` type($result)
    }];
}

#endif // PHYSICALPLAN_OPS