#ifndef PHYSICALPLAN_PASSES
#define PHYSICALPLAN_PASSES

include "mlir/Pass/PassBase.td"

def VectorizePipelines : Pass<"vectorize-pipelines", "::mlir::ModuleOp"> {
    let summary = "Vectorize pipeline ops";
}

#endif // PHYSICALPLAN_PASSES