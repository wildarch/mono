#ifndef PHYSICALPLAN_PASSES
#define PHYSICALPLAN_PASSES

include "mlir/Pass/PassBase.td"

def VectorizeCompute : Pass<"vectorize-compute", "::mlir::ModuleOp"> {
    let summary = "Vectorize compute bodies";

    let dependentDialects = [
        "mlir::vector::VectorDialect",
    ];
}

def VectorizePipelines : Pass<"vectorize-pipelines", "::mlir::ModuleOp"> {
    let summary = "Vectorize pipeline ops";
}

#endif // PHYSICALPLAN_PASSES