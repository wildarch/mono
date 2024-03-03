#ifndef PHYSICALPLAN_TYPES
#define PHYSICALPLAN_TYPES

include "mlir/IR/AttrTypeBase.td"
include "PhysicalPlanDialect.td"

class PhysicalPlan_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PhysicalPlan_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SampleType : PhysicalPlan_Type<"Sample", "sample"> {
    let summary = "sample type";
    let description = "TODO: describe";

    let parameters = (ins "mlir::Type":$type);
    
    let assemblyFormat = "`sample` `<` $type `>`";
}

#endif // PHYSICALPLAN_TYPES