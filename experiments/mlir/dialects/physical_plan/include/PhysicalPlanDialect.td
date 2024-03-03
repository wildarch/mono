#ifndef PHYSICALPLAN_DIALECT
#define PHYSICALPLAN_DIALECT

include "mlir/IR/OpBase.td"

def PhysicalPlan_Dialect : Dialect {
    let name = "physicalplan";
    let summary = "Scaffolded dialect 'PhysicalPlan'";
    let description = [{
        TODO: describe this
    }];
    let cppNamespace = "::physicalplan";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

class PhysicalPlan_Op<string mnemonic, list<Trait> traits = []> :
        Op<PhysicalPlan_Dialect, mnemonic, traits>;

#endif // PHYSICALPLAN_DIALECT