#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

#include "PhysicalPlanDialect.h"
#include "PhysicalPlanOps.h"

namespace physicalplan {

#define GEN_PASS_DECL
#include "PhysicalPlanPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "PhysicalPlanPasses.h.inc"

}
