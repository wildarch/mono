#pragma once

#include <llvm/ExecutionEngine/Orc/CoreContainers.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>

namespace columnar {

llvm::orc::SymbolMap registerRuntimeSymbols(llvm::orc::MangleAndInterner mai);

} // namespace columnar
