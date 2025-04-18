#pragma once

#include <llvm/ADT/ArrayRef.h>

#include "columnar/Columnar.h"

namespace columnar {

class Catalog {
private:
  llvm::StringMap<TableAttr> _tablesByName;
  llvm::SmallDenseMap<TableAttr, llvm::SmallVector<TableColumnAttr>>
      _columnsPerTable;

public:
  void addTable(columnar::TableAttr table);
  void addColumn(columnar::TableColumnAttr column);

  columnar::TableAttr lookupTable(llvm::StringRef name) const;
  llvm::ArrayRef<TableColumnAttr> columnsOf(TableAttr table) const;

  void dump() const;
};

} // namespace columnar
