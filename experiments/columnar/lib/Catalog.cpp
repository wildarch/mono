#include <llvm/ADT/StringRef.h>

#include "columnar/Catalog.h"

namespace columnar {

namespace {

class DuplicateTableNameException : public std::runtime_error {
public:
  DuplicateTableNameException(llvm::StringRef name)
      : std::runtime_error("Duplicate table name: " + name.str()) {}
};

} // namespace

void Catalog::addTable(columnar::TableAttr table) {
  // If we already have a table with the same name, throw an exception
  if (_tablesByName.contains(table.getName())) {
    throw DuplicateTableNameException(table.getName());
  }

  _tablesByName[table.getName()] = table;
}

void Catalog::addColumn(columnar::TableColumnAttr column) {
  auto table = column.getTable();
  _columnsPerTable[table].push_back(column);
}

columnar::TableAttr Catalog::lookupTable(llvm::StringRef name) {
  return _tablesByName.lookup(name);
}

llvm::ArrayRef<TableColumnAttr> Catalog::columnsOf(TableAttr table) {
  auto it = _columnsPerTable.find(table);
  assert(it != _columnsPerTable.end() && "Table not found in columns map");
  return it->second;
}

void Catalog::dump() {
  for (const auto &e : _tablesByName) {
    auto name = e.first();
    auto table = e.second;
    llvm::outs() << "Table: " << name << ": " << table << "\n";
    for (auto [i, column] : llvm::enumerate(_columnsPerTable[table])) {
      llvm::outs() << "  Column " << i << ": " << column << "\n";
    }
  }
}

} // namespace columnar
