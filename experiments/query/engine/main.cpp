#include <iostream>
#include <memory>
#include <string>
#include <vector>

enum class Datatype { STRING, INT };

struct ColumnDef {
  std::string name;
  Datatype type;
};

struct TableSchema {
  std::vector<ColumnDef> columns;
};

class ConstantColumn {
public:
  virtual ~ConstantColumn() = default;
  static std::shared_ptr<ConstantColumn> ofInts(std::vector<int> data);
  static std::shared_ptr<ConstantColumn>
  ofStrings(std::vector<std::string> data);

  virtual size_t getRows() const = 0;
  virtual std::string getRowValue(size_t idx) const = 0;
};
struct IntColumn : public ConstantColumn {
  std::vector<int> data;

  size_t getRows() const override { return data.size(); }
  std::string getRowValue(size_t idx) const {
    return std::to_string(data.at(idx));
  }
};
struct StringColumn : public ConstantColumn {
  std::vector<std::string> data;

  size_t getRows() const override { return data.size(); }
  std::string getRowValue(size_t idx) const { return data.at(idx); }
};

std::shared_ptr<ConstantColumn> ConstantColumn::ofInts(std::vector<int> data) {
  auto *col = new IntColumn();
  col->data = data;
  return std::shared_ptr<ConstantColumn>(col);
}

std::shared_ptr<ConstantColumn>
ConstantColumn::ofStrings(std::vector<std::string> data) {
  auto *col = new StringColumn();
  col->data = data;
  return std::shared_ptr<ConstantColumn>(col);
}

class Table {};
struct ConstantTable : public Table {
  TableSchema schema;
  std::vector<std::shared_ptr<ConstantColumn>> columnData;
};

class Operator {
public:
  virtual ~Operator() = default;
  // true if there is a next row
  virtual bool next() = 0;
  virtual std::string getColumnValue(size_t idx) = 0;
};

class ConstantTableScanner : public Operator {
private:
  ConstantTable &table;
  ssize_t rowIdx;

public:
  ConstantTableScanner(ConstantTable &table) : table(table), rowIdx(-1) {}

  bool next() override {
    rowIdx++;

    // Check if we have another row
    if (table.columnData.empty()) {
      return false;
    }
    for (const auto &col : table.columnData) {
      // Technically checking one is enough
      if (rowIdx >= col->getRows()) {
        return false;
      }
    }

    return true;
  }

  std::string getColumnValue(size_t idx) override {
    return table.columnData.at(idx)->getRowValue(rowIdx);
  }
};

int main(int argc, char **argv) {
  ConstantTable table{
      .schema = TableSchema{.columns = {ColumnDef{
                                            .name = "id",
                                            .type = Datatype::INT,
                                        },
                                        ColumnDef{.name = "name",
                                                  .type = Datatype::STRING}}},
      .columnData = {ConstantColumn::ofInts({1, 2, 3}),
                     ConstantColumn::ofStrings({"Alice", "Bob", "Charlie"})}};

  ConstantTableScanner scanner(table);
  while (scanner.next()) {
    std::cout << "id: " << scanner.getColumnValue(0)
              << ", name: " << scanner.getColumnValue(1) << "\n";
  }

  return 0;
}