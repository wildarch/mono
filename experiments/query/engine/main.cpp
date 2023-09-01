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

class ConstantColumn {};
struct IntColumn : public ConstantColumn {
  std::vector<int> data;
};
struct StringColumn : public ConstantColumn {
  std::vector<std::string> data;
};

class Table {};
struct ConstantTable : public Table {
  TableSchema schema;
  std::vector<ConstantColumn> columnData;
};

int main(int argc, char **argv) {
  Table table = ConstantTable{
      .schema = TableSchema{.columns = {ColumnDef{
                                            .name = "id",
                                            .type = Datatype::INT,
                                        },
                                        ColumnDef{.name = "name",
                                                  .type = Datatype::STRING}}},
      .columnData = {IntColumn{.data = {1, 2, 3}},
                     StringColumn{.data = {"Alice", "Bob", "Charlie"}}}};
}