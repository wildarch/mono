#include <columnar/Columnar.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>

#include "pg_query.h"
#include "pg_query.pb.h"

namespace {

/** Handles memory management and error conversion of the
 * \c pg_query_parse_protobuf API. */
class PgQueryParseProtobufWrapper {
private:
  PgQueryProtobufParseResult _res;

  PgQueryParseProtobufWrapper(PgQueryProtobufParseResult res) : _res(res) {}

public:
  ~PgQueryParseProtobufWrapper() { pg_query_free_protobuf_parse_result(_res); }

  static PgQueryParseProtobufWrapper parse(const char *query) {
    return PgQueryParseProtobufWrapper(pg_query_parse_protobuf(query));
  }

  llvm::Error takeError() const {
    if (_res.error) {
      // TODO: include line and column numbers.
      llvm::Twine msg(_res.error->message);
      return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
    } else {
      return llvm::Error::success();
    }
  }

  PgQueryProtobuf parseTree() const {
    assert(!_res.error);
    return _res.parse_tree;
  }
};

class Catalog {
private:
  llvm::StringMap<columnar::TableAttr> _tables;

public:
  void addTable(columnar::TableAttr table) {
    if (_tables.contains(table.getName())) {
      mlir::emitError(mlir::UnknownLoc::get(table.getContext()))
          << "Attempt to add table " << table
          << " to catalog, but the catalog already contains "
          << _tables.at(table.getName()) << " with the same name";
      llvm_unreachable("Catalog already contains a table with this name");
    }

    _tables[table.getName()] = table;
  }

  columnar::TableAttr findTable(llvm::StringRef name) const {
    return _tables.lookup(name);
  }
};

class SQLParser {
private:
  mlir::ModuleOp _module;
  const Catalog &_catalog;
  mlir::OpBuilder _builder;

  // Available columns, indexed by name.
  llvm::StringMap<mlir::Value> _columnsByName;
  // The current set of top-level columns.
  llvm::SmallVector<mlir::Value> _currentColumns;

  SQLParser(mlir::ModuleOp module, const Catalog &catalog);

  void parseStmt(const pg_query::RawStmt &stmt);
  void parseSelect(const pg_query::SelectStmt &stmt, mlir::Location loc);
  void parseFromRelation(const pg_query::RangeVar &rel);
  void parseResTarget(const pg_query::ResTarget &target,
                      llvm::SmallVectorImpl<mlir::Value> &outputValues,
                      llvm::SmallVectorImpl<mlir::StringAttr> &outputNames);

  mlir::InFlightDiagnostic emitError(std::int32_t loc,
                                     const google::protobuf::Message &msg);

  mlir::InFlightDiagnostic emitError(const google::protobuf::Message &msg);

  mlir::Location loc(std::int32_t l);

  mlir::StringAttr columnName(mlir::Value column);

public:
  static mlir::OwningOpRef<mlir::ModuleOp>
  parseQuery(mlir::MLIRContext *ctx, mlir::FileLineColLoc loc,
             const pg_query::ParseResult &proto, const Catalog &catalog);
};

} // namespace

SQLParser::SQLParser(mlir::ModuleOp module, const Catalog &catalog)
    : _module(module), _catalog(catalog), _builder(module.getContext()) {
  _builder.setInsertionPointToStart(module.getBody());
}

void SQLParser::parseStmt(const pg_query::RawStmt &stmt) {
  if (!stmt.has_stmt()) {
    emitError(stmt.stmt_location(), stmt) << "does not contain a statement";
    return;
  }

  const auto &node = stmt.stmt();
  if (!node.has_select_stmt()) {
    emitError(stmt.stmt_location(), stmt) << "is not a SELECT statement";
    return;
  }

  _builder.setInsertionPointToEnd(_module.getBody());
  parseSelect(node.select_stmt(), loc(stmt.stmt_location()));
}

void SQLParser::parseSelect(const pg_query::SelectStmt &stmt,
                            mlir::Location loc) {
  auto queryOp = _builder.create<columnar::QueryOp>(loc);
  auto &body = queryOp.getBody().emplaceBlock();
  _builder.setInsertionPointToStart(&body);

  _columnsByName.clear();
  _currentColumns.clear();

  // Steps:
  // 1. Create table reads (FROM)
  // 2. Join tables
  // 2. Apply filters (WHERE)
  // 3. Aggregation
  // 4. ORDER BY/LIMIT

  // TODO:
  // * Sub-queries

  // Things we do not support
  if (stmt.distinct_clause_size() || stmt.has_into_clause() ||
      stmt.has_where_clause() || stmt.group_clause_size() ||
      stmt.group_distinct() || stmt.has_having_clause() ||
      stmt.window_clause_size() || stmt.values_lists_size() ||
      stmt.sort_clause_size() || stmt.has_limit_offset() ||
      stmt.has_limit_count() ||
      stmt.limit_option() != pg_query::LIMIT_OPTION_DEFAULT ||
      stmt.locking_clause_size() || stmt.has_with_clause() ||
      stmt.op() != pg_query::SETOP_NONE || stmt.all() || stmt.has_larg() ||
      stmt.has_rarg()) {
    emitError(stmt) << "unsupported feature used in SELECT";
    return;
  }

  // Base table reads (FROM)
  for (const auto &from : stmt.from_clause()) {
    if (!from.has_range_var()) {
      emitError(from) << "is not a table name";
      continue;
    }

    parseFromRelation(from.range_var());
  }

  // TODO: Join tables (FROM, JOIN)
  // TODO: apply predicates (WHERE)
  // TODO: aggregation
  // TODO: ORDER_BY/LIMIT

  llvm::SmallVector<mlir::Value> outputColumns;
  llvm::SmallVector<mlir::StringAttr> outputNames;
  for (const auto &target : stmt.target_list()) {
    if (!target.has_res_target()) {
      emitError(target) << "unsupported target";
    } else {
      parseResTarget(target.res_target(), outputColumns, outputNames);
    }
  }

  _builder.create<columnar::QueryOutputOp>(queryOp.getLoc(), outputColumns,
                                           outputNames);
}

void SQLParser::parseFromRelation(const pg_query::RangeVar &rel) {
  if (!rel.catalogname().empty() || !rel.schemaname().empty() || !rel.inh() ||
      rel.relpersistence() != "p" || rel.has_alias()) {
    emitError(rel.location(), rel)
        << "unsupported feature used in relation ref";
    return;
  }

  auto table = _catalog.findTable(rel.relname());
  if (!table) {
    emitError(rel.location(), rel) << "unknown relation " << rel.relname();
    return;
  }

  llvm::SmallVector<mlir::Value> columnReads;
  llvm::SmallVector<mlir::StringAttr> columnNames;
  for (auto col : table.getColumns()) {
    auto readOp = _builder.create<columnar::ReadTableOp>(
        loc(rel.location()),
        _builder.getType<columnar::ColumnType>(col.getType()), table.getName(),
        col.getName());
    columnReads.push_back(readOp);
    // TODO: handle collisions
    columnNames.push_back(readOp.getColumnAttr());
  }

  if (_currentColumns.empty()) {
    _currentColumns.append(columnReads);
    for (auto [name, read] : llvm::zip_equal(columnNames, columnReads)) {
      _columnsByName[name] = read;
    }
  } else {
    // Join the current set of columns with this new set.
    auto joinOp = _builder.create<columnar::JoinOp>(
        loc(rel.location()), _currentColumns, columnReads);
    _currentColumns.clear();
    _currentColumns.append(joinOp->getResults().begin(),
                           joinOp->getResults().end());

    // TODO: preserve names better
    _columnsByName.clear();
    for (auto res : joinOp->getResults()) {
      _columnsByName[columnName(res)] = res;
    }
  }
}

void SQLParser::parseResTarget(
    const pg_query::ResTarget &target,
    llvm::SmallVectorImpl<mlir::Value> &outputValues,
    llvm::SmallVectorImpl<mlir::StringAttr> &outputNames) {
  if (!target.has_val() || target.indirection_size()) {
    emitError(target.location(), target) << "unknown result target";
    return;
  }

  const auto &val = target.val();
  if (!val.has_column_ref()) {
    emitError(target.location(), val) << "unknown result target";
    return;
  }

  const auto &columnRef = val.column_ref();
  if (columnRef.fields_size() != 1) {
    emitError(columnRef.location(), columnRef) << "expected one target column";
    return;
  }

  const auto &field = columnRef.fields(0);
  if (field.has_a_star()) {
    for (auto col : _currentColumns) {
      outputValues.push_back(col);
      outputNames.push_back(columnName(col));
    }
    return;
  } else if (!field.has_string()) {
    emitError(columnRef.location(), field) << "expected a string target name";
    return;
  }

  const auto &name = field.string().sval();

  mlir::Value column = _columnsByName.lookup(name);
  if (!column) {
    emitError(columnRef.location(), field) << "unknown column: " << name;
    return;
  }

  outputValues.push_back(column);
  auto outputName = target.name().empty()
                        ? _builder.getStringAttr(name)
                        : _builder.getStringAttr(target.name());
  outputNames.push_back(outputName);
}

mlir::InFlightDiagnostic
SQLParser::emitError(std::int32_t loc, const google::protobuf::Message &msg) {
  // TODO: location tracking
  return emitError(msg);
}

mlir::InFlightDiagnostic
SQLParser::emitError(const google::protobuf::Message &msg) {
  auto diag = mlir::emitError(_module->getLoc());
  diag.attachNote() << "in proto " << msg.DebugString();
  return diag;
}

mlir::Location SQLParser::loc(std::int32_t l) {
  // TODO location tracking.
  return _builder.getUnknownLoc();
}

mlir::StringAttr SQLParser::columnName(mlir::Value column) {
  // TODO: dataflow analysis?
  auto res = llvm::dyn_cast<mlir::OpResult>(column);
  if (auto readOp = column.getDefiningOp<columnar::ReadTableOp>()) {
    return readOp.getColumnAttr();
  } else if (auto joinOp = column.getDefiningOp<columnar::JoinOp>()) {
    // Take from input.
    return columnName(joinOp->getOperand(res.getResultNumber()));
  }

  // TODO: smarter impl.
  return mlir::StringAttr::get(column.getContext(), "NO_NAME");
}

mlir::OwningOpRef<mlir::ModuleOp>
SQLParser::parseQuery(mlir::MLIRContext *ctx, mlir::FileLineColLoc loc,
                      const pg_query::ParseResult &proto,
                      const Catalog &catalog) {
  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
  SQLParser parser(*module, catalog);

  for (const auto &stmt : proto.stmts()) {
    parser.parseStmt(stmt);
  }

  return module;
}

static llvm::Expected<pg_query::ParseResult>
parseSQLToProto(const char *query) {
  auto result = PgQueryParseProtobufWrapper::parse(query);
  if (auto err = result.takeError()) {
    return err;
  }

  pg_query::ParseResult proto;
  if (!proto.ParseFromArray(result.parseTree().data, result.parseTree().len)) {
    llvm_unreachable("Cannot re-parse proto generated by pg_query");
  }

  return proto;
}

static mlir::Type typeIdent(mlir::MLIRContext *ctx) {
  return mlir::IntegerType::get(ctx, 64);
}

static mlir::Type typeString(mlir::MLIRContext *ctx) {
  return columnar::StringType::get(ctx);
}

static mlir::Type typeInteger(mlir::MLIRContext *ctx) {
  return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
}

static mlir::Type typeDecimal(mlir::MLIRContext *ctx) {
  return columnar::DecimalType::get(ctx);
}

static mlir::Type typeDate(mlir::MLIRContext *ctx) {
  return columnar::DateType::get(ctx);
}

static void initTPCHCatalog(mlir::MLIRContext *ctx, Catalog &catalog) {
  using TypeBuilder = mlir::Type(mlir::MLIRContext * ctx);

  struct ColumnDef {
    const char *name;
    TypeBuilder *type;
  };

  struct TableDef {
    const char *name;
    std::initializer_list<ColumnDef> columns;
  };

  static constexpr TableDef PART{"part",
                                 {
                                     {"p_partkey", typeIdent},
                                     {"p_name", typeString},
                                     {"p_mfgr", typeString},
                                     {"p_brand", typeString},
                                     {"p_size", typeInteger},
                                     {"p_container", typeString},
                                     {"p_retailprice", typeDecimal},
                                     {"p_comment", typeString},
                                 }};

  static constexpr TableDef SUPPLIER{"supplier",
                                     {
                                         {"s_suppkey", typeIdent},
                                         {"s_name", typeString},
                                         {"s_address", typeString},
                                         {"s_nationkey", typeIdent},
                                         {"s_phone", typeString},
                                         {"s_acctbal", typeDecimal},
                                         {"s_comment", typeString},
                                     }};

  static constexpr TableDef PARTSUPP{"partsupp",
                                     {
                                         {"ps_partkey", typeIdent},
                                         {"ps_suppkey", typeIdent},
                                         {"ps_availqty", typeInteger},
                                         {"ps_supplycost", typeDecimal},
                                         {"ps_comment", typeString},
                                     }};

  static constexpr TableDef CUSTOMER{"customer",
                                     {
                                         {"c_custkey", typeIdent},
                                         {"c_name", typeString},
                                         {"c_address", typeString},
                                         {"c_nationkey", typeIdent},
                                         {"c_phone", typeString},
                                         {"c_acctbal", typeDecimal},
                                         {"c_mktsegment", typeString},
                                         {"c_comment", typeString},
                                     }};

  static constexpr TableDef ORDERS{"orders",
                                   {
                                       {"o_orderkey", typeIdent},
                                       {"o_custkey", typeIdent},
                                       {"o_orderstatus", typeString},
                                       {"o_totalprice", typeDecimal},
                                       {"o_orderdate", typeDate},
                                       {"o_orderpriority", typeString},
                                       {"o_clerk", typeString},
                                       {"o_shippriority", typeInteger},
                                       {"o_comment", typeString},
                                   }};

  static constexpr TableDef LINEITEM{"lineitem",
                                     {
                                         {"l_orderkey", typeIdent},
                                         {"l_partkey", typeIdent},
                                         {"l_suppkey", typeIdent},
                                         {"l_linenumber", typeInteger},
                                         {"l_quantity", typeDecimal},
                                         {"l_extendedprice", typeDecimal},
                                         {"l_discount", typeDecimal},
                                         {"l_tax", typeDecimal},
                                         {"l_returnflag", typeString},
                                         {"l_linestatus", typeString},
                                         {"l_shipdate", typeDate},
                                         {"l_commitdate", typeDate},
                                         {"l_receiptdate", typeDate},
                                         {"l_shipinstruct", typeString},
                                         {"l_shipmode", typeString},
                                         {"l_comment", typeString},
                                     }};

  static constexpr TableDef NATION{"nation",
                                   {
                                       {"n_nationkey", typeIdent},
                                       {"n_name", typeString},
                                       {"n_regionkey", typeIdent},
                                       {"n_comment", typeString},
                                   }};

  static constexpr TableDef REGION{"region",
                                   {
                                       {"r_regionkey", typeIdent},
                                       {"r_name", typeString},
                                       {"r_comment", typeString},
                                   }};

  TableDef tableDefs[] = {PART,   SUPPLIER, PARTSUPP, CUSTOMER,
                          ORDERS, LINEITEM, NATION,   REGION};
  for (const auto &def : tableDefs) {
    llvm::SmallVector<columnar::TableColumnAttr> columns;
    for (const auto &col : def.columns) {
      auto type = col.type(ctx);
      columns.push_back(columnar::TableColumnAttr::get(ctx, col.name, type));
    }

    catalog.addTable(columnar::TableAttr::get(ctx, def.name, columns));
  }
}

static mlir::OwningOpRef<mlir::ModuleOp>
translateSQLToColumnar(llvm::SourceMgr &sourceMgr, mlir::MLIRContext *ctx) {
  ctx->getOrLoadDialect<columnar::ColumnarDialect>();

  auto *sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  auto loc = mlir::FileLineColLoc::get(ctx, sourceBuf->getBufferIdentifier(),
                                       /*line=*/1, /*column=*/0);

  auto proto = parseSQLToProto(sourceBuf->getBufferStart());
  if (auto err = proto.takeError()) {
    mlir::emitError(loc) << "SQL parse error: "
                         << llvm::toString(std::move(err));
    return nullptr;
  }

  Catalog catalog;
  initTPCHCatalog(ctx, catalog);
  return SQLParser::parseQuery(ctx, loc, *proto, catalog);
}

int main(int argc, char *argv[]) {
  mlir::TranslateToMLIRRegistration importSQL(
      "import-sql", "Import SQL to Columnar IR", translateSQLToColumnar);
  return failed(
      mlir::mlirTranslateMain(argc, argv, "Translates SQL to columnar IR"));
}
