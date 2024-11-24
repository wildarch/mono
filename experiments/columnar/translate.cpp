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

class SQLParser {
private:
  struct BaseColumnRead {
    std::string columnName;
    mlir::Value readOp;
  };
  struct BaseTableRead {
    // TODO: alias
    std::string tableName;
    llvm::SmallVector<BaseColumnRead> columnReads;
  };

  mlir::ModuleOp _module;
  mlir::OpBuilder _builder;
  llvm::SmallVector<BaseTableRead> _baseTableReads;

  SQLParser(mlir::ModuleOp module);

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

public:
  static mlir::OwningOpRef<mlir::ModuleOp>
  parseQuery(mlir::MLIRContext *ctx, mlir::FileLineColLoc loc,
             const pg_query::ParseResult &proto);
};

} // namespace

SQLParser::SQLParser(mlir::ModuleOp module)
    : _module(module), _builder(module.getContext()) {
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

  parseSelect(node.select_stmt(), loc(stmt.stmt_location()));
}

void SQLParser::parseSelect(const pg_query::SelectStmt &stmt,
                            mlir::Location loc) {
  auto queryOp = _builder.create<columnar::QueryOp>(loc);
  auto &body = queryOp.getBody().emplaceBlock();
  _builder.setInsertionPointToStart(&body);

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

  auto idType = _builder.getType<columnar::ColumnType>(_builder.getI64Type());

  if (rel.relname() == "part") {
    auto &partTable = _baseTableReads.emplace_back();
    partTable.tableName = "part";
    partTable.columnReads.push_back(
        BaseColumnRead{.columnName = "p_partkey",
                       .readOp = _builder.create<columnar::ReadTableOp>(
                           loc(rel.location()), idType, "part", "p_partkey")});
    return;
  }

  emitError(rel.location(), rel) << "unknown relation " << rel.relname();
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
  if (!field.has_string()) {
    emitError(columnRef.location(), field) << "expected a string target name";
    return;
  }

  const auto &name = field.string().sval();

  mlir::Value readOp;
  for (const auto &table : _baseTableReads) {
    for (const auto &col : table.columnReads) {
      if (name == col.columnName) {
        if (readOp) {
          emitError(columnRef.location(), field)
              << "column name is ambiguous: " << name;
        }

        readOp = col.readOp;
      }
    }
  }

  if (!readOp) {
    emitError(columnRef.location(), field) << "unknown column: " << name;
    return;
  }

  outputValues.push_back(readOp);

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

mlir::OwningOpRef<mlir::ModuleOp>
SQLParser::parseQuery(mlir::MLIRContext *ctx, mlir::FileLineColLoc loc,
                      const pg_query::ParseResult &proto) {
  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
  SQLParser parser(*module);

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

  return SQLParser::parseQuery(ctx, loc, *proto);
}

int main(int argc, char *argv[]) {
  mlir::TranslateToMLIRRegistration importSQL(
      "import-sql", "Import SQL to Columnar IR", translateSQLToColumnar);
  return failed(
      mlir::mlirTranslateMain(argc, argv, "Translates SQL to columnar IR"));
}
