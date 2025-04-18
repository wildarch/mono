#pragma once

#include "columnar/Catalog.h"
#include "columnar/Columnar.h"

#include "pg_query.pb.h"

namespace columnar {

class SQLParser {
private:
  struct AggregateState {
    llvm::SmallVector<mlir::Value> groupBy;
    llvm::SmallVector<mlir::Value> aggregate;
    llvm::SmallVector<columnar::Aggregator> aggregators;

    llvm::SmallVector<mlir::StringAttr> groupByNames;
    llvm::SmallVector<mlir::StringAttr> aggregateNames;
  };

  const Catalog &_catalog;
  mlir::OpBuilder _builder;

  // Available columns, indexed by name.
  llvm::StringMap<mlir::Value> _columnsByName;
  // The current set of top-level columns.
  llvm::SmallVector<mlir::Value> _currentColumns;

  llvm::StringMap<mlir::Value> _correlatedColumns;

  SQLParser(const Catalog &catalog, mlir::OpBuilder &&builder);
  SQLParser newSubParser(mlir::OpBuilder &&builder,
                         const mlir::IRMapping &mapping);

  void parseStmt(const pg_query::RawStmt &stmt);
  void parseSelect(const pg_query::SelectStmt &stmt);
  void parseFromRelation(const pg_query::RangeVar &rel);
  void parseWhere(const pg_query::Node &expr);
  void parseResTarget(const pg_query::ResTarget &target,
                      llvm::SmallVectorImpl<mlir::Value> &outputValues,
                      llvm::SmallVectorImpl<mlir::StringAttr> &outputNames);
  void parseAggregateResTarget(const pg_query::ResTarget &target,
                               AggregateState &state);
  void parseAggregate(const pg_query::SelectStmt &stmt,
                      llvm::SmallVectorImpl<mlir::Value> &outputValues,
                      llvm::SmallVectorImpl<mlir::StringAttr> &outputNames);
  void parseOrderBy(const pg_query::SelectStmt &stmt,
                    llvm::SmallVectorImpl<mlir::Value> &outputValues,
                    llvm::SmallVectorImpl<mlir::StringAttr> &outputNames);
  void parseLimit(const pg_query::SelectStmt &stmt,
                  llvm::SmallVectorImpl<mlir::Value> &outputValues);

  void parseAggregateSum(const pg_query::ResTarget &target,
                         const pg_query::FuncCall &call, AggregateState &state);
  void parseAggregateMin(const pg_query::ResTarget &target,
                         const pg_query::FuncCall &call, AggregateState &state);
  void parseAggregateAvg(const pg_query::ResTarget &target,
                         const pg_query::FuncCall &call, AggregateState &state);
  void parseAggregateCount(const pg_query::ResTarget &target,
                           const pg_query::FuncCall &call,
                           AggregateState &state);

  // NOTE: Called inside of selection predicate.
  void parsePredicate(const pg_query::Node &expr);

  mlir::Value parseTupleExpr(const pg_query::Node &expr);
  mlir::Value parseTupleExpr(const pg_query::A_Expr &expr);
  mlir::Value parseTupleExprOp(const pg_query::A_Expr &expr);
  mlir::Value parseTupleExprBetween(const pg_query::A_Expr &expr);
  mlir::Value parseTupleExprLike(const pg_query::A_Expr &expr);
  mlir::Value parseTupleExpr(const pg_query::ColumnRef &expr);
  mlir::Value parseTupleExpr(const pg_query::A_Const &expr);
  mlir::Value parseTupleExpr(const pg_query::TypeCast &expr);
  mlir::Value parseTupleExpr(const pg_query::BoolExpr &expr);
  mlir::Value parseTupleExprCmp(columnar::CmpPredicate pred,
                                const pg_query::A_Expr &expr);
  mlir::Value parseTupleExprBinary(
      const pg_query::A_Expr &expr,
      llvm::function_ref<mlir::Value(mlir::Value, mlir::Value)> buildFunc);
  mlir::Value parseTupleExpr(const pg_query::SubLink &expr);

  template <typename T>
  mlir::Value parseTupleExprBinary(const pg_query::A_Expr &expr) {
    return parseTupleExprBinary(
        expr, [&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
          return _builder.create<T>(loc(expr.location()), lhs, rhs);
        });
  }

  llvm::SmallString<16> parseTupleExprName(const pg_query::A_Expr &expr);

  mlir::Type parseType(const pg_query::TypeName &name);

  mlir::TypedAttr parseFloat(const std::string &s);

  bool detectAggregate(const pg_query::SelectStmt &stmt);

  mlir::LogicalResult tryUnifyTypes(mlir::Value &lhs, mlir::Value &rhs);

  mlir::InFlightDiagnostic emitError(std::int32_t loc,
                                     const google::protobuf::Message &msg);

  mlir::InFlightDiagnostic emitError(const google::protobuf::Message &msg);

  mlir::Location loc(std::int32_t l);
  mlir::Location loc(const pg_query::Node &n);

  mlir::Value findColumn(mlir::Location loc, llvm::StringRef name);
  mlir::StringAttr columnName(const pg_query::ResTarget &target,
                              mlir::Value column);
  mlir::StringAttr columnName(mlir::Value column);

  void remapCurrentColumns(const mlir::IRMapping &mapping);

public:
  static mlir::OwningOpRef<mlir::ModuleOp>
  parseQuery(mlir::MLIRContext *ctx, mlir::FileLineColLoc loc,
             const pg_query::ParseResult &proto, const Catalog &catalog);
};

} // namespace columnar
