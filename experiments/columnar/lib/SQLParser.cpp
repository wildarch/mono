#include "columnar/SQLParser.h"

namespace columnar {

SQLParser::SQLParser(const Catalog &catalog, mlir::OpBuilder &&builder)
    : _catalog(catalog), _builder(std::move(builder)) {}

SQLParser SQLParser::newSubParser(mlir::OpBuilder &&builder,
                                  const mlir::IRMapping &mapping) {
  SQLParser subParser(_catalog, std::move(builder));
  for (const auto &[name, value] : _columnsByName) {
    auto newValue = mapping.lookupOrNull(value);
    if (newValue) {
      subParser._columnsByName[name] = newValue;
    }
  }

  for (const auto &[name, value] : _correlatedColumns) {
    auto newValue = mapping.lookupOrNull(value);
    if (newValue) {
      subParser._correlatedColumns[name] = newValue;
    }
  }

  return subParser;
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

  // Parse the query.
  auto queryOp = _builder.create<columnar::QueryOp>(loc(stmt.stmt_location()));
  auto &body = queryOp.getBody().emplaceBlock();
  SQLParser queryParser(_catalog, _builder.atBlockBegin(&body));
  queryParser.parseSelect(node.select_stmt());
}

void SQLParser::parseSelect(const pg_query::SelectStmt &stmt) {
  // Things we do not support
  if (stmt.distinct_clause_size() || stmt.has_into_clause() ||
      stmt.group_distinct() || stmt.has_having_clause() ||
      stmt.window_clause_size() || stmt.values_lists_size() ||
      stmt.has_limit_offset() || stmt.locking_clause_size() ||
      stmt.has_with_clause() || stmt.op() != pg_query::SETOP_NONE ||
      stmt.all() || stmt.has_larg() || stmt.has_rarg()) {
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

  // TODO: JOIN clause

  if (stmt.has_where_clause()) {
    parseWhere(stmt.where_clause());
  }

  llvm::SmallVector<mlir::Value> outputColumns;
  llvm::SmallVector<mlir::StringAttr> outputNames;

  if (detectAggregate(stmt)) {
    parseAggregate(stmt, outputColumns, outputNames);
  } else {
    // Final SELECT
    for (const auto &target : stmt.target_list()) {
      if (!target.has_res_target()) {
        emitError(target) << "unsupported target";
      } else {
        parseResTarget(target.res_target(), outputColumns, outputNames);
      }
    }
  }

  parseOrderBy(stmt, outputColumns, outputNames);
  parseLimit(stmt, outputColumns);

  auto queryOp = _builder.getInsertionBlock()->getParentOp();
  _builder.create<columnar::QueryOutputOp>(queryOp->getLoc(), outputColumns,
                                           outputNames);
}

void SQLParser::parseFromRelation(const pg_query::RangeVar &rel) {
  if (!rel.catalogname().empty() || !rel.schemaname().empty() || !rel.inh() ||
      rel.relpersistence() != "p" || rel.has_alias()) {
    emitError(rel.location(), rel)
        << "unsupported feature used in relation ref";
    return;
  }

  auto table = _catalog.lookupTable(rel.relname());
  if (!table) {
    emitError(rel.location(), rel) << "unknown relation " << rel.relname();
    return;
  }

  auto columns = _catalog.columnsOf(table);
  llvm::SmallVector<mlir::Value> columnReads;
  llvm::SmallVector<mlir::StringAttr> columnNames;
  for (auto col : columns) {
    auto readOp = _builder.create<columnar::ReadColumnOp>(
        loc(rel.location()),
        _builder.getType<columnar::ColumnType>(col.getLogicalType()), col);
    columnReads.push_back(readOp);
    // TODO: handle collisions
    columnNames.push_back(_builder.getStringAttr(readOp.getColumn().getName()));
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

    // Update column name references
    mlir::IRMapping mapping;
    mapping.map(_currentColumns, joinOp.getLhsResults());
    for (auto &e : _columnsByName) {
      e.setValue(mapping.lookup(e.getValue()));
    }

    for (auto [name, read] :
         llvm::zip_equal(columnNames, joinOp.getRhsResults())) {
      _columnsByName[name] = read;
    }

    _currentColumns.clear();
    _currentColumns.append(joinOp->getResults().begin(),
                           joinOp->getResults().end());
  }
}

void SQLParser::parseWhere(const pg_query::Node &expr) {
  llvm::SmallVector<mlir::Value> inputs = _currentColumns;
  // HACK: Propagate correlated columns too.
  for (const auto &[name, value] : _correlatedColumns) {
    // TODO: inefficient
    if (!llvm::is_contained(inputs, value)) {
      inputs.emplace_back(value);
    }
  }

  auto selectOp = _builder.create<columnar::SelectOp>(loc(expr), inputs);
  auto &body = selectOp.getPredicates().front();

  // Remap to block arguments
  mlir::IRMapping mapping;
  mapping.map(inputs, body.getArguments());

  auto predParser = newSubParser(_builder.atBlockBegin(&body), mapping);
  // Refer to predicate block arguments.
  predParser._currentColumns.append(body.getArguments().begin(),
                                    body.getArguments().end());

  // Parse the predicate
  predParser.parsePredicate(expr);

  // Replace columns with the filtered ones.
  mapping.map(_currentColumns,
              selectOp.getResults().take_front(_currentColumns.size()));
  remapCurrentColumns(mapping);
}

void SQLParser::parsePredicate(const pg_query::Node &expr) {
  auto selectOp = llvm::cast<columnar::SelectOp>(
      _builder.getInsertionBlock()->getParentOp());
  auto &selectBody = selectOp.getPredicates().front();

  auto predOp = _builder.create<columnar::PredicateOp>(
      selectOp.getLoc(), selectBody.getArguments());
  auto &body = predOp.getBody().emplaceBlock();
  for (auto arg : selectBody.getArguments()) {
    body.addArgument(arg.getType(), arg.getLoc());
  }

  mlir::IRMapping mapping;
  mapping.map(selectBody.getArguments(), body.getArguments());

  auto subParser = newSubParser(_builder.atBlockBegin(&body), mapping);
  // Refer to predicate block arguments.
  subParser._currentColumns.append(body.getArguments().begin(),
                                   body.getArguments().end());

  auto val = subParser.parseTupleExpr(expr);
  if (!val) {
    // Default value if parseTupleExpr failed.
    val = subParser._builder.create<columnar::ConstantOp>(
        loc(expr), _builder.getBoolAttr(false));
  }

  subParser._builder.create<columnar::PredicateEvalOp>(selectOp.getLoc(), val);
}

mlir::Value SQLParser::parseTupleExpr(const pg_query::Node &expr) {
  if (expr.has_a_expr()) {
    return parseTupleExpr(expr.a_expr());
  } else if (expr.has_column_ref()) {
    return parseTupleExpr(expr.column_ref());
  } else if (expr.has_a_const()) {
    return parseTupleExpr(expr.a_const());
  } else if (expr.has_type_cast()) {
    return parseTupleExpr(expr.type_cast());
  } else if (expr.has_bool_expr()) {
    return parseTupleExpr(expr.bool_expr());
  } else if (expr.has_sub_link()) {
    return parseTupleExpr(expr.sub_link());
  }

  emitError(expr) << "unsupported tuple expr";
  return nullptr;
}

mlir::Value SQLParser::parseTupleExpr(const pg_query::A_Expr &expr) {
  switch (expr.kind()) {
  case pg_query::AEXPR_OP:
    return parseTupleExprOp(expr);
  case pg_query::AEXPR_BETWEEN:
    return parseTupleExprBetween(expr);
  case pg_query::AEXPR_LIKE:
    return parseTupleExprLike(expr);
  case pg_query::A_EXPR_KIND_UNDEFINED:
  case pg_query::AEXPR_OP_ANY:
  case pg_query::AEXPR_OP_ALL:
  case pg_query::AEXPR_DISTINCT:
  case pg_query::AEXPR_NOT_DISTINCT:
  case pg_query::AEXPR_NULLIF:
  case pg_query::AEXPR_IN:
  case pg_query::AEXPR_ILIKE:
  case pg_query::AEXPR_SIMILAR:
  case pg_query::AEXPR_NOT_BETWEEN:
  case pg_query::AEXPR_BETWEEN_SYM:
  case pg_query::AEXPR_NOT_BETWEEN_SYM:
  default:
    break;
  }

  emitError(expr) << "unsupported kind "
                  << pg_query::A_Expr_Kind_Name(expr.kind());
  return nullptr;
}

mlir::Value SQLParser::parseTupleExprOp(const pg_query::A_Expr &expr) {
  auto name = parseTupleExprName(expr);

  // Compare
  if (name == "=") {
    return parseTupleExprCmp(columnar::CmpPredicate::EQ, expr);
  } else if (name == "<>") {
    return parseTupleExprCmp(columnar::CmpPredicate::NE, expr);
  } else if (name == "<") {
    return parseTupleExprCmp(columnar::CmpPredicate::LT, expr);
  } else if (name == "<=") {
    return parseTupleExprCmp(columnar::CmpPredicate::LE, expr);
  } else if (name == ">") {
    return parseTupleExprCmp(columnar::CmpPredicate::GT, expr);
  } else if (name == ">=") {
    return parseTupleExprCmp(columnar::CmpPredicate::GE, expr);
  }

  // Arithmetic
  if (name == "+") {
    return parseTupleExprBinary<columnar::AddOp>(expr);
  } else if (name == "-") {
    return parseTupleExprBinary<columnar::SubOp>(expr);
  } else if (name == "*") {
    return parseTupleExprBinary<columnar::MulOp>(expr);
  } else if (name == "/") {
    return parseTupleExprBinary<columnar::DivOp>(expr);
  }

  emitError(expr) << "unknown op '" << name << "'";
  return nullptr;
}

mlir::Value SQLParser::parseTupleExprBetween(const pg_query::A_Expr &expr) {
  auto value = parseTupleExpr(expr.lexpr());
  if (!value) {
    return nullptr;
  }

  // Parse lower and upper bounds for value.
  if (!expr.rexpr().has_list()) {
    emitError(expr.location(), expr.rexpr()) << "invalid bounds for between";
    return nullptr;
  }

  const auto &list = expr.rexpr().list();
  if (list.items_size() != 2) {
    emitError(expr.location(), list)
        << "expected lower and upper bound (2 items)";
    return nullptr;
  }

  auto lower = parseTupleExpr(list.items(0));
  auto upper = parseTupleExpr(list.items(1));
  if (!lower || !upper) {
    return nullptr;
  }

  auto location = loc(expr.location());
  auto lowerCmp = _builder.create<columnar::CmpOp>(
      location, columnar::CmpPredicate::LE, lower, value);
  auto upperCmp = _builder.create<columnar::CmpOp>(
      location, columnar::CmpPredicate::LE, value, upper);
  return _builder.create<columnar::AndOp>(location,
                                          mlir::ValueRange{lowerCmp, upperCmp});
}

mlir::Value SQLParser::parseTupleExprLike(const pg_query::A_Expr &expr) {
  auto name = parseTupleExprName(expr);

  bool notLike;
  if (name == "~~") {
    // Regular LIKE.
    notLike = false;
  } else if (name == "!~~") {
    // NOT LIKE
    notLike = true;
  } else {
    emitError(expr.location(), expr)
        << "invalid function name for like: " << name;
    return nullptr;
  }

  auto lhs = parseTupleExpr(expr.lexpr());
  auto rhs = parseTupleExpr(expr.rexpr());
  if (!lhs || !rhs) {
    return nullptr;
  }

  mlir::Value likeOp =
      _builder.create<columnar::LikeOp>(loc(expr.location()), lhs, rhs);
  if (notLike) {
    likeOp = _builder.create<columnar::NotOp>(loc(expr.location()), likeOp);
  }

  return likeOp;
}

mlir::Value SQLParser::parseTupleExpr(const pg_query::ColumnRef &expr) {
  if (expr.fields_size() != 1) {
    emitError(expr.location(), expr) << "expected one column field";
    return nullptr;
  }

  const auto &field = expr.fields(0);
  if (!field.has_string()) {
    emitError(expr.location(), field) << "expected a string target name";
    return nullptr;
  }

  const auto &name = field.string().sval();

  mlir::Value column = findColumn(loc(expr.location()), name);
  if (!column) {
    return nullptr;
  }

  return column;
}

mlir::Value SQLParser::parseTupleExpr(const pg_query::A_Const &expr) {
  mlir::TypedAttr attr;
  if (expr.has_ival()) {
    attr = _builder.getIntegerAttr(
        _builder.getIntegerType(64, /*isSigned=*/true), expr.ival().ival());
  } else if (expr.has_sval()) {
    attr = _builder.getAttr<columnar::StringAttr>(
        _builder.getStringAttr(expr.sval().sval()));
  } else if (expr.has_fval()) {
    attr = parseFloat(expr.fval().fval());
  }

  if (!attr) {
    emitError(expr) << "unsupported constant type";
    return nullptr;
  }

  return _builder.create<columnar::ConstantOp>(loc(expr.location()), attr);
}

mlir::Value SQLParser::parseTupleExpr(const pg_query::TypeCast &expr) {
  auto arg = parseTupleExpr(expr.arg());
  auto type = parseType(expr.type_name());
  if (!arg || !type) {
    return nullptr;
  }

  return _builder.create<columnar::CastOp>(
      loc(expr.location()), _builder.getType<columnar::ColumnType>(type), arg);
}

mlir::Value SQLParser::parseTupleExpr(const pg_query::BoolExpr &expr) {
  if (expr.has_xpr()) {
    emitError(expr.location(), expr) << "unsupported boolean expr";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value> inputs;
  for (const auto &arg : expr.args()) {
    inputs.push_back(parseTupleExpr(arg));
  }

  auto nullInput =
      llvm::any_of(inputs, [](mlir::Value input) { return input == nullptr; });
  if (nullInput) {
    return nullptr;
  }

  switch (expr.boolop()) {
  case pg_query::AND_EXPR:
    return _builder.create<columnar::AndOp>(loc(expr.location()), inputs);
  case pg_query::OR_EXPR:
  case pg_query::NOT_EXPR:
  default:
    break;
  }

  emitError(expr.location(), expr) << "unsupported boolean expr";
  return nullptr;
}

mlir::Value SQLParser::parseTupleExprCmp(columnar::CmpPredicate pred,
                                         const pg_query::A_Expr &expr) {
  auto lhs = parseTupleExpr(expr.lexpr());
  auto rhs = parseTupleExpr(expr.rexpr());
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (mlir::failed(tryUnifyTypes(lhs, rhs))) {
    return nullptr;
  }

  return _builder.create<columnar::CmpOp>(loc(expr.location()), pred, lhs, rhs);
}

mlir::Value SQLParser::parseTupleExprBinary(
    const pg_query::A_Expr &expr,
    llvm::function_ref<mlir::Value(mlir::Value, mlir::Value)> buildFunc) {
  auto lhs = parseTupleExpr(expr.lexpr());
  auto rhs = parseTupleExpr(expr.rexpr());
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (mlir::failed(tryUnifyTypes(lhs, rhs))) {
    return nullptr;
  }

  return buildFunc(lhs, rhs);
}

mlir::Value SQLParser::parseTupleExpr(const pg_query::SubLink &expr) {
  if (expr.has_xpr() || expr.sub_link_id() != 0 || expr.has_testexpr() ||
      expr.oper_name_size() || expr.sub_link_type() != pg_query::EXPR_SUBLINK) {
    emitError(expr.location(), expr) << "unsupported feature in sublink";
    return nullptr;
  }

  const auto &subSelect = expr.subselect();
  const auto &select = subSelect.select_stmt();

  // Placeholder, adjusted after the body has been populated.
  auto resultTypePlaceholder =
      _builder.getType<columnar::ColumnType>(_builder.getI1Type());

  // Sharing current columns to handle correlated subqueries.
  auto subOp = _builder.create<columnar::SubQueryOp>(
      loc(expr.location()), resultTypePlaceholder, _currentColumns);
  auto &subBody = subOp.getBody().emplaceBlock();

  mlir::IRMapping mapping;
  for (auto col : _currentColumns) {
    auto arg = subBody.addArgument(col.getType(), col.getLoc());
    mapping.map(col, arg);
  }

  auto subParser =
      newSubParser(_builder.atBlockBegin(&subBody), mlir::IRMapping());
  for (const auto &[name, value] : _columnsByName) {
    subParser._correlatedColumns[name] = mapping.lookup(value);
  }

  subParser.parseSelect(select);

  // Infer the result type
  if (!subBody.mightHaveTerminator()) {
    return nullptr;
  }

  auto outputOp = llvm::dyn_cast_if_present<columnar::QueryOutputOp>(
      subBody.getTerminator());
  if (!outputOp) {
    return nullptr;
  }

  if (outputOp.getColumns().size() != 1) {
    emitError(expr.location(), expr)
        << "does not produce exactly one output column";
    return nullptr;
  }

  auto result = outputOp.getColumns()[0];
  subOp.getResult().setType(llvm::cast<columnar::ColumnType>(result.getType()));

  return subOp;
}

llvm::SmallString<16>
SQLParser::parseTupleExprName(const pg_query::A_Expr &expr) {
  llvm::SmallString<16> fullName;
  for (const auto &name : expr.name()) {
    if (!name.has_string()) {
      emitError(expr.location(), expr) << "op name is not a string";
      return fullName;
    }

    if (!fullName.empty()) {
      fullName.push_back('.');
    }

    fullName.append(name.string().sval());
  }

  return fullName;
}

mlir::Type SQLParser::parseType(const pg_query::TypeName &name) {
  if (name.names_size() == 1 && name.type_oid() == 0 && !name.setof() &&
      !name.pct_type() && name.typmods_size() == 0 && name.typemod() == -1 &&
      name.array_bounds_size() == 0) {
    const auto &node = name.names(0);
    if (node.has_string()) {
      const auto &name = node.string().sval();
      if (name == "date") {
        return _builder.getType<columnar::DateType>();
      }

      // TODO: more types.
    }
  }
  emitError(name) << "unsupported type";
  return nullptr;
}

mlir::TypedAttr SQLParser::parseFloat(const std::string &s) {
  // Can we parse as decimal?
  auto dot = s.find_first_of('.');
  if (dot == std::string::npos) {
    return nullptr;
  }

  // Example:
  // 123.45
  //    3
  // 6 - 3 - 1 = 2 decimals
  std::size_t decimals = s.size() - dot - 1;

  // Decimal allows at most two digits after the dot.
  std::string str(s);
  if (decimals > 2) {
    // Regular float
    return _builder.getF64FloatAttr(std::stod(str));
  }

  // Padding with trailing zeros.
  // Example:
  // 123.4 => 123.40
  for (int i = decimals; i < 2; i++) {
    str.push_back('0');
  }

  // Example: 123.40 => 12340
  str.erase(dot, 1);
  std::int64_t intVal;
  if (llvm::StringRef(str).consumeInteger(10, intVal)) {
    return nullptr;
  }

  return _builder.getAttr<columnar::DecimalAttr>(intVal);
}

bool SQLParser::detectAggregate(const pg_query::SelectStmt &stmt) {
  if (stmt.group_clause_size()) {
    return true;
  }

  // Look for aggregation functions
  for (const auto &target : stmt.target_list()) {
    if (!target.has_res_target()) {
      continue;
    }

    const auto &resTarget = target.res_target();
    if (!resTarget.has_val()) {
      continue;
    }

    const auto &val = resTarget.val();
    if (!val.has_func_call()) {
      continue;
    }

    const auto &call = val.func_call();
    const auto &names = call.funcname();
    if (names.size() != 1) {
      continue;
    }

    const auto &name = names[0];
    const auto &nameVal = name.string().sval();

    if (nameVal == "sum" || nameVal == "avg" || nameVal == "count" ||
        nameVal == "min") {
      return true;
    }
  }

  return false;
}

static bool canCoerceToDecimal(mlir::Type columnType) {
  auto type = llvm::cast<columnar::ColumnType>(columnType).getElementType();
  return llvm::isa<columnar::DecimalType>(type) || type.isSignedInteger();
}

mlir::LogicalResult SQLParser::tryUnifyTypes(mlir::Value &lhs,
                                             mlir::Value &rhs) {
  if (lhs.getType() == rhs.getType()) {
    return mlir::success();
  }

  // Promote both to decimal.
  if (canCoerceToDecimal(lhs.getType()) && canCoerceToDecimal(rhs.getType())) {
    auto decType = _builder.getType<columnar::ColumnType>(
        _builder.getType<columnar::DecimalType>());
    lhs = _builder.createOrFold<columnar::CastOp>(lhs.getLoc(), decType, lhs);
    rhs = _builder.createOrFold<columnar::CastOp>(lhs.getLoc(), decType, rhs);
    return mlir::success();
  }

  auto loc = _builder.getFusedLoc({lhs.getLoc(), rhs.getLoc()});
  return mlir::emitError(loc)
         << "cannot unify types of values " << lhs << " and " << rhs;
}

static bool isAStar(const pg_query::Node &node) {
  if (!node.has_column_ref()) {
    return false;
  }

  const auto &columnRef = node.column_ref();
  if (columnRef.fields_size() != 1) {
    return false;
  }

  const auto &field = columnRef.fields(0);
  return field.has_a_star();
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
  if (isAStar(val)) {
    for (auto col : _currentColumns) {
      outputValues.push_back(col);
      outputNames.push_back(columnName(col));
    }
    return;
  }

  auto expr = parseTupleExpr(val);
  if (!expr) {
    return;
  }

  outputValues.push_back(expr);
  auto outputName = target.name().empty()
                        ? columnName(expr)
                        : _builder.getStringAttr(target.name());
  outputNames.push_back(outputName);
}

void SQLParser::parseAggregateResTarget(const pg_query::ResTarget &target,
                                        AggregateState &state) {
  if (!target.has_val() || target.indirection_size()) {
    emitError(target.location(), target) << "unknown result target";
    return;
  }

  const auto &val = target.val();

  // TODO: Try detect an aggregator function.
  if (val.has_func_call()) {
    const auto &call = val.func_call();
    if (call.funcname_size() == 1) {
      const auto &name = call.funcname(0).string().sval();

      // TODO: checks for various aggregator names.
      // NOTE: Need to do early return here to avoid re-parse as tuple expr.
      if (name == "sum") {
        return parseAggregateSum(target, call, state);
      } else if (name == "avg") {
        return parseAggregateAvg(target, call, state);
      } else if (name == "count") {
        return parseAggregateCount(target, call, state);
      } else if (name == "min") {
        return parseAggregateMin(target, call, state);
      }
    }
  }

  auto expr = parseTupleExpr(val);
  if (!expr) {
    return;
  }

  state.groupBy.push_back(expr);
  state.groupByNames.push_back(columnName(target, expr));
}

void SQLParser::parseAggregate(
    const pg_query::SelectStmt &stmt,
    llvm::SmallVectorImpl<mlir::Value> &outputValues,
    llvm::SmallVectorImpl<mlir::StringAttr> &outputNames) {
  AggregateState state;
  for (const auto &target : stmt.target_list()) {
    if (!target.has_res_target()) {
      emitError(target) << "unsupported target";
      return;
    }

    parseAggregateResTarget(target.res_target(), state);
  }

  // TODO: check group-by

  // Create AggregateOp
  auto queryOp = _builder.getBlock()->getParentOp();
  auto aggOp = _builder.create<columnar::AggregateOp>(
      queryOp->getLoc(), state.groupBy, state.aggregate,
      _builder.getAttr<columnar::AggregatorArrayAttr>(state.aggregators));

  // Populate outputValues and outputNames
  assert(aggOp.getGroupByResults().size() == state.groupByNames.size());
  outputValues.append(aggOp.getGroupByResults().begin(),
                      aggOp.getGroupByResults().end());
  outputNames.append(state.groupByNames);

  assert(aggOp.getAggregationResults().size() == state.aggregateNames.size());
  outputValues.append(aggOp.getAggregationResults().begin(),
                      aggOp.getAggregationResults().end());
  outputNames.append(state.aggregateNames);
}

static columnar::SortDirection mapEnum(pg_query::SortByDir d) {
  switch (d) {
  case pg_query::SORTBY_ASC:
    return columnar::SortDirection::ASC;
  case pg_query::SORTBY_DESC:
    return columnar::SortDirection::DESC;
  default:
    return columnar::SortDirection::ASC;
  }
}

void SQLParser::parseOrderBy(
    const pg_query::SelectStmt &stmt,
    llvm::SmallVectorImpl<mlir::Value> &outputValues,
    llvm::SmallVectorImpl<mlir::StringAttr> &outputNames) {
  if (stmt.sort_clause_size() == 0) {
    // Nothing to sort.
    return;
  }

  llvm::SmallDenseMap<mlir::StringAttr, columnar::SortDirection> keyToDir;
  for (const auto &clause : stmt.sort_clause()) {
    if (!clause.has_sort_by()) {
      emitError(clause) << "unsupported sort clause";
      continue;
    }

    const auto &sortBy = clause.sort_by();
    if (sortBy.sortby_nulls() != pg_query::SORTBY_NULLS_DEFAULT ||
        sortBy.use_op_size()) {
      emitError(sortBy.location(), sortBy) << "unsupported sort clause";
      continue;
    }

    // Get the name
    const auto &node = sortBy.node();
    if (!node.has_column_ref()) {
      emitError(sortBy.location(), sortBy) << "unsupported sort clause";
      continue;
    }

    const auto &columnRef = node.column_ref();
    if (columnRef.fields_size() != 1) {
      emitError(sortBy.location(), sortBy) << "unsupported sort clause";
      continue;
    }

    const auto &field = columnRef.fields(0);
    const auto &name = field.string().sval();
    keyToDir[_builder.getStringAttr(name)] = mapEnum(sortBy.sortby_dir());
  }

  // Partition outputs into keys and values
  llvm::SmallVector<mlir::Value> keys;
  llvm::SmallVector<columnar::SortDirection> dirs;
  llvm::SmallVector<mlir::Value> values;
  for (auto [name, value] : llvm::zip_equal(outputNames, outputValues)) {
    if (keyToDir.contains(name)) {
      keys.push_back(value);
      dirs.push_back(keyToDir.at(name));
    } else {
      values.push_back(value);
    }
  }

  auto queryOp = _builder.getBlock()->getParentOp();
  auto orderOp = _builder.create<columnar::OrderByOp>(queryOp->getLoc(), keys,
                                                      dirs, values);
  mlir::IRMapping mapping;
  for (auto [input, output] :
       llvm::zip_equal(orderOp.getKeys(), orderOp.getKeyResults())) {
    mapping.map(input, output);
  }

  for (auto [input, output] :
       llvm::zip_equal(orderOp.getValues(), orderOp.getValueResults())) {
    mapping.map(input, output);
  }

  // Update the output values
  for (auto &v : outputValues) {
    v = mapping.lookup(v);
  }
}

void SQLParser::parseLimit(const pg_query::SelectStmt &stmt,
                           llvm::SmallVectorImpl<mlir::Value> &outputValues) {
  switch (stmt.limit_option()) {
  case pg_query::LIMIT_OPTION_DEFAULT:
    // No limit applied
    return;
  case pg_query::LIMIT_OPTION_COUNT:
    // Handle limit below break;
    break;
  default:
    emitError(stmt) << "unsupported limit option";
    return;
  }

  assert(stmt.limit_option() == pg_query::LIMIT_OPTION_COUNT);
  const auto &node = stmt.limit_count();
  if (!node.has_a_const()) {
    emitError(node) << "limit is not a constant";
    return;
  }

  const auto &cnst = node.a_const();
  auto limit = cnst.ival().ival();
  auto limitOp = _builder.create<columnar::LimitOp>(loc(cnst.location()), limit,
                                                    outputValues);

  // Update the output values
  outputValues.clear();
  auto results = limitOp.getResults();
  outputValues.append(results.begin(), results.end());
}

void SQLParser::parseAggregateSum(const pg_query::ResTarget &target,
                                  const pg_query::FuncCall &call,
                                  AggregateState &state) {
  if (call.agg_order_size() || call.has_agg_filter() || call.has_over() ||
      call.agg_within_group() || call.agg_star() || call.agg_distinct() ||
      call.func_variadic() ||
      call.funcformat() != pg_query::COERCE_EXPLICIT_CALL) {
    emitError(call.location(), call) << "unsupported feature for sum";
    return;
  }

  if (call.args_size() != 1) {
    emitError(call.location(), call) << "expected exactly 1 argument for sum";
    return;
  }

  const auto &arg = call.args(0);
  auto expr = parseTupleExpr(arg);
  if (!expr) {
    return;
  }

  state.aggregate.emplace_back(expr);
  state.aggregators.emplace_back(columnar::Aggregator::SUM);
  state.aggregateNames.emplace_back(columnName(target, expr));
}

void SQLParser::parseAggregateMin(const pg_query::ResTarget &target,
                                  const pg_query::FuncCall &call,
                                  AggregateState &state) {
  // TODO: dedup with SUM
  if (call.agg_order_size() || call.has_agg_filter() || call.has_over() ||
      call.agg_within_group() || call.agg_star() || call.agg_distinct() ||
      call.func_variadic() ||
      call.funcformat() != pg_query::COERCE_EXPLICIT_CALL) {
    emitError(call.location(), call) << "unsupported feature for min";
    return;
  }

  if (call.args_size() != 1) {
    emitError(call.location(), call) << "expected exactly 1 argument for min";
    return;
  }

  const auto &arg = call.args(0);
  auto expr = parseTupleExpr(arg);
  if (!expr) {
    return;
  }

  state.aggregate.emplace_back(expr);
  state.aggregators.emplace_back(columnar::Aggregator::MIN);
  state.aggregateNames.emplace_back(columnName(target, expr));
}

void SQLParser::parseAggregateAvg(const pg_query::ResTarget &target,
                                  const pg_query::FuncCall &call,
                                  AggregateState &state) {
  if (call.agg_order_size() || call.has_agg_filter() || call.has_over() ||
      call.agg_within_group() || call.agg_star() || call.agg_distinct() ||
      call.func_variadic() ||
      call.funcformat() != pg_query::COERCE_EXPLICIT_CALL) {
    emitError(call.location(), call) << "unsupported feature for avg";
    return;
  }

  if (call.args_size() != 1) {
    emitError(call.location(), call) << "expected exactly 1 argument for avg";
    return;
  }

  const auto &arg = call.args(0);
  auto expr = parseTupleExpr(arg);
  if (!expr) {
    return;
  }

  state.aggregate.emplace_back(expr);
  state.aggregators.emplace_back(columnar::Aggregator::AVG);
  state.aggregateNames.emplace_back(columnName(target, expr));
}

void SQLParser::parseAggregateCount(const pg_query::ResTarget &target,
                                    const pg_query::FuncCall &call,
                                    AggregateState &state) {
  if (call.agg_order_size() || call.has_agg_filter() || call.has_over() ||
      call.agg_within_group() || call.agg_distinct() || call.func_variadic() ||
      call.funcformat() != pg_query::COERCE_EXPLICIT_CALL) {
    emitError(call.location(), call) << "unsupported feature for count";
    return;
  }

  if (call.agg_star()) {
    // Special-case COUNT(*)
    // Pick an arbitrary column as input
    if (_currentColumns.empty()) {
      emitError(call.location(), call) << "no rows to aggregate";
      return;
    }

    auto expr = _currentColumns[0];
    state.aggregate.emplace_back(expr);
    state.aggregators.emplace_back(columnar::Aggregator::COUNT_ALL);
    state.aggregateNames.emplace_back(columnName(target, expr));
    return;
  } else if (call.args_size() != 1) {
    emitError(call.location(), call) << "expected exactly 1 argument for count";
    return;
  }

  const auto &arg = call.args(0);
  auto expr = parseTupleExpr(arg);
  if (!expr) {
    return;
  }

  state.aggregate.emplace_back(expr);
  state.aggregators.emplace_back(columnar::Aggregator::COUNT);
  state.aggregateNames.emplace_back(columnName(target, expr));
}

mlir::InFlightDiagnostic
SQLParser::emitError(std::int32_t loc, const google::protobuf::Message &msg) {
  // TODO: location tracking
  return emitError(msg);
}

mlir::InFlightDiagnostic
SQLParser::emitError(const google::protobuf::Message &msg) {
  auto diag = mlir::emitError(_builder.getUnknownLoc());
  diag.attachNote() << "in proto " << msg.DebugString();
  return diag;
}

mlir::Location SQLParser::loc(std::int32_t l) {
  // TODO location tracking.
  return _builder.getUnknownLoc();
}

mlir::Location SQLParser::loc(const pg_query::Node &n) {
  // TODO: location tracking.
  return _builder.getUnknownLoc();
}

mlir::Value SQLParser::findColumn(mlir::Location loc, llvm::StringRef name) {
  auto c = _columnsByName.lookup(name);
  if (c) {
    return c;
  }

  c = _correlatedColumns.lookup(name);
  if (c) {
    return c;
  }

  auto err = mlir::emitError(loc) << "unknown column: " << name;
  for (auto name : _columnsByName.keys()) {
    err.attachNote() << "available: " << name;
  }

  for (auto name : _correlatedColumns.keys()) {
    err.attachNote() << "available (correlated): " << name;
  }

  return nullptr;
}

mlir::StringAttr SQLParser::columnName(const pg_query::ResTarget &target,
                                       mlir::Value column) {
  return target.name().empty() ? columnName(column)
                               : _builder.getStringAttr(target.name());
}

mlir::StringAttr SQLParser::columnName(mlir::Value column) {
  // TODO: dataflow analysis?
  auto res = llvm::dyn_cast<mlir::OpResult>(column);
  if (auto readOp = column.getDefiningOp<columnar::ReadColumnOp>()) {
    return _builder.getStringAttr(readOp.getColumn().getName());
  } else if (auto joinOp = column.getDefiningOp<columnar::JoinOp>()) {
    // Take from input.
    return columnName(joinOp->getOperand(res.getResultNumber()));
  } else if (auto selectOp = column.getDefiningOp<columnar::SelectOp>()) {
    // Take from input.
    return columnName(selectOp->getOperand(res.getResultNumber()));
  }

  // FIXME: Inefficient
  for (const auto &[name, value] : _columnsByName) {
    if (column == value) {
      return _builder.getStringAttr(name);
    }
  }

  return mlir::StringAttr::get(column.getContext(), "NO_NAME");
}

void SQLParser::remapCurrentColumns(const mlir::IRMapping &mapping) {
  for (auto &value : _currentColumns) {
    auto newValue = mapping.lookupOrNull(value);
    if (newValue) {
      value = newValue;
    }
  }

  for (auto k : _columnsByName.keys()) {
    auto &value = _columnsByName[k];
    auto newValue = mapping.lookupOrNull(value);
    if (newValue) {
      value = newValue;
    }
  }
}

mlir::OwningOpRef<mlir::ModuleOp>
SQLParser::parseQuery(mlir::MLIRContext *ctx, mlir::FileLineColLoc loc,
                      const pg_query::ParseResult &proto,
                      const Catalog &catalog) {
  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(module->getBody());
  SQLParser parser(catalog, std::move(builder));

  for (const auto &stmt : proto.stmts()) {
    parser.parseStmt(stmt);
  }

  return module;
}

} // namespace columnar
