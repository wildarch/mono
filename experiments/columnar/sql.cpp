#include <iostream>

#include "pg_query.h"
#include "pg_query.pb.h"

static bool parseQuery(const std::string &query,
                       pg_query::ParseResult &result) {
  bool success = true;

  // Use libpg_query to parse the SQL string into a binary protobuf.
  auto res = pg_query_parse_protobuf(query.c_str());
  if (res.error) {
    success = false;
    std::cerr << "parse error: " << res.error->message << " at "
              << res.error->cursorpos << std::endl;
  }

  // Parse the binary proto using the generated C++ class.
  if (!result.ParseFromArray(res.parse_tree.data, res.parse_tree.len)) {
    std::cerr << "re-parse of protobuf failed\n";
    success = false;
  }

  pg_query_free_protobuf_parse_result(res);
  return success;
}

int main(int argc, char **argv) {
  std::string query;
  for (std::string line; std::getline(std::cin, line);) {
    query += line;
    query += '\n';
  }

  pg_query::ParseResult resProto;
  if (!parseQuery(query, resProto)) {
    return 1;
  }

  for (const auto &stmt : resProto.stmts()) {
    const auto &node = stmt.stmt();
    if (!node.has_select_stmt()) {
      std::cerr << "Not a select statement: " << node.DebugString()
                << std::endl;
      return 1;
    }

    const auto &select = node.select_stmt();
    select.PrintDebugString();
  }

  return 0;
}