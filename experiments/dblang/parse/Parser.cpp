#include "parse/Lexer.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <optional>
#include <span>
#include <vector>

namespace dblang {

namespace {

class Parser {
private:
  std::span<const Token> tokens;
  std::size_t offset = 0;

  std::optional<Token> cur() {
    if (offset < tokens.size()) {
      return tokens[offset];
    } else {
      return std::nullopt;
    }
  }

  void eat() { offset++; }

  LogicalResult parseSpecifier();
  LogicalResult parseDeclaration();
  LogicalResult parseFuncDef();

public:
  Parser(std::span<const Token> tokens) : tokens(tokens) {}

  LogicalResult parseFile();
};

} // namespace

static bool isQualifier(Token::Kind kind) {
  switch (kind) {
  case Token::CONST:
  case Token::VOLATILE:
  case Token::RESTRICT:
    return true;
  default:
    return false;
  }
}

LogicalResult Parser::parseSpecifier() {
  if (!cur()) {
    return LogicalResult::failure();
  }

  switch (cur()->kind) {
  case Token::VOID:
    eat();
    break;
    // storage class
  case Token::TYPEDEF:
    eat();
    break;
  case Token::STATIC:
    eat();
    break;
  case Token::EXTERN:
    eat();
    break;
    // arithmetic types
  case Token::BOOL:
    eat();
    break;
  case Token::CHAR_KW:
    eat();
    break;
  case Token::INT_KW:
    eat();
    break;
  case Token::FLOAT_KW:
    eat();
    break;
  case Token::DOUBLE:
    eat();
    break;
  case Token::SIGNED:
  case Token::UNSIGNED:
  case Token::SHORT:
  case Token::LONG:
    return reportError(cur()->loc, "unsupported type");
  default:
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

LogicalResult Parser::parseDeclaration() {
  // 1. specifiers-and-qualifiers
  std::vector<Token::Kind> quals;
  while (cur()) {
    if (isQualifier(cur()->kind)) {
      quals.push_back(cur()->kind);
      eat();
      continue;
    } else if (succeeded(parseSpecifier())) {
      continue;
    } else {
      return LogicalResult::failure();
    }
  }

  // 2. declarators-and-initializers

  // ;
  if (cur() && cur()->kind == Token::SEMI) {
    eat();
    return LogicalResult::success();
  } else if (cur()) {
    return reportError(cur()->loc, "invalid declaration");
  } else {
    return reportError(tokens.back().loc, "unexpected end of declaration");
  }
}
LogicalResult Parser::parseFuncDef() { return LogicalResult::failure(); }

LogicalResult Parser::parseFile() {
  while (cur()) {
    if (succeeded(parseDeclaration())) {
      continue;
    } else if (succeeded(parseFuncDef())) {
      continue;
    } else {
      return reportError(cur()->loc,
                         "expected declaration or function definition");
    }
  }

  return LogicalResult::success();
}

LogicalResult parse(std::span<const Token> tokens) {
  Parser parser(tokens);
  return parser.parseFile();
}

} // namespace dblang