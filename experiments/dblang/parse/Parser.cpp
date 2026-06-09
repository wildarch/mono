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
  LogicalResult parseDeclarator();
  LogicalResult parseNoPtrDeclarator();
  LogicalResult parseInitializer();
  LogicalResult parseParameterList();

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
      break;
    }
  }

  // 2. declarators-and-initializers
  bool first = true;
  while (cur() && cur()->kind != Token::SEMI) {
    if (first) {
      first = false;
    } else {
      // comma separated
      if (cur()->kind != Token::COMMA) {
        return reportError(cur()->loc,
                           "expected , to separate declarators (or ;)");
      }

      eat();
    }

    if (failed(parseDeclarator())) {
      return reportError(cur()->loc, "invalid declaration");
    }

    if (cur() && cur()->kind == Token::EQUAL) {
      if (failed(parseInitializer())) {
        return reportError(cur()->loc, "invalid initializer");
      }
    }
  }

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

LogicalResult Parser::parseDeclarator() {
  if (!cur()) {
    return reportError(tokens.back().loc, "expected a declarator");
  } else if (cur()->kind != Token::ASTERISK) {
    return parseNoPtrDeclarator();
  }

  // Pointer declarator
  eat();

  std::vector<Token::Kind> quals;
  while (cur() && isQualifier(cur()->kind)) {
    quals.push_back(cur()->kind);
    eat();
  }

  return parseDeclarator();
}
LogicalResult Parser::parseNoPtrDeclarator() {
  if (!cur()) {
    return reportError(tokens.back().loc, "expected a declarator");
  }

  if (cur()->kind == Token::IDENT) {
    eat();
    return LogicalResult::success();
  } else if (cur()->kind == Token::LPAREN) {
    eat();

    if (failed(parseDeclarator())) {
      return LogicalResult::failure();
    }

    if (cur()->kind != Token::RPAREN) {
      return reportError(tokens.back().loc, "unclosed paren for declarator");
    }
    eat();
    return LogicalResult::success();
  } else {
    // array or function
    if (failed(parseNoPtrDeclarator())) {
      return LogicalResult::failure();
    }

    if (cur() && cur()->kind == Token::LSBRACKET) {
      // array
      return reportError(cur()->loc, "not implemented: array declarators");
    } else if (cur() && cur()->kind == Token::LPAREN) {
      // function
      eat();

      if (failed(parseParameterList())) {
        return LogicalResult::failure();
      }

      if (cur()->kind != Token::RPAREN) {
        return reportError(tokens.back().loc,
                           "unclosed paren for function params");
      }
      eat();
      return LogicalResult::success();
    } else {
      return reportError(cur()->loc, "invalid declarator");
    }
  }
}
LogicalResult Parser::parseInitializer() {
  return reportError(cur()->loc, "not implemented: initializer");
}
LogicalResult Parser::parseParameterList() {
  return reportError(cur()->loc, "not implemented: parameter list");
}

LogicalResult Parser::parseFile() {
  while (cur()) {
    if (failed(parseDeclaration())) {
      return reportError(cur()->loc, "expected declaration");
    }
  }

  return LogicalResult::success();
}

LogicalResult parse(std::span<const Token> tokens) {
  Parser parser(tokens);
  return parser.parseFile();
}

} // namespace dblang