#include "parse/Lexer.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <cassert>
#include <charconv>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <system_error>
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
  LogicalResult parseDeclarator(bool &isFuncDef);
  LogicalResult parseDeclaratorAtom();
  LogicalResult parseInitializer();
  LogicalResult parseParameterList();
  LogicalResult parseParameter();

  // Also called 'compound statement'
  LogicalResult parseBlock();
  LogicalResult parseStatement();
  LogicalResult parseReturn();
  // Utility for checking the statment ends with ';'
  LogicalResult parseStatementEndSemi();

  LogicalResult parseExpression();
  LogicalResult parseInt();

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

    bool isFuncDef;
    if (failed(parseDeclarator(isFuncDef))) {
      return reportError(cur()->loc, "invalid declaration");
    }

    if (isFuncDef) {
      // Don't end with a semi (and we don't chain function defs).
      return LogicalResult::success();
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

LogicalResult Parser::parseDeclarator(bool &isFuncDef) {
  isFuncDef = false;
  if (failed(parseDeclaratorAtom())) {
    return LogicalResult::failure();
  }

  // Check for array or function decl
  while (cur() &&
         (cur()->kind == Token::LSBRACKET || cur()->kind == Token::LPAREN)) {
    if (cur()->kind == Token::LSBRACKET) {
      // array
      return reportError(cur()->loc, "not implemented: array declarators");
    } else {
      // function
      assert(cur()->kind == Token::LPAREN);
      eat(); // (

      if (cur() && cur()->kind != Token::RPAREN) {
        if (failed(parseParameterList())) {
          return LogicalResult::failure();
        }
      }

      if (!cur()) {
        return reportError(tokens.back().loc, "unexpected end of declarator");
      }

      eat(); // )

      if (cur() && cur()->kind == Token::LBRACE) {
        // Function definition rather than declaration.
        isFuncDef = true;
        if (failed(parseBlock())) {
          return LogicalResult::failure();
        }
      }

      return LogicalResult::success();
    }
  }

  return LogicalResult::success();
}
LogicalResult Parser::parseDeclaratorAtom() {
  if (!cur()) {
    return reportError(tokens.back().loc, "expected a declarator");
  }

  bool isFuncDef;
  if (cur()->kind == Token::IDENT) {
    // <identifier>
    eat();
    return LogicalResult::success();
  } else if (cur()->kind == Token::LPAREN) {
    // (<declarator>)
    eat();

    if (failed(parseDeclarator(isFuncDef))) {
      return LogicalResult::failure();
    }

    if (cur()->kind != Token::RPAREN) {
      return reportError(tokens.back().loc, "unclosed paren for declarator");
    }
    eat();
    return LogicalResult::success();
  } else if (cur()->kind == Token::ASTERISK) {
    // * <quals> <declarator>
    eat();

    std::vector<Token::Kind> quals;
    while (cur() && isQualifier(cur()->kind)) {
      quals.push_back(cur()->kind);
      eat();
    }

    return parseDeclarator(isFuncDef);
  }

  return reportError(cur()->loc, "expected a declarator");
}
LogicalResult Parser::parseInitializer() {
  return reportError(cur()->loc, "not implemented: initializer");
}
LogicalResult Parser::parseParameterList() {
  // NOTE: we don't support identifier-list format: all types must be explicit
  if (failed(parseParameter())) {
    return LogicalResult::failure();
  }

  while (cur() && cur()->kind == Token::COMMA) {
    eat();

    if (failed(parseParameter())) {
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}

LogicalResult Parser::parseParameter() {
  // parameters are declarations with a single identifier. The identifier is
  // optional.
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

  // 2. declarator
  // TODO: handle optional identifier.
  bool isFuncDef;
  if (failed(parseDeclarator(isFuncDef))) {
    return reportError(cur()->loc, "invalid declaration");
  }

  return LogicalResult::success();
}

LogicalResult Parser::parseBlock() {
  assert(cur()->kind == Token::LBRACE);
  auto openLoc = cur()->loc;
  eat();

  while (cur() && cur()->kind != Token::RBRACE) {
    // Parse statements
    // TODO: can also include declarations
    if (failed(parseStatement())) {
      return LogicalResult::failure();
    }
  }

  if (!cur()) {
    return reportError(openLoc, "unclosed block");
  }

  eat();
  return LogicalResult::success();
}

LogicalResult Parser::parseStatement() {
  // TODO: handle labels
  switch (cur()->kind) {
  case Token::LBRACE:
    // compound statement
    return parseBlock();
  case Token::IF:
  case Token::SWITCH:
  case Token::WHILE:
  case Token::DO:
  case Token::FOR:
  case Token::BREAK:
  case Token::CONTINUE:
    return reportError(cur()->loc, "not implemented");
  case Token::RETURN:
    return parseReturn();
  case Token::GOTO:
    return reportError(cur()->loc, "not implemented");
  default:
    return parseExpression();
  }
}

LogicalResult Parser::parseReturn() {
  assert(cur()->kind == Token::RETURN);
  eat();
  if (failed(parseExpression())) {
    return LogicalResult::failure();
  }

  return parseStatementEndSemi();
}

LogicalResult Parser::parseStatementEndSemi() {
  if (!cur()) {
    return reportError(tokens.back().loc, "unexpected end of statement");
  } else if (cur()->kind != Token::SEMI) {
    return reportError(tokens.back().loc, "expected ';' at end of statement");
  }

  eat(); // ;
  return LogicalResult::success();
}

LogicalResult Parser::parseExpression() {
  if (cur()->kind == Token::INT) {
    return parseInt();
  }

  return reportError(cur()->loc, "unsupported expression");
}

LogicalResult Parser::parseInt() {
  assert(cur()->kind == Token::INT);
  auto body = cur()->body;
  int64_t value;
  auto [ptr, ec] = std::from_chars(body.begin(), body.end(), value);
  assert(ptr == body.end());
  if (ec != std::errc()) {
    return reportError(cur()->loc, "invalid integer value");
  }

  eat();
  return LogicalResult::success();
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