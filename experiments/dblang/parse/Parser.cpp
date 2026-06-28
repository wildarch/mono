#include "parse/Lexer.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <algorithm>
#include <cassert>
#include <charconv>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dblang {

namespace {

struct TypeSpec {
  enum Kind {
    VOID,
    BOOL,
    CHAR,
    INT,
    FLOAT,
    DOUBLE,
    LONG,
    SHORT,
    UNSIGNED,
    SIGNED,
    ATOMIC,
    TYPEDEF, /* to a previous typedef. */
    STRUCT,
    UNION,
    ENUM,
  } kind;
  std::string_view name; // for struct, union, enum
};

enum class StorageClass {
  TYPEDEF,
  REGISTER,
  STATIC,
  EXTERN,
  THREAD_LOCAL,
};

enum class TypeQualifier {
  CONST,
  VOLATILE,
  RESTRICT,
  ATOMIC,
};

struct Declarator {
  enum Kind {
    IDENT,
    PTR,
    ARRAY,
    FUNC,
  } kind;
};

template <Declarator::Kind KIND, typename T>
struct DeclaratorBase : public Declarator {
  DeclaratorBase() { this->kind = KIND; }

  static T *dynCast(Declarator *self) {
    if (self->kind == KIND) {
      return static_cast<T *>(self);
    }

    return nullptr;
  }

  static const T *dynCast(const Declarator *self) {
    if (self->kind == KIND) {
      return static_cast<const T *>(self);
    }

    return nullptr;
  }
};

struct DeclaratorIdent : DeclaratorBase<Declarator::IDENT, DeclaratorIdent> {
  std::string_view ident;
  DeclaratorIdent(std::string_view ident) : ident(ident) {}
};

struct DeclaratorPtr : DeclaratorBase<Declarator::PTR, DeclaratorPtr> {
  Declarator *inner;
  DeclaratorPtr(Declarator *inner) : inner(inner) {}
};

struct DeclaratorArray : DeclaratorBase<Declarator::ARRAY, DeclaratorArray> {
  Declarator *base;
  // TODO: expression
  DeclaratorArray(Declarator *base) : base(base) {}
};

struct Declaration {
  std::vector<TypeSpec> specs;
  std::vector<StorageClass> storage;
  std::vector<TypeQualifier> quals;
  std::vector<Declarator *> declarators;
  // TODO: function specifiers
  // TODO: alignment specifiers.
};

struct DeclaratorFunc : DeclaratorBase<Declarator::FUNC, DeclaratorFunc> {
  Declarator *ret;
  std::vector<Declaration> params;
  DeclaratorFunc(Declarator *ret, std::vector<Declaration> &&params)
      : ret(ret), params(std::move(params)) {}
};

class Parser {
private:
  std::span<const Token> tokens;
  std::size_t offset = 0;

  // TODO: Map to type
  std::unordered_set<std::string> typedefs;

  std::optional<Token> peek(std::size_t ahead) const {
    if (offset + ahead < tokens.size()) {
      return tokens[offset + ahead];
    } else {
      return std::nullopt;
    }
  }

  std::optional<Token> cur() const { return peek(0); }

  void eat() { offset++; }

  LogicalResult parseSpecifierOrQualifier(Declaration &decl);
  LogicalResult parseSpecifierStruct(TypeSpec &spec);
  LogicalResult parseSpecifierEnum(TypeSpec &spec);
  LogicalResult parseSpecifierUnion(TypeSpec &spec);
  LogicalResult parseDeclaration(Declaration &decl);
  LogicalResult parseDeclarator(Declarator *&decl, bool allowAnonymous);
  LogicalResult parseDeclaratorAtom(Declarator *&decl, bool allowAnonymous);
  LogicalResult parseInitializer();
  LogicalResult parseParameterList(std::vector<Declaration> &params);
  LogicalResult parseParameter(Declaration &decl);

  // Also called 'compound statement'
  LogicalResult parseBlock();
  LogicalResult parseStatement();
  LogicalResult parseReturn();
  // Utility for checking the statment ends with ';'
  LogicalResult parseStatementEndSemi();

  LogicalResult parseType();

  LogicalResult parseExpression();
  LogicalResult parseFunctionCall();
  LogicalResult parseArrayAccess();
  LogicalResult parseExpressionAtom();
  LogicalResult parseInt();

  template <typename T, typename... Args> T *build(Args &&...args) {
    // TODO: proper allocator that can free things
    return new T(std::forward<Args>(args)...);
  }

  LogicalResult finish(const Declaration &decl);

public:
  Parser(std::span<const Token> tokens) : tokens(tokens) {
    typedefs.insert("size_t");
  }

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

LogicalResult Parser::parseSpecifierOrQualifier(Declaration &decl) {
  switch (cur()->kind) {
  // Qualifiers
  case Token::CONST:
    eat();
    decl.quals.push_back(TypeQualifier::CONST);
    return LogicalResult::success();
  case Token::VOLATILE:
    eat();
    decl.quals.push_back(TypeQualifier::VOLATILE);
    return LogicalResult::success();
  case Token::RESTRICT:
    eat();
    decl.quals.push_back(TypeQualifier::RESTRICT);
    return LogicalResult::success();
  // Specifiers
  case Token::VOID:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::VOID});
    return LogicalResult::success();
  // Storage class
  case Token::TYPEDEF:
    eat();
    decl.storage.push_back(StorageClass::TYPEDEF);
    return LogicalResult::success();
  case Token::STATIC:
    eat();
    decl.storage.push_back(StorageClass::STATIC);
    return LogicalResult::success();
  case Token::EXTERN:
    eat();
    decl.storage.push_back(StorageClass::EXTERN);
    return LogicalResult::success();
  // Arithmetic type
  case Token::BOOL:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::BOOL});
    return LogicalResult::success();
  case Token::CHAR_KW:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::CHAR});
    return LogicalResult::success();
  case Token::INT_KW:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::INT});
    return LogicalResult::success();
  case Token::FLOAT_KW:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::FLOAT});
    return LogicalResult::success();
  case Token::DOUBLE:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::DOUBLE});
    return LogicalResult::success();
  case Token::SHORT:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::SHORT});
    return LogicalResult::success();
  case Token::LONG:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::LONG});
    return LogicalResult::success();
  case Token::SIGNED:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::SIGNED});
    return LogicalResult::success();
  case Token::UNSIGNED:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::UNSIGNED});
    return LogicalResult::success();
  case Token::STRUCT:
    return parseSpecifierStruct(decl.specs.emplace_back());
  case Token::ENUM:
    return parseSpecifierEnum(decl.specs.emplace_back());
  case Token::UNION:
    return parseSpecifierUnion(decl.specs.emplace_back());
  case Token::IDENT: {
    // Check if known typedef
    auto name = cur()->body;
    if (!typedefs.contains(std::string(name))) {
      return LogicalResult::failure();
    }

    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::TYPEDEF, name});
    return LogicalResult::success();
  }
  // Atomic
  case Token::ATOMIC:
    eat();
    decl.specs.push_back(TypeSpec{TypeSpec::ATOMIC});

    if (cur() && cur()->kind == Token::LPAREN) {
      // _Atomic(..)
      eat();

      // HACK: pretend like there is no wrapping going on here
      while (cur()->kind != Token::RPAREN) {
        if (failed(parseSpecifierOrQualifier(decl))) {
          return LogicalResult::failure();
        }
      }

      eat(); // )
    }

    return LogicalResult::success();
  default:
    return LogicalResult::failure();
  }
}
LogicalResult Parser::parseSpecifierStruct(TypeSpec &spec) {
  assert(cur()->kind == Token::STRUCT);
  eat(); // struct
  spec.kind = TypeSpec::STRUCT;

  if (!cur()) {
    return reportError(tokens.back().loc,
                       "unexpected end of struct declaration");
  } else if (cur()->kind == Token::IDENT) {
    spec.name = cur()->body;
    eat();
  }

  // may also include a struct decl.
  if (cur() && cur()->kind == Token::LBRACE) {
    eat(); // {
    while (cur() && cur()->kind != Token::RBRACE) {
      Declaration decl;
      if (failed(parseDeclaration(decl))) {
        return LogicalResult::failure();
      }
    }

    if (!cur()) {
      return reportError(tokens.back().loc, "unexpected end of struct fields");
    }

    eat(); // }
    return LogicalResult::success();
  }

  return LogicalResult::success();
}

LogicalResult Parser::parseSpecifierEnum(TypeSpec &spec) {
  assert(cur()->kind == Token::ENUM);
  eat(); // enum
  spec.kind = TypeSpec::ENUM;

  if (!cur()) {
    return reportError(tokens.back().loc, "unexpected end of enum declaration");
  } else if (cur()->kind == Token::IDENT) {
    spec.name = cur()->body;
    eat();
  }

  // MUST include enum options (C does not do forward decls for enums)
  if (!cur() || cur()->kind != Token::LBRACE) {
    {
      return reportError(tokens.back().loc,
                         "unexpected end of enum declaration");
    }
  }

  eat(); // {

  auto parseEnumOption = [&]() -> LogicalResult {
    if (!cur()) {
      return reportError(tokens.back().loc, "unexpected end of enum");
    } else if (cur()->kind != Token::IDENT) {
      return reportError(cur()->loc, "expected identifier");
    }

    eat(); // ident

    if (cur() && cur()->kind == Token::EQUAL) {
      eat(); // =
      return reportError(cur()->loc, "not supported: expression parse");
    }

    return LogicalResult::success();
  };

  if (failed(parseEnumOption())) {
    return LogicalResult::failure();
  }

  while (cur() && cur()->kind == Token::COMMA) {
    eat(); // ,
    if (failed(parseEnumOption())) {
      return LogicalResult::failure();
    }
  }

  if (!cur() || cur()->kind != Token::RBRACE) {
    return reportError(tokens.back().loc, "unexpected end of enum");
  }

  eat(); // }
  return LogicalResult::success();
}

LogicalResult Parser::parseSpecifierUnion(TypeSpec &spec) {
  assert(cur()->kind == Token::UNION);
  eat(); // union
  spec.kind = TypeSpec::UNION;

  if (!cur()) {
    return reportError(tokens.back().loc,
                       "unexpected end of union declaration");
  } else if (cur()->kind == Token::IDENT) {
    spec.name = cur()->body;
    eat();
  }

  // may also include a union decl.
  if (cur() && cur()->kind == Token::LBRACE) {
    eat(); // {
    while (cur() && cur()->kind != Token::RBRACE) {
      Declaration decl;
      if (failed(parseDeclaration(decl))) {
        return LogicalResult::failure();
      }
    }

    if (!cur()) {
      return reportError(tokens.back().loc, "unexpected end of union fields");
    }

    eat(); // }
    return LogicalResult::success();
  }

  return LogicalResult::success();
}

static bool isFunc(const Declarator &decl) {
  switch (decl.kind) {
  case Declarator::IDENT:
    return false;
  case Declarator::PTR:
    return isFunc(*DeclaratorPtr::dynCast(&decl)->inner);
  case Declarator::ARRAY:
    return isFunc(*DeclaratorArray::dynCast(&decl)->base);
  case Declarator::FUNC:
    return true;
  }
}

LogicalResult Parser::parseDeclaration(Declaration &decl) {
  auto start = cur()->loc;

  // 1. specifiers-and-qualifiers
  while (cur()) {
    if (failed(parseSpecifierOrQualifier(decl))) {
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

    auto &d = decl.declarators.emplace_back(nullptr);
    if (failed(parseDeclarator(d, false))) {
      return reportError(cur()->loc, "invalid declaration");
    }

    if (isFunc(*d)) {
      // Don't end with a semi (and we don't chain function defs).
      return LogicalResult::success();
    }

    // bit-width
    // TODO: only allowed for fields
    if (cur() && cur()->kind == Token::COLON) {
      eat(); // :
      if (failed(parseExpression())) {
        return reportError(cur()->loc, "invalid bit-width");
      }
    }

    if (cur() && cur()->kind == Token::ASSIGN) {
      eat();
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

LogicalResult Parser::parseDeclarator(Declarator *&decl, bool allowAnonymous) {
  if (failed(parseDeclaratorAtom(decl, allowAnonymous))) {
    return LogicalResult::failure();
  }

  // Check for array or function decl
  while (cur() &&
         (cur()->kind == Token::LSBRACKET || cur()->kind == Token::LPAREN)) {
    if (cur()->kind == Token::LSBRACKET) {
      // array
      eat(); // [

      if (cur() && cur()->kind != Token::RSBRACKET) {
        // Expression for array size
        if (failed(parseExpression())) {
          return LogicalResult::failure();
        }
      }

      if (!cur()) {
        return reportError(tokens.back().loc, "unexpected end of declarator");
      } else if (cur()->kind != Token::RSBRACKET) {
        return reportError(tokens.back().loc, "expected array closer ']'");
      }

      eat(); // ]
      decl = build<DeclaratorArray>(decl);
      return LogicalResult::success();
    } else {
      // function
      assert(cur()->kind == Token::LPAREN);
      eat(); // (

      std::vector<Declaration> params;
      if (cur() && cur()->kind != Token::RPAREN) {
        if (failed(parseParameterList(params))) {
          return LogicalResult::failure();
        }
      }

      if (!cur()) {
        return reportError(tokens.back().loc, "unexpected end of declarator");
      }

      eat(); // )

      if (cur() && cur()->kind == Token::LBRACE) {
        // Function definition rather than declaration.
        // TODO: include in decl
        if (failed(parseBlock())) {
          return LogicalResult::failure();
        }
      }

      decl = build<DeclaratorFunc>(decl, std::move(params));
      return LogicalResult::success();
    }
  }

  return LogicalResult::success();
}
LogicalResult Parser::parseDeclaratorAtom(Declarator *&decl,
                                          bool allowAnonymous) {
  if (!cur()) {
    return reportError(tokens.back().loc, "expected a declarator");
  }

  bool isFuncDef;
  if (cur()->kind == Token::IDENT) {
    // <identifier>
    auto ident = cur()->body;
    eat();
    decl = build<DeclaratorIdent>(ident);
    return LogicalResult::success();
  } else if (cur()->kind == Token::LPAREN) {
    // (<declarator>)
    eat();

    if (failed(parseDeclarator(decl, allowAnonymous))) {
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

    // assert(quals.empty());

    Declarator *inner;
    if (failed(parseDeclarator(inner, allowAnonymous))) {
      return LogicalResult::failure();
    }

    decl = build<DeclaratorPtr>(inner);
    return LogicalResult::success();
  } else if (allowAnonymous) {
    decl = build<DeclaratorIdent>("<anonymous>");
    return LogicalResult::success();
  }

  return reportError(cur()->loc, "expected a declarator");
}
LogicalResult Parser::parseInitializer() {
  if (cur() && cur()->kind == Token::LBRACE) {
    eat(); // {

    bool first = true;
    while (cur() && cur()->kind != Token::RBRACE) {
      if (first) {
        first = false;
      } else {
        if (cur()->kind != Token::COMMA) {
          return reportError(cur()->loc, "expected comma separator");
        }

        eat(); // ,
      }

      if (cur() && cur()->kind == Token::RBRACE) {
        // trailing comma
        break;
      }

      if (failed(parseInitializer())) {
        return LogicalResult::failure();
      }
    }

    if (!cur()) {
      return reportError(tokens.back().loc, "unexpected end of initializer");
    }

    eat(); // }
    return LogicalResult::success();
  }

  return parseExpression();
}
LogicalResult Parser::parseParameterList(std::vector<Declaration> &params) {
  if (cur()->kind == Token::VOID && peek(1) && peek(1)->kind == Token::RPAREN) {
    // (void)
    eat();
    return LogicalResult::success();
  }

  // NOTE: we don't support identifier-list format: all types must be explicit
  auto &decl = params.emplace_back();
  if (failed(parseParameter(decl))) {
    return LogicalResult::failure();
  }

  while (cur() && cur()->kind == Token::COMMA) {
    eat();

    if (cur() && cur()->kind == Token::ELLIPSIS) {
      // TODO: store ellipsis
      eat();
      continue;
    }

    auto &decl = params.emplace_back();
    if (failed(parseParameter(decl))) {
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}

LogicalResult Parser::parseParameter(Declaration &decl) {
  // parameters are declarations with a single identifier. The identifier is
  // optional.
  // 1. specifiers-and-qualifiers
  while (cur()) {
    if (failed(parseSpecifierOrQualifier(decl))) {
      break;
    }
  }

  // 2. declarator
  bool isFuncDef;
  auto &d = decl.declarators.emplace_back();
  if (failed(parseDeclarator(d, true))) {
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
  case Token::SEMI:
    // Empty statement
    eat(); // ;
    return LogicalResult::success();
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
    if (failed(parseExpression())) {
      return LogicalResult::failure();
    }

    if (!cur()) {
      reportError(tokens.back().loc, "expected ';'");
    } else if (cur()->kind != Token::SEMI) {
      reportError(cur()->loc, "expected ';'");
    }

    eat(); // ;
    return LogicalResult::success();
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

LogicalResult Parser::parseType() {
  Declaration dummy;
  while (true) {
    if (succeeded(parseSpecifierOrQualifier(dummy))) {
      continue;
    }

    // HACK: eat pointers too
    if (cur() && cur()->kind == Token::ASTERISK) {
      eat();
      continue;
    }

    break;
  }

  if (dummy.specs.empty() && dummy.quals.empty()) {
    // Parsed nothing.
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

LogicalResult Parser::parseFunctionCall() {
  assert(cur()->kind == Token::LPAREN);
  auto loc = cur()->loc;
  eat(); // (

  bool first = true;
  while (cur() && cur()->kind != Token::RPAREN) {
    if (first) {
      first = false;
    } else {
      if (cur()->kind != Token::COMMA) {
        return reportError(cur()->loc, "expected comma");
      }

      eat(); // ,
    }

    if (failed(parseExpression())) {
      return LogicalResult::failure();
    }
  }

  if (!cur()) {
    return reportError(loc, "unclosed function call paren");
  }

  eat(); // )
  return LogicalResult::success();
}

LogicalResult Parser::parseArrayAccess() {
  assert(cur()->kind == Token::LSBRACKET);
  auto loc = cur()->loc;
  eat(); // [

  if (failed(parseExpression())) {
    return LogicalResult::failure();
  }

  if (!cur()) {
    return reportError(loc, "unclosed ']'");
  } else if (cur()->kind != Token::RSBRACKET) {
    return reportError(cur()->loc, "expected ']'");
  }

  eat(); // ]
  return LogicalResult::success();
}

LogicalResult Parser::parseExpression() {
  if (failed(parseExpressionAtom())) {
    return LogicalResult::failure();
  }

  if (cur() && cur()->kind == Token::LPAREN) {
    if (failed(parseFunctionCall())) {
      return LogicalResult::failure();
    }
  }

  if (cur() && cur()->kind == Token::LSBRACKET) {
    if (failed(parseArrayAccess())) {
      return LogicalResult::failure();
    }
  }

  bool stopClimb = false;
  while (cur() && !stopClimb) {
    switch (cur()->kind) {
    case Token::ASSIGN:
    case Token::PLUS_EQ:
    case Token::PLUS:
    case Token::DAMPERSAND:
    case Token::DPIPE:
    case Token::DOT:
    case Token::MINUS:
    case Token::ASTERISK:
    case Token::EQUAL:
    case Token::NOT_EQUAL:
    case Token::LEQ:
    case Token::GEQ:
    case Token::LANGLE:
    case Token::RANGLE:
    case Token::SLASH: {
      eat();

      if (failed(parseExpression())) {
        return LogicalResult::failure();
      }

      break;
    }
    case Token::QMARK: {
      eat(); // ?

      if (failed(parseExpression())) {
        return LogicalResult::failure();
      }

      if (!cur()) {
        return reportError(tokens.back().loc, "unexpected end of ternary");
      } else if (cur()->kind != Token::COLON) {
        return reportError(cur()->loc, "expected ':' as part of ternary");
      }

      eat(); // :

      if (failed(parseExpression())) {
        return LogicalResult::failure();
      }

      break;
    }
    default:
      stopClimb = true;
      break;
    }
  }

  return LogicalResult::success();
}

LogicalResult Parser::parseExpressionAtom() {
  switch (cur()->kind) {
  case Token::INT:
    return parseInt();
  case Token::STRING:
    // NOTE: looping here to handle e.g. "hello" "world" -> "helloworld"
    while (cur() && cur()->kind == Token::STRING) {
      eat();
    }
    return LogicalResult::success();
  case Token::LPAREN: {
    auto loc = cur()->loc;
    eat(); // (

    if (succeeded(parseType())) {
      // type-cast
      if (!cur()) {
        return reportError(loc, "unclosed '('");
      } else if (cur()->kind != Token::RPAREN) {
        return reportError(cur()->loc, "invalid cast, expected ')'");
      }

      eat(); // )

      if (failed(parseExpressionAtom())) {
        return LogicalResult::failure();
      }

      return LogicalResult::success();
    } else {
      // A parenthesized expression

      if (failed(parseExpression())) {
        return LogicalResult::failure();
      }

      if (!cur()) {
        return reportError(loc, "unclosed '('");
      } else if (cur()->kind != Token::RPAREN) {
        return reportError(cur()->loc, "invalid expression, expected ')'");
      }

      eat(); // )
      return LogicalResult::success();
    }
  }
  case Token::ASTERISK: {
    eat(); // *
    if (failed(parseExpressionAtom())) {
      return LogicalResult::failure();
    }

    return LogicalResult::success();
  }
  case Token::AMPERSAND: {
    eat(); // &
    if (failed(parseExpressionAtom())) {
      return LogicalResult::failure();
    }

    return LogicalResult::success();
  }
  case Token::IDENT:
  case Token::SIZEOF: {
    eat();
    return LogicalResult::success();
  }
  default:
    return reportError(cur()->loc, "unsupported expression");
  }
}

LogicalResult Parser::parseInt() {
  assert(cur()->kind == Token::INT);
  /*
  auto body = cur()->body;
  int64_t value;
  auto [ptr, ec] = std::from_chars(body.begin(), body.end(), value);
  if (ec != std::errc() || ptr != body.end()) {
    return reportError(cur()->loc, "invalid integer value '") << body << "'";
  }
  */

  eat();
  return LogicalResult::success();
}

static std::string_view nameOf(const Declarator *decl) {
  if (auto ident = DeclaratorIdent::dynCast(decl)) {
    return ident->ident;
  } else if (auto ptr = DeclaratorPtr::dynCast(decl)) {
    return nameOf(ptr->inner);
  } else if (auto arr = DeclaratorArray::dynCast(decl)) {
    return nameOf(arr->base);
  } else {
    auto func = DeclaratorFunc::dynCast(decl);
    assert(!!func);
    return nameOf(func->ret);
  }
}

LogicalResult Parser::finish(const Declaration &decl) {
  // Check for typedef
  bool isTypeDef = std::ranges::any_of(decl.storage, [](const auto &stor) {
    return stor == StorageClass::TYPEDEF;
  });
  if (isTypeDef) {
    for (const auto *deor : decl.declarators) {
      auto name = nameOf(deor);
      typedefs.insert(std::string(name));
    }
  }

  return LogicalResult::success();
}

LogicalResult Parser::parseFile() {
  while (cur()) {
    Declaration decl;
    auto loc = cur()->loc;
    if (failed(parseDeclaration(decl))) {
      return reportError(loc, "expected declaration");
    }

    if (failed(finish(decl))) {
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}

LogicalResult parse(std::span<const Token> tokens) {
  Parser parser(tokens);
  return parser.parseFile();
}

} // namespace dblang