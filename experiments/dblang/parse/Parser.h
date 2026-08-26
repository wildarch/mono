#pragma once

#include "parse/Lexer.h"
#include "util/Result.h"
#include <span>

namespace dblang {

LogicalResult parseDef(std::span<const Token> tokens);

} // namespace dblang