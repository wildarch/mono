#include "util/FileSystem.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <fstream>
#include <string>

namespace dblang {

LogicalResult readFileToString(const std::string &filename, std::string &out) {
  std::ifstream file{filename};
  if (!file.is_open()) {
    return dblang::reportError(dblang::Loc{filename}, "could not open file");
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  if (file.bad()) {
    return dblang::reportError(dblang::Loc{filename}, "error reading file");
  }

  out = buffer.str();
  return dblang::LogicalResult::success();
}

} // namespace dblang