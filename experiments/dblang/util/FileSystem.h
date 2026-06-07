#include "util/Result.h"
#include <string>

namespace dblang {

LogicalResult readFileToString(const std::string &filename, std::string &out);

} // namespace dblang