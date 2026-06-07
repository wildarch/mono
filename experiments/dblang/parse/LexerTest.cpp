#include "util/FileSystem.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <iostream>
#include <string>
#include <vector>

using namespace dblang;

static const char *TESTS_PATH = "experiments/dblang/parse/Lexer.test";

struct TestCase {
  std::string_view input;
  std::string_view expected;
};

static LogicalResult parseTests(std::string_view testsContent,
                                std::vector<TestCase> &out) {
  std::string midSep(80, '-');
  std::string testSep(80, '=');

  // TODO
  std::size_t testStart = 0;
  while (testStart < testsContent.size()) {
    auto mid = testsContent.find(midSep, testStart);
    if (mid == std::string::npos) {
      return reportError(Loc{TESTS_PATH}, "expected mid separator");
    }

    auto end = testsContent.find(testSep, mid);
    if (end == std::string::npos) {
      return reportError(Loc{TESTS_PATH}, "expected end separator");
    }

    auto input = testsContent.substr(testStart, mid - testStart);
    auto expected = testsContent.substr(mid + 80, end);
    out.push_back(TestCase{input, expected});

    testStart = end + 80;
  }

  return LogicalResult::success();
}

int main(int argc, char **argv) {
  std::string testsContent;
  if (failed(readFileToString(TESTS_PATH, testsContent))) {
    return 1;
  }

  std::vector<TestCase> tests;
  if (failed(parseTests(testsContent, tests))) {
    return 1;
  }

  std::cout << "got " << tests.size() << " tests\n";
  for (auto test : tests) {
    std::cout << "input:\n";
    std::cout << test.input;
    std::cout << "expected:\n";
    std::cout << test.expected;
  }

  std::cout << "\n";

  return 0;
}