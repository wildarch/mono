#include "parse/Lexer.h"
#include "util/FileSystem.h"
#include "util/ReportError.h"
#include "util/Result.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace dblang;

static const char *TESTS_PATH = "experiments/dblang/parse/Lexer.test";
static const char *PROPOSE_PATH = "/tmp/Lexer.test";

struct TestCase {
  std::string_view input;
  std::string_view expected;
  std::string actual;
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
    auto expectedStart = mid + 80;
    auto expected = testsContent.substr(expectedStart, end - expectedStart);
    out.push_back(TestCase{input, expected});

    testStart = end + 80;
  }

  return LogicalResult::success();
}

static LogicalResult runTest(TestCase &test) {
  std::vector<Token> tokens;
  // TODO: allow setting start location in file.
  if (failed(lex(TESTS_PATH, test.input, tokens))) {
    return LogicalResult::failure();
  }

  std::stringstream ss;
  for (const auto &token : tokens) {
    ss << token << "\n";
  }

  test.actual = ss.str();
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

  std::cout << "running " << tests.size() << " tests\n";
  std::size_t pass = 0;
  std::size_t fail = 0;
  for (auto &test : tests) {
    runTest(test);
    if (test.actual == test.expected) {
      pass++;
    } else {
      fail++;
    }
  }

  std::cout << pass << "/" << tests.size() << " pass\n";
  if (fail) {
    std::cout << fail << "/" << tests.size() << " FAIL\n";

    std::ofstream out(PROPOSE_PATH);
    std::string midSep(80, '-');
    std::string testSep(80, '=');

    for (const auto &test : tests) {
      out << test.input;
      out << midSep << "\n";
      out << test.actual;
      out << testSep << "\n";
    }

    std::cout << "wrote candidate test file to " << PROPOSE_PATH << "\n";
    std::cout << "to compare, run:\n";
    std::cout << "    vimdiff " << TESTS_PATH << " " << PROPOSE_PATH << "\n";
    std::cout << "to accept the current diff:\n";
    std::cout << "    cp " << PROPOSE_PATH << " " << TESTS_PATH << "\n";
  }

  return 0;
}