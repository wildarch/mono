#include "thirdparty/ninja-1.13.2/src/disk_interface.h"
#include "thirdparty/ninja-1.13.2/src/graph.h"
#include "thirdparty/ninja-1.13.2/src/manifest_parser.h"
#include "thirdparty/ninja-1.13.2/src/state.h"
#include <iostream>
#include <unistd.h>

void walk(const Node *, int);
void walk(const Edge *, int);

static void indent(int level) {
  for (int i = 0; i < level; i++)
    std::cout << "  ";
}

void walk(const Node *node, int level) {
  indent(level);
  std::cout << "node " << node->path() << "\n";
  if (node->in_edge()) {
    walk(node->in_edge(), level + 1);
  }
}

void walk(const Edge *edge, int level) {
  indent(level);
  std::cout << "edge rule " << edge->rule().name() << "\n";

  indent(level);
  std::cout << "edge command: \n";
  indent(level + 1);
  std::cout << edge->EvaluateCommand() << "\n";

  indent(level);
  std::cout << "inputs:\n";
  for (const auto *in : edge->inputs_) {
    walk(in, level + 1);
  }

  indent(level);
  std::cout << "outputs:\n";
  for (const auto *out : edge->outputs_) {
    indent(level + 1);
    std::cout << "node (out) " << out->path() << "\n";
  }
}

int main(int argc, char **argv) {
  std::cout << "Hello, world!\n";

  State state;
  RealDiskInterface diskInterface;
  ManifestParserOptions options;
  ManifestParser parser(&state, &diskInterface, options);
  std::string err;
  chdir("experiments/dabu/cmake-hello/build/");
  if (!parser.Load("build.ninja", &err)) {
    std::cerr << err << "\n";
    return 1;
  }

  /*
  auto ninjaBinary = state.LookupNode("hello");
  if (!ninjaBinary) {
    std::cerr << "no 'ninja' node found\n";
    return 1;
  }

  walk(ninjaBinary, 0);
  */

  // NOTE: relevant data of edge:
  // - command
  // - inputs
  // - outputs
  for (auto *edge : state.edges_) {
    const auto &rule = edge->rule();
    if (rule.IsPhony()) {
      std::cout << "phony (edge)\n";
    } else {
      // NOTE: command is expected to be executed with 'sh -c'
      auto cmd = edge->EvaluateCommand();
      std::cout << "edge: " << cmd << "\n";
    }

    // NOTE: relevant data of node: path
    for (const auto *node : edge->inputs_) {
      std::cout << "  in " << node->path() << "\n";
    }

    for (const auto *node : edge->outputs_) {
      std::cout << "  out " << node->path() << "\n";
    }
  }

  return 0;
}
