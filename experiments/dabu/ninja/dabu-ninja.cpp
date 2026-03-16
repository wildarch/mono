#include "thirdparty/ninja-1.13.2/src/disk_interface.h"
#include "thirdparty/ninja-1.13.2/src/graph.h"
#include "thirdparty/ninja-1.13.2/src/manifest_parser.h"
#include "thirdparty/ninja-1.13.2/src/state.h"
#include <iostream>

void walk(const Node *, int);
void walk(const Edge *, int);

void walk(const Node *node, int level) {
  for (int i = 0; i < level; i++)
    std::cout << "    ";
  std::cout << "node " << node->path() << "\n";
  if (node->in_edge()) {
    walk(node->in_edge(), level);
  }
}

void walk(const Edge *edge, int level) {
  for (int i = 0; i < level; i++)
    std::cout << "    ";
  std::cout << "edge rule " << edge->rule().name() << "\n";
  std::cout << "edge command: " << edge->EvaluateCommand() << "\n";
  for (const auto *in : edge->inputs_) {
    walk(in, level + 1);
  }
}

int main(int argc, char **argv) {
  std::cout << "Hello, world!\n";

  State state;
  RealDiskInterface diskInterface;
  ManifestParserOptions options;
  ManifestParser parser(&state, &diskInterface, options);
  std::string err;
  if (!parser.Load("build.ninja", &err)) {
    std::cerr << err << "\n";
    return 1;
  }

  auto ninjaBinary = state.LookupNode("ninja");
  if (!ninjaBinary) {
    std::cerr << "no 'ninja' node found\n";
    return 1;
  }

  walk(ninjaBinary, 0);

  return 0;
}
