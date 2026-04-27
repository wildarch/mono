#include "thirdparty/ninja-1.13.2/src/disk_interface.h"
#include "thirdparty/ninja-1.13.2/src/graph.h"
#include "thirdparty/ninja-1.13.2/src/manifest_parser.h"
#include "thirdparty/ninja-1.13.2/src/state.h"
#include <boost/json/array.hpp>
#include <boost/json/impl/serialize.ipp>
#include <boost/json/object.hpp>
#include <iostream>
#include <unistd.h>

#include <boost/json/src.hpp>
using namespace boost::json;

int main(int argc, char **argv) {
  State state;
  RealDiskInterface diskInterface;
  ManifestParserOptions options;
  ManifestParser parser(&state, &diskInterface, options);
  std::string err;
  chdir("experiments/dabu/ninja/build/");
  if (!parser.Load("build.ninja", &err)) {
    std::cerr << err << "\n";
    return 1;
  }

  for (auto *edge : state.edges_) {
    const auto &rule = edge->rule();
    object jsonEdge;
    if (rule.IsPhony()) {
      jsonEdge["Phony"] = true;
    } else {
      // NOTE: command is expected to be executed with 'sh -c'
      jsonEdge["Phony"] = false;
      auto cmd = edge->EvaluateCommand();
      jsonEdge["Command"] = cmd;
    }

    // NOTE: relevant data of node: path
    array inputs;
    for (const auto *node : edge->inputs_) {
      inputs.emplace_back(node->path());
    }

    array outputs;
    for (const auto *node : edge->outputs_) {
      outputs.emplace_back(node->path());
    }

    jsonEdge["Inputs"] = inputs;
    jsonEdge["Outputs"] = outputs;

    std::cout << serialize(jsonEdge) << "\n";
  }

  return 0;
}
