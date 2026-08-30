#pragma once

namespace dblang::ir {

struct Operation;

/** A sequence of operations. */
class Block {
private:
  Operation *head;
  Operation *tail;
};

} // namespace dblang::ir