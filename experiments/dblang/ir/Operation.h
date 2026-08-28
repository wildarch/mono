#pragma once

namespace dblang::ir {

/**
 * Operation memory layout is inspired by MLIR:
 * - OpResult 2
 * - OpResult 1
 * - OpResult 0
 * - Operation class (the header)
 * - Operand 0
 * - Operand 1
 * - Operand 2
 * - Block 0
 * - Block 1
 * - Block 2
 * - Other properties
 *
 * OpResult stores:
 * - Use chain: pointer to the first OpOperand
 * - Type
 * - Result index
 * Since Type is 8-byte aligned, we can steal the low 3 bits of the type pointer
 * for the result index.
 *
 * Operation header stores:
 * - Operation kind
 * - Source location
 * - prev/next pointers for the doubly-linked list of operations within a block
 * - Number of operands
 * - Number of results
 * - Number of regions
 *
 * Operand stores:
 * - Pointer to the SSA value (an OpResult or a block argument)
 * - Next use of the same value (OpOperand*)
 * - Back pointer to the previous pointer pointing to this OpOperand (for fast
 * removal from the linked list)
 */

class Operation {
private:
  // results
  // operands
  // attributes
};

} // namespace dblang::ir