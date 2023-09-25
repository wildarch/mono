#pragma once

#include <cstdint>
#include <vector>

#include <mlir/IR/Operation.h>

/*
Core operations:
- Load vector
- Load constant
- Binop
*/

namespace execution {
namespace expression {
namespace bytecode {

enum class Op : uint8_t {
  LOAD_I32,
  CONSTANT_I32,
  ADD_I32,
  SUB_I32,
  MUL_I32,
  DIV_I32,
};

using reg_t = uint8_t;

struct Instruction {
  Op op;
  reg_t rd;
  union {
    struct {
      reg_t ra;
      reg_t rb;
    };
    int32_t val_i32;
    uint16_t col_idx;
  };
};

/**
 * Sethi-Ullman Register Allocation.
 * https://pages.cs.wisc.edu/~horwitz/CS701-NOTES/5.REGISTER-ALLOCATION.html
 *
 * The algorithm is modified somewhat because we do not support operating
 * directly on memory addresses. This has implications for the number of
 * registers needed for right nodes.
 */
class RegisterAllocator {
private:
  llvm::DenseMap<mlir::Value, reg_t> _registersNeeded;

public:
  /**
   * Finds the minimal number of registers needed for every op in the tree.
   *
   * Corresponds to step 1 of the Sethi-Ullman algorithm.
   */
  uint16_t analyseRegistersNeeded(mlir::Value val);
};

class ByteCodeCompiler {
private:
  std::vector<Instruction> _code;

  enum class Reg {
    A,
    B,
  };

  reg_t compile(mlir::Value val, reg_t nextFreeReg);

public:
  static std::vector<Instruction> compile(mlir::Value val);
};

} // namespace bytecode
} // namespace expression
} // namespace execution