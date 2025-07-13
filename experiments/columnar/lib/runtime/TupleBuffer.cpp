#include "columnar/runtime/TupleBuffer.h"
#include <llvm/Support/Alignment.h>

namespace columnar::runtime {

TupleArena::TupleArena(std::size_t tupleSize, std::size_t tupleAlignment)
    : _tupleSizeAligned(llvm::alignTo(tupleSize, tupleAlignment)) {}

TupleBufferLocal::TupleBufferLocal(std::size_t tupleSize,
                                   std::size_t tupleAlignment)
    : _partitions{TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment),
                  TupleArena(tupleSize, tupleAlignment)} {}

} // namespace columnar::runtime
