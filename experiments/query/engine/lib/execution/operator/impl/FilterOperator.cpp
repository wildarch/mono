#include "execution/operator/impl/FilterOperator.h"
#include "execution/Batch.h"
#include "execution/expression/ExpressionEvaluator.h"
#include <stdexcept>
#include <variant>

namespace execution {

static bool evaluatePredicate(mlir::Operation *expr, const Batch &batch,
                              uint32_t row) {
  auto res = evaluate(expr, batch, row);
  if (!std::holds_alternative<bool>(res)) {
    throw std::invalid_argument("Predicate does not return bool");
  }

  return std::get<bool>(res);
}

template <PhysicalColumnType type>
static void copyRow(const Batch &input, uint32_t inputRow, Batch &output,
                    uint32_t outputRow, size_t col) {
  output.columnsForWrite().at(col).getForWrite<type>()[outputRow] =
      input.columns().at(col).get<type>()[inputRow];
}

static void copyRow(const Batch &input, uint32_t inputRow, Batch &output,
                    uint32_t outputRow) {
  for (size_t i = 0; i < input.columns().size(); i++) {
    switch (input.columns().at(i).type()) {
#define CASE(t)                                                                \
  case PhysicalColumnType::t:                                                  \
    copyRow<PhysicalColumnType::t>(input, inputRow, output, outputRow, i);     \
    break;

      CASE(INT32)
      CASE(DOUBLE)
      CASE(STRING_PTR)
#undef CASE
    }
  }
}

std::optional<Batch> FilterOperator::poll() {
  auto input = child()->poll();
  if (!input) {
    return std::nullopt;
  }

  std::vector<PhysicalColumnType> columnTypes;
  for (const auto &c : input->columns()) {
    columnTypes.push_back(c.type());
  }

  Batch output(columnTypes, input->rows());

  uint32_t outputRow = 0;
  for (uint32_t inputRow = 0; inputRow < input->rows(); inputRow++) {
    if (!evaluatePredicate(_expr, *input, inputRow)) {
      continue;
    }

    copyRow(*input, inputRow, output, outputRow);
    outputRow++;
  }
  output.setRows(outputRow);

  return output;
}

} // namespace execution