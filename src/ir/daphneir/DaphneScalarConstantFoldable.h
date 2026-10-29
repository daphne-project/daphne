/*
 * Copyright 2026 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_H
#define SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>

namespace mlir::daphne {
#include <ir/daphneir/DaphneScalarConstantFoldable.h.inc>
} // namespace mlir::daphne

namespace mlir::daphne {

/**
 * @brief Fold a scalar op with constant operands by delegating to the op's
 *        `ScalarConstantFoldable` methods.
 *
 * The driver tries the numeric kinds in the same precedence the legacy per-op
 * templates used (float, then bool, then signed int, unsigned int) and returns
 * the first non-empty result. Bool comes before int because a BoolAttr is an i1
 * IntegerAttr, so it would otherwise be caught by the int path. If none of the
 * kinds apply, a null Attribute is returned, which the caller should propagate
 * as "no folding possible".
 *
 * Arity is inferred from `operands.size()`; the driver dispatches to the
 * unary or binary methods accordingly. Comparison ops (numeric in, bool out)
 * are recognised by the `ValueTypeCmp` trait, not the result type: a comparison
 * carries the most-general argument value type (e.g. si64/f64), never i1, so a
 * result-type heuristic would never route them to the `foldScalarCmp*` methods.
 *
 * String semantics and division-by-zero raising are intentionally out of
 * scope; the corresponding ops keep their bespoke folders in `Fold.cpp`.
 */
::mlir::Attribute foldScalarOp(ScalarConstantFoldable op, ::llvm::ArrayRef<::mlir::Attribute> operands,
                               ::mlir::Type resultType, ::mlir::Location loc);

} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_H
