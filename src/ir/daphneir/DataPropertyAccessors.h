/*
 * Copyright 2025 The DAPHNE Consortium
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

#ifndef SRC_IR_DAPHNEIR_DATAPROPERTYACCESSORS_H
#define SRC_IR_DAPHNEIR_DATAPROPERTYACCESSORS_H

#include <ir/daphneir/DataPropertyTypes.h>

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include <optional>

// Fail-closed accessors for the inferred data properties (shape, sparsity) that
// DAPHNE stores on the MLIR type of a matrix or frame value. Each accessor
// returns std::nullopt when the property is unknown, either because the type
// is not a matrix/frame or because the property still holds its "unknown"
// sentinel (numRows/numCols == -1, sparsity == -1.0).
//
// This matters because the canonicalizer runs both before property inference has
// filled these in and after it, so a rewrite may see a property unknown.
// Comparing the raw getter against a threshold is unsafe there: the -1.0 sentinel
// compares as an ordinary number, so `getSparsity() < 0.25` is true for an
// uninferred sparsity. These accessors only yield a value once the property is
// genuinely known:
//
//     if (std::optional<double> sp = knownSparsity(v); sp && *sp < 0.25)
//         ... // safe: never entered when sparsity is unknown

namespace mlir::daphne {

// Returns the statically known number of rows of `type`, or nullopt if the row
// extent is unknown or `type` is not a matrix/frame.
std::optional<ssize_t> knownNumRows(mlir::Type type);

// Returns the statically known number of columns of `type`, or nullopt if the
// column extent is unknown or `type` is not a matrix/frame.
std::optional<ssize_t> knownNumCols(mlir::Type type);

// Returns the statically known sparsity (nonzero fraction) of `type`, or nullopt
// if the sparsity is unknown or `type` is not a matrix. Frames carry no sparsity.
std::optional<double> knownSparsity(mlir::Type type);

// Convenience overloads reading the property off the type of a value.
inline std::optional<ssize_t> knownNumRows(mlir::Value value) { return knownNumRows(value.getType()); }
inline std::optional<ssize_t> knownNumCols(mlir::Value value) { return knownNumCols(value.getType()); }
inline std::optional<double> knownSparsity(mlir::Value value) { return knownSparsity(value.getType()); }

} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DATAPROPERTYACCESSORS_H
