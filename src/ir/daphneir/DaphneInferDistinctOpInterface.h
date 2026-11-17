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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERDISTINCTOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEINFERDISTINCTOPINTERFACE_H

// ****************************************************************************
// Distinct inference interfaces
// ****************************************************************************

#include "ir/daphneir/Daphne.h"
namespace mlir::daphne {
#include <ir/daphneir/DaphneInferDistinctOpInterface.h.inc>
}

// ****************************************************************************
// Distinct inference function
// ****************************************************************************

namespace mlir::daphne {
/**
 * @brief Tries to infer the number of distinct values of a matrix.
 *
 * If any related inference traits are attached to the given operation, these are
 * applied to infer this data property of the result. If the operation implements any related
 * inference interface, that implementation is invoked. If this data property cannot be
 * infered based on the available information, or if the operation does not
 * have any related traits or interfaces, an value representing unknown for this data property will be returned for all
 * results.
 *
 * @param op The operation for whose results' this data property shall be inferred.
 * @return
 */
std::vector<ssize_t> tryInferDistinct(mlir::Operation *op);
} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERDISTINCTOPINTERFACE_H