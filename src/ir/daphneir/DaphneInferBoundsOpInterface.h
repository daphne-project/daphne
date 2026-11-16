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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERBOUNDSOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEINFERBOUNDSOPINTERFACE_H

// ****************************************************************************
// Bounds inference interfaces
// ****************************************************************************

#include "ir/daphneir/Daphne.h"
namespace mlir::daphne {
#include <ir/daphneir/DaphneInferBoundsOpInterface.h.inc>
}

// ****************************************************************************
// Bounds inference function
// ****************************************************************************

namespace mlir::daphne {
/**
 * @brief Tries to infer lower and upper value bounds for each result of the operation.
 *
 * If the operation implements InferBounds, its implementation is invoked.
 * Otherwise, both bounds are unknown for every result. Each endpoint is represented
 * independently by an optional double; std::nullopt denotes an unknown bound.
 *
 * @param op The operation whose result bounds shall be inferred.
 * @return One pair of lower and upper bounds per result.
 */
std::vector<std::pair<std::optional<double>, std::optional<double>>> tryInferBounds(mlir::Operation *op);
} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERBOUNDSOPINTERFACE_H
