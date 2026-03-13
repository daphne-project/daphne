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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERMNCSKETCHIDOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEINFERMNCSKETCHIDOPINTERFACE_H

// ****************************************************************************
// MncSketchId inference interfaces
// ****************************************************************************
#include "ir/daphneir/Daphne.h"
namespace mlir::daphne {
#include <ir/daphneir/DaphneInferMncSketchIdOpInterface.h.inc>
}

// ****************************************************************************
// MncSketchId inference function
// ****************************************************************************

namespace mlir::daphne {
/**
 * @brief Tries to infer whether the results of the given operation are symmetric.
 *
 * If any related inference traits are attached to the given operation, these are
 * applied to infer this data property of the result. If the operation implements any related
 * inference interface, that implementation is invoked. If this data property cannot be
 * inferred based on the available information, or if the operation does not
 * have any related traits or interfaces, an value representing unknown for this data property will be returned for all
 * results.
 *
 * @param op The operation for whose results' this data property shall be inferred.
 * @return
 */
std::vector<int64_t> tryInferMncSketchId(mlir::Operation *op);
} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNEINFERMNCSKETCHIDOPINTERFACE_H