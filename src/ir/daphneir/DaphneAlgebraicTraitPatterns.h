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

#ifndef SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITPATTERNS_H
#define SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITPATTERNS_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::daphne {

/**
 * @brief Adds one rewrite pattern per algebraic trait to `patterns`.
 *
 * Each pattern matches any op carrying the corresponding trait, so this
 * populator does not need to be updated when new adopters are added.
 */
void populateAlgebraicTraitPatterns(RewritePatternSet &patterns);

} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITPATTERNS_H
