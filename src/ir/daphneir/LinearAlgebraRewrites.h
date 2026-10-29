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

#ifndef SRC_IR_DAPHNEIR_LINEARALGEBRAREWRITES_H
#define SRC_IR_DAPHNEIR_LINEARALGEBRAREWRITES_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::daphne {

/**
 * @brief Adds the linear-algebra simplification rewrites to `patterns`.
 *
 * These are structural rewrites that match a specific chain of DAPHNE ops (as
 * opposed to the trait-driven patterns in DaphneAlgebraicTraitPatterns) and
 * replace it with a cheaper but numerically equivalent chain. Each rewrite is a
 * strict win and fails closed when the shape properties it relies on are not yet
 * inferred.
 */
void populateLinearAlgebraRewritePatterns(RewritePatternSet &patterns);

} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_LINEARALGEBRAREWRITES_H
