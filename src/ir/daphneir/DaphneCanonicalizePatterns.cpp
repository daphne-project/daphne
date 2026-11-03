/*
 * Copyright 2024 The DAPHNE Consortium
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

// Registers the declarative (DRR) canonicalization patterns generated from
// DaphneCanonicalizePatterns.td.

#include "ir/daphneir/Daphne.h"

#include "mlir/IR/PatternMatch.h"

namespace {
#include <ir/daphneir/DaphneCanonicalizePatterns.inc>
} // namespace

void mlir::daphne::RenameOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
    populateWithGenerated(results);
}
