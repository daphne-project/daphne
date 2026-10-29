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

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DaphneAlgebraicTraitPatterns.h"
#include "ir/daphneir/LinearAlgebraRewrites.h"
#include "ir/daphneir/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/**
 * @brief Applies algebraic simplification rewrites to the DAPHNE IR.
 *
 * The pass runs two families of rewrite drivers against every op in the module:
 * - The trait-driven patterns from `populateAlgebraicTraitPatterns` each gate on
 *   a specific op trait (Involutive, NeutralOnZeroRHS, etc.), so the pass picks
 *   up new adopters of those traits without further modification.
 * - The structural patterns from `populateLinearAlgebraRewritePatterns` instead
 *   match a specific chain of ops and replace it with a cheaper equivalent one.
 */
struct AlgebraicSimplifyPass : public PassWrapper<AlgebraicSimplifyPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<daphne::DaphneDialect>(); }

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        daphne::populateAlgebraicTraitPatterns(patterns);
        daphne::populateLinearAlgebraRewritePatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }

    StringRef getArgument() const final { return "daphne-algebraic-simplify"; }
    StringRef getDescription() const final {
        return "Simplifies DaphneIR using rewrite drivers keyed on algebraic op traits.";
    }
};

} // namespace

std::unique_ptr<Pass> mlir::daphne::createAlgebraicSimplifyPass() { return std::make_unique<AlgebraicSimplifyPass>(); }
