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

#include "ir/daphneir/DaphneAlgebraicTraitPatterns.h"

#include "ir/daphneir/Daphne.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {

// Returns the element type of a matrix-typed value, or a null Type otherwise.
static Type getMatrixElementTypeOrNull(Value v) {
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(v.getType()))
        return mt.getElementType();
    return {};
}

// Returns true when `v` has an integer element type (matrix element type for a
// matrix, the value type itself for a scalar). Absorbing/neutral rewrites that
// drop an operand are only valid over integers; the IEEE float cases they'd
// break (NaN*0, Inf*0, (-0.0)+0.0) are ruled out by keeping them integer-only.
static bool hasIntegerElementType(Value v) {
    Type elemTy = getMatrixElementTypeOrNull(v);
    if (!elemTy)
        elemTy = v.getType();
    return llvm::isa<IntegerType>(elemTy);
}

// Returns true when `v` is a ConstantLike op whose scalar value is zero. Over
// floats only positive zero counts. Negative zero is excluded because the
// neutral/absorbing identities don't hold for it (e.g. x - (-0.0) = x + 0.0
// flips -0.0 to +0.0), so dropping it would miscompute.
static bool isConstantZero(Value v) {
    return matchPattern(v, m_Zero()) || matchPattern(v, m_PosZeroFloat());
}

// Returns true when `v` is a ConstantLike op whose scalar value is one.
static bool isConstantOne(Value v) { return matchPattern(v, m_One()) || matchPattern(v, m_OneFloat()); }

/**
 * @brief Collapses `f(f(x))` to `x` for any op tagged `Involutive`.
 *
 * Only checks that the outer op's result type matches the inner op's operand
 * type, so shape-swapping involutions (e.g. TransposeOp on rectangular
 * matrices) are handled correctly.
 */
struct InvolutivePattern : public RewritePattern {
    InvolutivePattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::Involutive>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Operation *inner = op->getOperand(0).getDefiningOp();
        if (!inner || inner->getName() != op->getName())
            return failure();
        if (inner->getNumOperands() != 1 || inner->getNumResults() != 1)
            return failure();

        Value innerInput = inner->getOperand(0);
        if (innerInput.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, innerInput);
        return success();
    }
};

/**
 * @brief Collapses `f(f(x))` to `f(x)` for any op tagged `IdempotentUnary`.
 *
 * Keeps the inner op's result (this op's operand) rather than its input, which
 * is the difference from `InvolutivePattern`. Since it forwards an
 * already-computed value unchanged, no integer-only or signed-zero/NaN caveat
 * applies. The type-equality guard fail-closes for any adopter whose result
 * type could differ from its operand type.
 */
struct IdempotentUnaryPattern : public RewritePattern {
    IdempotentUnaryPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdempotentUnary>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Operation *inner = op->getOperand(0).getDefiningOp();
        if (!inner || inner->getName() != op->getName())
            return failure();
        if (inner->getNumOperands() != 1 || inner->getNumResults() != 1)
            return failure();

        Value innerResult = op->getOperand(0);
        if (innerResult.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, innerResult);
        return success();
    }
};

/**
 * @brief Collapses `agg(reorder(X))` to `agg(X)`.
 *
 * An order-agnostic reduction over every element (outer op tagged
 * `OrderAgnosticAggregate`) is invariant under a producer that only permutes
 * elements without changing their multiset (inner op tagged
 * `OnlyReordersElements`, e.g. transpose or reverse), so the reorder is dead.
 * Both are Pure, so DCE removes the now-unused inner op.
 *
 * The matched aggregate is rebuilt generically on the reorder's input, so every
 * All-agg reduction adopting the trait (sumAll, meanAll, varAll, stddevAll)
 * folds from one pattern. Like any order-agnostic reduction this reassociates a
 * floating-point sum, which is the tolerance the trait licenses. The inner
 * operand must be a matrix, the only thing an aggregate reduces.
 */
struct ReorderAgnosticAggPattern : public RewritePattern {
    ReorderAgnosticAggPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::OrderAgnosticAggregate>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Operation *inner = op->getOperand(0).getDefiningOp();
        if (!inner || !inner->hasTrait<OpTrait::OnlyReordersElements>())
            return failure();
        if (inner->getNumOperands() != 1 || inner->getNumResults() != 1)
            return failure();
        if (!getMatrixElementTypeOrNull(inner->getOperand(0)))
            return failure();

        // Clone the matched aggregate with its single operand remapped to the
        // reorder's input rather than hard-coding one op class. The reorder
        // preserves the element type, so the clone keeps its already-inferred
        // result type. The arity guards above keep this well-formed for the
        // single-operand All-agg family.
        IRMapping map;
        map.map(op->getOperand(0), inner->getOperand(0));
        Operation *rebuilt = rewriter.clone(*op, map);
        rewriter.replaceOp(op, rebuilt->getResults());
        return success();
    }
};

/**
 * @brief Replaces `f(X)` with `X` for any op tagged `IdentityOnIntegerElementType`
 *        when X has an integer element type and the result type matches.
 */
struct IdentityOnIntegerElementTypePattern : public RewritePattern {
    IdentityOnIntegerElementTypePattern(MLIRContext *ctx)
        : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdentityOnIntegerElementType>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Value operand = op->getOperand(0);
        if (!hasIntegerElementType(operand))
            return failure();
        if (operand.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, operand);
        return success();
    }
};

/**
 * @brief Replaces `f(X)` with `X` when X's inferred symmetric property is true.
 *
 * Bails out on `BoolOrUnknown::Unknown` so the rewrite stays sound under
 * imprecise property inference.
 */
struct IdentityWhenSymmetricPattern : public RewritePattern {
    IdentityWhenSymmetricPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdentityWhenSymmetric>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Value operand = op->getOperand(0);
        auto mt = llvm::dyn_cast<daphne::MatrixType>(operand.getType());
        if (!mt || mt.getSymmetric() != BoolOrUnknown::True)
            return failure();
        if (operand.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, operand);
        return success();
    }
};

/**
 * @brief Replaces `f(x, 0)` with `x` for any op tagged `NeutralOnZeroRHS`.
 *
 * `isConstantZero` already excludes a negative-zero RHS (see there). On top of
 * that, `ewAdd` is suppressed over any floating-point element type, because
 * `(-0.0) + 0.0` is `+0.0`: even a positive-zero RHS would drop a negative-zero
 * lhs. `ewSub` needs no such guard, since `x - 0.0 = x` holds for every IEEE
 * value once a negative-zero RHS is ruled out.
 */
struct NeutralOnZeroRHSPattern : public RewritePattern {
    NeutralOnZeroRHSPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::NeutralOnZeroRHS>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();
        if (!isConstantZero(op->getOperand(1)))
            return failure();
        if (llvm::isa<daphne::EwAddOp>(op) && !hasIntegerElementType(op->getResult(0)))
            return failure();

        Value lhs = op->getOperand(0);
        if (lhs.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, lhs);
        return success();
    }
};

/**
 * @brief Replaces `f(x, 1)` with `x` for any op tagged `NeutralOnOneRHS`.
 */
struct NeutralOnOneRHSPattern : public RewritePattern {
    NeutralOnOneRHSPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::NeutralOnOneRHS>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();
        if (!isConstantOne(op->getOperand(1)))
            return failure();

        Value lhs = op->getOperand(0);
        if (lhs.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, lhs);
        return success();
    }
};

/**
 * @brief Replaces `f(0, x)` with `0` for any op tagged `LeftAbsorbingOnZero`.
 *
 * Restricted to integer element types: over IEEE floating point `0 * x = 0` is
 * false (NaN * 0 = NaN, Inf * 0 = NaN). Fires only when the zero operand's type
 * equals the op's result type, so broadcasting cases (scalar zero * matrix) do
 * not shrink the result to a scalar.
 */
struct LeftAbsorbingOnZeroPattern : public RewritePattern {
    LeftAbsorbingOnZeroPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::LeftAbsorbingOnZero>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();
        Value zero = op->getOperand(0);
        if (!isConstantZero(zero))
            return failure();
        if (!hasIntegerElementType(op->getResult(0)))
            return failure();
        if (zero.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, zero);
        return success();
    }
};

/**
 * @brief Replaces `f(x, 0)` with `0` for any op tagged `RightAbsorbingOnZero`.
 *
 * Restricted to integer element types for the same reason as
 * `LeftAbsorbingOnZeroPattern`.
 */
struct RightAbsorbingOnZeroPattern : public RewritePattern {
    RightAbsorbingOnZeroPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::RightAbsorbingOnZero>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();
        Value zero = op->getOperand(1);
        if (!isConstantZero(zero))
            return failure();
        if (!hasIntegerElementType(op->getResult(0)))
            return failure();
        if (zero.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, zero);
        return success();
    }
};

} // namespace

namespace mlir::daphne {

void populateAlgebraicTraitPatterns(RewritePatternSet &patterns) {
    MLIRContext *ctx = patterns.getContext();
    patterns.add<InvolutivePattern, IdempotentUnaryPattern, IdentityOnIntegerElementTypePattern, IdentityWhenSymmetricPattern,
                 ReorderAgnosticAggPattern, NeutralOnZeroRHSPattern, NeutralOnOneRHSPattern,
                 LeftAbsorbingOnZeroPattern, RightAbsorbingOnZeroPattern>(ctx);
}

} // namespace mlir::daphne
