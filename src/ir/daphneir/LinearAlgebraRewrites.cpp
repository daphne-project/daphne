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

#include "ir/daphneir/LinearAlgebraRewrites.h"

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DataPropertyAccessors.h"

#include "compiler/utils/CompilerUtils.h"

using namespace mlir;

namespace {

// Returns the element type of a matrix-typed value, or a null Type otherwise.
static Type getMatrixElementTypeOrNull(Value v) {
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(v.getType()))
        return mt.getElementType();
    return {};
}

// Returns true when `elem` is an integer element type wider than one bit, the
// condition under which the arithmetic-regrouping rewrites are value-preserving:
// integer add/mul are associative in Z/2^n, so regrouping them changes nothing,
// whereas over floats it rounds differently. A one-bit integer is excluded too,
// since the emitted EwMul cannot hold a coefficient of 2 (and rejects strings).
static bool isRewritableIntElem(Type elem) {
    auto intTy = llvm::dyn_cast<IntegerType>(elem);
    return intTy && intTy.getWidth() > 1;
}

/**
 * @brief Rewrites the trace idiom `sum(diag(X @ Y))` to `sum(X * t(Y))`.
 *
 * DAPHNE has no trace op, so the trace of a matrix product is spelled
 * `sum(diagVector(X @ Y))`. Using the identity
 *
 *     trace(X @ Y) = sum_i (X @ Y)[i, i]
 *                  = sum_i sum_j X[i, j] * Y[j, i]
 *                  = sum(X (elementwise*) t(Y)),
 *
 * this replaces an O(n^2 * m) matrix product, whose off-diagonal entries the
 * diagVector immediately discards, with an O(n * m) element-wise product.
 *
 * The rewrite fails closed, matching only a non-transposed product of operands
 * that share an element type and whose result is statically known to be square;
 * see the per-guard comments below for why each is needed.
 */
struct TraceIdiomPattern : public OpRewritePattern<daphne::AllAggSumOp> {
    using OpRewritePattern<daphne::AllAggSumOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::AllAggSumOp sumOp, PatternRewriter &rewriter) const override {
        auto diagOp = sumOp.getArg().getDefiningOp<daphne::DiagVectorOp>();
        if (!diagOp)
            return failure();

        auto matMulOp = diagOp.getArg().getDefiningOp<daphne::MatMulOp>();
        if (!matMulOp)
            return failure();

        // Only the plain, non-transposed product is the trace idiom we handle.
        if (CompilerUtils::constantOrDefault<bool>(matMulOp.getTransa(), false) ||
            CompilerUtils::constantOrDefault<bool>(matMulOp.getTransb(), false))
            return failure();

        Value x = matMulOp.getLhs();
        Value y = matMulOp.getRhs();

        // The emitted ewMul must be well-typed: X and Y need the same element type.
        Type elemX = getMatrixElementTypeOrNull(x);
        Type elemY = getMatrixElementTypeOrNull(y);
        if (!elemX || !elemY || elemX != elemY)
            return failure();

        // X @ Y must be square: X is m x n, Y is n x m. Fail closed on unknown
        // shape (the diagVector kernel would throw at runtime otherwise, so a
        // non-square case is a malformed program either way).
        std::optional<ssize_t> rowsX = daphne::knownNumRows(x);
        std::optional<ssize_t> colsX = daphne::knownNumCols(x);
        std::optional<ssize_t> rowsY = daphne::knownNumRows(y);
        std::optional<ssize_t> colsY = daphne::knownNumCols(y);
        if (!rowsX || !colsX || !rowsY || !colsY)
            return failure();
        if (*colsX != *rowsY || *rowsX != *colsY)
            return failure();

        // sum(X * t(Y)): t(Y) is an m x n matrix matching X, so the element-wise
        // product needs no broadcasting and has X's shape. Both created ops are
        // given concrete result types computed from the known extents: shape
        // inference has already run by the time this pass fires, so a shape left
        // unknown here would never be resolved and kernel dispatch would fail.
        // The element-wise product's sparsity and symmetry, however, are NOT X's:
        // an element-wise product is at most as dense as either factor, so
        // reusing X's type would over-state the density (and X's symmetry says
        // nothing about X * t(Y)). Reset both to unknown so a later inference pass
        // derives them, rather than baking in a wrong value that representation
        // selection would then consume.
        auto xMatTy = llvm::cast<daphne::MatrixType>(x.getType());
        auto yMatTy = llvm::cast<daphne::MatrixType>(y.getType());
        Value transposedY =
            rewriter.create<daphne::TransposeOp>(matMulOp.getLoc(), yMatTy.withShape(*colsY, *rowsY), y);
        Type ewMulTy = xMatTy.withSparsity(-1.0).withSymmetric(daphne::BoolOrUnknown::Unknown);
        Value ewMul = rewriter.create<daphne::EwMulOp>(sumOp.getLoc(), ewMulTy, x, transposedY);
        rewriter.replaceOpWithNewOp<daphne::AllAggSumOp>(sumOp, sumOp.getResult().getType(), ewMul);
        return success();
    }
};

/**
 * @brief Rewrites the row-scaling idiom `diag(v) @ X` to `X * v`.
 *
 * Multiplying by a diagonal matrix on the left scales each row: with v an
 * n x 1 column vector and X an n x m matrix, `diag(v) @ X` is n x m and its
 * row i equals `v[i] * X[i, :]`. The element-wise product `X * v` broadcasts
 * the column vector down the rows and computes exactly that, so the rewrite
 * replaces an O(n^2 * m) matrix product against a materialized n x n diagonal
 * with an O(n * m) element-wise product (and, since diagMatrix is Pure, lets
 * DCE drop the now-dead diagonal).
 *
 * Operand order matters here: the element-wise kernel broadcasts a column
 * vector only when the matrix is the left operand, so the matrix must be the
 * lhs and the vector the rhs. That order is also safe from later reordering:
 * EwMul's canonicalizer only swaps a scalar lhs, and neither operand here is a
 * scalar.
 *
 * The rewrite fails closed, matching only a non-transposed product whose
 * diagonal argument is a statically-known column vector, whose operands share
 * an element type, and whose row extents statically agree (so the broadcast
 * branch is guaranteed to fire); see the per-guard comments below.
 */
struct RowScalePattern : public OpRewritePattern<daphne::MatMulOp> {
    using OpRewritePattern<daphne::MatMulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::MatMulOp matMulOp, PatternRewriter &rewriter) const override {
        auto diagOp = matMulOp.getLhs().getDefiningOp<daphne::DiagMatrixOp>();
        if (!diagOp)
            return failure();

        // Only the plain, non-transposed product carries the row-scaling identity.
        if (CompilerUtils::constantOrDefault<bool>(matMulOp.getTransa(), false) ||
            CompilerUtils::constantOrDefault<bool>(matMulOp.getTransb(), false))
            return failure();

        Value v = diagOp.getArg();
        Value x = matMulOp.getRhs();

        // The emitted ewMul must be well-typed: X and v need the same element type.
        Type elemV = getMatrixElementTypeOrNull(v);
        Type elemX = getMatrixElementTypeOrNull(x);
        if (!elemV || !elemX || elemV != elemX)
            return failure();

        // v must be a statically-known n x 1 column vector, and X's row extent
        // must statically match it, so the element-wise kernel is guaranteed to
        // take the column-vector broadcast branch (matrix rows == vector rows,
        // vector has one column). Fail closed on any unknown or mismatched
        // extent: a wrong broadcast would silently miscompute rather than throw.
        std::optional<ssize_t> rowsV = daphne::knownNumRows(v);
        std::optional<ssize_t> colsV = daphne::knownNumCols(v);
        std::optional<ssize_t> rowsX = daphne::knownNumRows(x);
        if (!rowsV || !colsV || !rowsX)
            return failure();
        if (*colsV != 1 || *rowsV != *rowsX)
            return failure();

        // X * v: matrix as lhs, column vector as rhs (the only order the
        // broadcast kernel handles, and the order EwMul's canonicalizer leaves
        // untouched since neither operand is a scalar). The product's shape
        // matches diag(v) @ X, so reuse the matmul's extents, but reset sparsity
        // and symmetry to unknown: they describe the matrix product, not the
        // element-wise scaling, so leave them for a later inference pass to derive.
        auto resTy = llvm::cast<daphne::MatrixType>(matMulOp.getResult().getType());
        Type ewMulTy = resTy.withSparsity(-1.0).withSymmetric(daphne::BoolOrUnknown::Unknown);
        rewriter.replaceOpWithNewOp<daphne::EwMulOp>(matMulOp, ewMulTy, x, v);
        return success();
    }
};

/**
 * @brief Rewrites the column-scaling idiom `X @ diag(v)` to `X * t(v)`.
 *
 * Multiplying by a diagonal matrix on the right scales each column: with v an
 * m x 1 column vector and X an n x m matrix, `X @ diag(v)` is n x m and its
 * column j equals `v[j] * X[:, j]`. Transposing v to the 1 x m row vector t(v)
 * and forming the element-wise product `X * t(v)` broadcasts that row vector
 * down the rows of X and computes exactly this, so the rewrite replaces an
 * O(n * m^2) matrix product against a materialized m x m diagonal with an
 * O(n * m) element-wise product plus a cheap transpose (and, since diagMatrix
 * is Pure, lets DCE drop the now-dead diagonal).
 *
 * The transpose is needed here: the element-wise kernel broadcasts across
 * columns only when the vector is a row vector (1 x m), so v (m x 1) must be
 * transposed. Operand order matters too: that broadcast branch is
 * anchored on the matrix being the left operand, so the matrix must be the lhs
 * and the row vector the rhs. The order is safe from later reordering:
 * EwMul's canonicalizer only swaps a scalar lhs, and neither operand here is a
 * scalar.
 *
 * The rewrite fails closed, matching only a non-transposed product whose
 * diagonal argument is a statically-known column vector, whose operands share
 * an element type, and whose column extent statically agrees with the vector
 * length (so the broadcast branch is guaranteed to fire); see the per-guard
 * comments below.
 */
struct ColScalePattern : public OpRewritePattern<daphne::MatMulOp> {
    using OpRewritePattern<daphne::MatMulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::MatMulOp matMulOp, PatternRewriter &rewriter) const override {
        auto diagOp = matMulOp.getRhs().getDefiningOp<daphne::DiagMatrixOp>();
        if (!diagOp)
            return failure();

        // Only the plain, non-transposed product carries the column-scaling identity.
        if (CompilerUtils::constantOrDefault<bool>(matMulOp.getTransa(), false) ||
            CompilerUtils::constantOrDefault<bool>(matMulOp.getTransb(), false))
            return failure();

        Value x = matMulOp.getLhs();
        Value v = diagOp.getArg();

        // The emitted ewMul must be well-typed: X and v need the same element type.
        Type elemV = getMatrixElementTypeOrNull(v);
        Type elemX = getMatrixElementTypeOrNull(x);
        if (!elemV || !elemX || elemV != elemX)
            return failure();

        // v must be a statically-known m x 1 column vector, and X's column extent
        // must statically match its length, so t(v) is a 1 x m row vector whose
        // width equals X's and the element-wise kernel is guaranteed to take the
        // row-vector broadcast branch (matrix cols == vector cols, vector has one
        // row). Compare against X's columns here: the diagonal is on the right,
        // so it scales columns, unlike the row-scaling pattern. Fail closed on any
        // unknown or mismatched extent: a wrong broadcast would silently
        // miscompute rather than throw.
        std::optional<ssize_t> rowsV = daphne::knownNumRows(v);
        std::optional<ssize_t> colsV = daphne::knownNumCols(v);
        std::optional<ssize_t> colsX = daphne::knownNumCols(x);
        if (!rowsV || !colsV || !colsX)
            return failure();
        if (*colsV != 1 || *rowsV != *colsX)
            return failure();

        // X * t(v): matrix as lhs, row vector as rhs (the only order the broadcast
        // kernel handles, and the order EwMul's canonicalizer leaves untouched since
        // neither operand is a scalar). t(v) is given a concrete 1 x m result type:
        // shape inference has already run by the time this pass fires, so a shape
        // left unknown here would never be resolved and kernel dispatch would fail.
        // The product's shape matches X @ diag(v), so reuse the matmul's extents,
        // but reset sparsity and symmetry to unknown: they describe the matrix
        // product, not the element-wise scaling, so leave them for a later
        // inference pass to derive.
        auto vMatTy = llvm::cast<daphne::MatrixType>(v.getType());
        Value tv = rewriter.create<daphne::TransposeOp>(matMulOp.getLoc(), vMatTy.withShape(1, *rowsV), v);
        auto resTy = llvm::cast<daphne::MatrixType>(matMulOp.getResult().getType());
        Type ewMulTy = resTy.withSparsity(-1.0).withSymmetric(daphne::BoolOrUnknown::Unknown);
        rewriter.replaceOpWithNewOp<daphne::EwMulOp>(matMulOp, ewMulTy, x, tv);
        return success();
    }
};

/**
 * @brief Rewrites the scalar-factor idiom `sum(s * X)` to `s * sum(X)`.
 *
 * A scalar factor distributes over a total sum: `sum(s * X) = s * sum(X)`. This
 * hoists the multiply out of the O(n * m) element-wise product into a single
 * scalar multiply after the aggregate (and, since the element-wise product is
 * Pure, lets DCE drop it when the sum was its only user).
 *
 * The rewrite fails closed. It matches only an element-wise product of exactly
 * one scalar and one matrix operand (in either order, since EwMul is commutative
 * and its canonicalizer may swap a scalar lhs to the rhs), and only over integer
 * matrices whose element type equals the sum's result type; see the per-guard
 * comments.
 */
struct SumScalarFactorPattern : public OpRewritePattern<daphne::AllAggSumOp> {
    using OpRewritePattern<daphne::AllAggSumOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::AllAggSumOp sumOp, PatternRewriter &rewriter) const override {
        auto mulOp = sumOp.getArg().getDefiningOp<daphne::EwMulOp>();
        if (!mulOp)
            return failure();

        // Require exactly one scalar and one matrix operand; accept either order
        // (EwMul is commutative). Fail closed otherwise.
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        Value scalar;
        Value matrix;
        if (CompilerUtils::hasScaType(lhs) && getMatrixElementTypeOrNull(rhs)) {
            scalar = lhs;
            matrix = rhs;
        } else if (CompilerUtils::hasScaType(rhs) && getMatrixElementTypeOrNull(lhs)) {
            scalar = rhs;
            matrix = lhs;
        } else {
            return failure();
        }

        // Regrouping the factor out of the sum is only value-preserving over
        // integers; the check rules out non-associative floats (and the
        // non-numeric scalars hasScaType would otherwise admit).
        Type elemMatrix = getMatrixElementTypeOrNull(matrix);
        if (!isRewritableIntElem(elemMatrix))
            return failure();

        // The aggregate accumulates in its result type, so sum(s * X) accumulates
        // in the product's type while s * sum(X) accumulates in X's. Fire only
        // when they match, else a narrower accumulation could overflow differently.
        if (elemMatrix != sumOp.getResult().getType())
            return failure();

        // Give sum(X) X's element type (which the guards proved equals the result
        // type); an unknown type would never resolve, as inference has already run.
        // The outer s * sum(X) is a scalar-by-scalar product reusing that type.
        Value innerSum = rewriter.create<daphne::AllAggSumOp>(sumOp.getLoc(), elemMatrix, matrix);
        rewriter.replaceOpWithNewOp<daphne::EwMulOp>(sumOp, sumOp.getResult().getType(), scalar, innerSum);
        return success();
    }
};

/**
 * @brief Rewrites a row-wise aggregate of a single-column matrix to its input.
 *
 * A row-wise aggregate reduces each row to one value, so over an n x 1 matrix it
 * touches exactly one element per row: `sumRow(X) = X`, and likewise for min/max.
 * The rewrite drops the aggregate entirely (and, since these aggregates are Pure,
 * DCE removes it when it was the operand's only user).
 *
 * Only sum/min/max are instantiated: over a singleton each simply returns that
 * element, so the value is unchanged and there is no accumulation to overflow.
 * mean/var/stddev are excluded, since they promote to a floating-point result type
 * (and a singleton variance is 0, not the element), as are idxMin/idxMax,
 * whose result is a position, not the value.
 *
 * The rewrite fails closed: it fires only when the column count is statically
 * known to be 1 and the aggregate preserves the element type (which sum/min/max
 * do via ValueTypeFromFirstArg). It deliberately does not require the full matrix
 * type to match, only the element type: the aggregate result carries no sparsity
 * (it has no SparsityFromArg trait), so replacing it with the input yields an
 * equal-or-more-precise type its users already accept.
 */
template <class AggOp> struct RowAggDim1IdentityPattern : public OpRewritePattern<AggOp> {
    using OpRewritePattern<AggOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AggOp op, PatternRewriter &rewriter) const override {
        Value arg = op.getArg();

        std::optional<ssize_t> cols = daphne::knownNumCols(arg);
        if (!cols || *cols != 1)
            return failure();

        Type elemArg = getMatrixElementTypeOrNull(arg);
        Type elemRes = getMatrixElementTypeOrNull(op.getResult());
        if (!elemArg || elemArg != elemRes)
            return failure();

        rewriter.replaceOp(op, arg);
        return success();
    }
};

/**
 * @brief Rewrites a column-wise aggregate of a single-row matrix to its input.
 *
 * The column-wise counterpart of `RowAggDim1IdentityPattern`: a column-wise
 * aggregate reduces each column to one value, so over a 1 x m matrix it is the
 * identity, `sumCol(X) = X` (and min/max). Guards on a statically-known row count
 * of 1; see `RowAggDim1IdentityPattern` for the sum/min/max-only rationale and
 * the element-type-only match.
 */
template <class AggOp> struct ColAggDim1IdentityPattern : public OpRewritePattern<AggOp> {
    using OpRewritePattern<AggOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AggOp op, PatternRewriter &rewriter) const override {
        Value arg = op.getArg();

        std::optional<ssize_t> rows = daphne::knownNumRows(arg);
        if (!rows || *rows != 1)
            return failure();

        Type elemArg = getMatrixElementTypeOrNull(arg);
        Type elemRes = getMatrixElementTypeOrNull(op.getResult());
        if (!elemArg || elemArg != elemRes)
            return failure();

        rewriter.replaceOp(op, arg);
        return success();
    }
};

// Emits `leaf * coeff` as an element-wise product with a scalar coefficient
// constant of `leaf`'s element type. Pinning the coefficient to the element type
// stops EwMul's CastArgsToResType from promoting the result to a wider type,
// which would change the accumulation width. `leaf` is the matrix (or scalar) to
// scale; the coefficient is placed on the rhs, the order the broadcast kernel and
// EwMul's canonicalizer both leave untouched for a non-scalar lhs. When `leaf` is
// a matrix the result reuses its type with sparsity and symmetry reset, since the
// scaled product's density and symmetry are not the leaf's. For an integer
// element type `coeff` is the bit pattern, which getIntegerAttr truncates to the
// element width, matching the modular arithmetic the integer kernels use at run
// time; for a floating-point element type it is the exact small coefficient.
static Value emitScaledMatrix(PatternRewriter &rewriter, Location loc, Value leaf, uint64_t coeff) {
    Type elemTy = getMatrixElementTypeOrNull(leaf);
    if (!elemTy)
        elemTy = leaf.getType();

    TypedAttr coeffAttr;
    if (auto floatTy = llvm::dyn_cast<FloatType>(elemTy))
        coeffAttr = rewriter.getFloatAttr(floatTy, static_cast<double>(coeff));
    else
        coeffAttr = rewriter.getIntegerAttr(elemTy, static_cast<int64_t>(coeff));
    Value coeffVal = rewriter.create<daphne::ConstantOp>(loc, elemTy, coeffAttr);

    Type resTy = leaf.getType();
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(resTy))
        resTy = mt.withSparsity(-1.0).withSymmetric(daphne::BoolOrUnknown::Unknown);
    return rewriter.create<daphne::EwMulOp>(loc, resTy, leaf, coeffVal);
}

/**
 * @brief Rewrites `a + a` to `a * 2`.
 *
 * A self-addition is an exact doubling: over integers it is `2 * a` in the ring
 * Z/2^n (modular add, so any overflow is preserved identically), and over IEEE
 * floating point it is a pure exponent increment: bit-exact, with NaN, Inf and
 * -0.0 all carried through unchanged. Both element types are therefore admitted;
 * this is the only one of the repeated-add rewrites that fires for floats, since
 * it doubles a single value rather than regrouping distinct addends.
 *
 * String and boolean (i1) element types are excluded: EwMul rejects strings, and
 * a one-bit integer cannot represent the coefficient 2.
 */
struct SelfAddPattern : public OpRewritePattern<daphne::EwAddOp> {
    using OpRewritePattern<daphne::EwAddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::EwAddOp op, PatternRewriter &rewriter) const override {
        if (op.getLhs() != op.getRhs())
            return failure();

        Type elem = getMatrixElementTypeOrNull(op.getLhs());
        if (!elem)
            elem = op.getLhs().getType();
        if (!llvm::isa<FloatType>(elem) && !isRewritableIntElem(elem))
            return failure();

        rewriter.replaceOp(op, emitScaledMatrix(rewriter, op.getLoc(), op.getLhs(), 2));
        return success();
    }
};

/**
 * @brief Rewrites `(x * c) + x` (and the commuted `x + (x * c)`) to `x * (c+1)`.
 *
 * Folds a scaled term and a bare copy of the same value into one scaled term.
 * Restricted to integer element types: over floats `(c*x) + x` and `(c+1)*x`
 * round differently, so the rewrite would not be bit-exact. Over integers it is
 * exact in Z/2^n, so the coefficient sum `c + 1` is computed modulo the element
 * width and never bails on overflow: the wrapped value is the correct result.
 *
 * EwAdd is not commutative in DAPHNE, so both operand orders are matched
 * explicitly. The scaled operand must be an ewMul of the same value `x` by a
 * constant.
 */
struct AddScaledLeafPattern : public OpRewritePattern<daphne::EwAddOp> {
    using OpRewritePattern<daphne::EwAddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::EwAddOp op, PatternRewriter &rewriter) const override {
        // Try both orders: the scaled term may be the lhs or the rhs.
        Value scaled = op.getLhs();
        Value plain = op.getRhs();
        auto mulOp = scaled.getDefiningOp<daphne::EwMulOp>();
        if (!mulOp) {
            scaled = op.getRhs();
            plain = op.getLhs();
            mulOp = scaled.getDefiningOp<daphne::EwMulOp>();
        }
        if (!mulOp)
            return failure();

        // The ewMul must scale `plain` (the bare copy) by a constant factor.
        auto [isConst, factor] = CompilerUtils::isConstant<int64_t>(mulOp.getRhs());
        Value leaf = mulOp.getLhs();
        if (!isConst) {
            auto [isConstLhs, factorLhs] = CompilerUtils::isConstant<int64_t>(mulOp.getLhs());
            isConst = isConstLhs;
            factor = factorLhs;
            leaf = mulOp.getRhs();
        }
        if (!isConst || leaf != plain)
            return failure();

        Type elem = getMatrixElementTypeOrNull(plain);
        if (!elem)
            elem = plain.getType();
        if (!isRewritableIntElem(elem))
            return failure();

        // The scaled term may have promoted the result to a wider type (e.g. an
        // si32 leaf times an si64 constant yields an si64 add). emitScaledMatrix
        // rebuilds from the narrow leaf type and truncates the coefficient to it,
        // so firing here would substitute a wrongly-typed, wrongly-wrapped value.
        // Fire only when the accumulation width is unchanged, as SumScalarFactor
        // does for the same promotion hazard.
        Type resElem = getMatrixElementTypeOrNull(op.getResult());
        if (!resElem)
            resElem = op.getResult().getType();
        if (resElem != elem)
            return failure();

        // Coefficient sum in Z/2^n: compute unsigned so overflow wraps defined,
        // matching the integer kernels' modular add.
        rewriter.replaceOp(op, emitScaledMatrix(rewriter, op.getLoc(), plain, static_cast<uint64_t>(factor) + 1));
        return success();
    }
};

/**
 * @brief Rewrites `(x * c1) + (x * c2)` to `x * (c1+c2)`.
 *
 * Merges two scaled copies of the same value into one. Integer-only for the same
 * reason as `AddScaledLeafPattern`, and the coefficient sum is likewise modular
 * (never bails on overflow).
 */
struct AddTwoScaledPattern : public OpRewritePattern<daphne::EwAddOp> {
    using OpRewritePattern<daphne::EwAddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::EwAddOp op, PatternRewriter &rewriter) const override {
        auto mul1 = op.getLhs().getDefiningOp<daphne::EwMulOp>();
        auto mul2 = op.getRhs().getDefiningOp<daphne::EwMulOp>();
        if (!mul1 || !mul2)
            return failure();

        // Each ewMul must scale a common leaf `x` by a constant factor. The
        // constant may sit on either side of each product.
        auto extract = [](daphne::EwMulOp mul, int64_t &factor, Value &leaf) -> bool {
            auto [isConstR, factorR] = CompilerUtils::isConstant<int64_t>(mul.getRhs());
            if (isConstR) {
                factor = factorR;
                leaf = mul.getLhs();
                return true;
            }
            auto [isConstL, factorL] = CompilerUtils::isConstant<int64_t>(mul.getLhs());
            if (isConstL) {
                factor = factorL;
                leaf = mul.getRhs();
                return true;
            }
            return false;
        };

        int64_t c1, c2;
        Value leaf1, leaf2;
        if (!extract(mul1, c1, leaf1) || !extract(mul2, c2, leaf2))
            return failure();
        if (leaf1 != leaf2)
            return failure();

        Type elem = getMatrixElementTypeOrNull(leaf1);
        if (!elem)
            elem = leaf1.getType();
        if (!isRewritableIntElem(elem))
            return failure();

        // Reject the promoting case for the same reason as AddScaledLeafPattern:
        // a wider result (e.g. an si32 leaf scaled by si64 constants) would be
        // rebuilt at the narrow leaf width, truncating the coefficient and
        // mistyping the replacement. Fire only when the width is unchanged.
        Type resElem = getMatrixElementTypeOrNull(op.getResult());
        if (!resElem)
            resElem = op.getResult().getType();
        if (resElem != elem)
            return failure();

        // Coefficient sum in Z/2^n: unsigned add wraps defined, matching the
        // integer kernels' modular arithmetic.
        rewriter.replaceOp(op, emitScaledMatrix(rewriter, op.getLoc(), leaf1,
                                                static_cast<uint64_t>(c1) + static_cast<uint64_t>(c2)));
        return success();
    }
};

/**
 * @brief Regroups `(M1 + s1) + (M2 + s2)` to `(M1 + M2) + (s1 + s2)`.
 *
 * When two matrix-plus-scalar broadcasts are added, the scalars can be summed
 * once up front instead of broadcast across every element twice. The rewrite
 * pulls the two matrix operands into one matrix add and the two scalars into one
 * scalar add, turning two element-wise scalar broadcasts into one.
 *
 * Integer-only. Over IEEE floating point addition is not associative: each `+`
 * rounds, and regrouping four distinct addends changes which intermediate is
 * rounded first, so the rewrite would not be value-preserving. Over integers it
 * is exact in Z/2^n (the scalar sum may overflow, but the wrapped value is the
 * correct ring element, so it is never a bail condition).
 *
 * All four element types are pinned to one integer type. `hasScaType` alone would
 * admit float, string, index, and boolean scalars, so soundness rests on the
 * explicit `IntegerType` check plus the element-type-equality guards, not on
 * `hasScaType`. Pinning the operand types also stops EwAdd's CastArgsToResType
 * from promoting the two new adds to a wider accumulation type.
 *
 * Terminating: the output is `ewAdd(M1 + M2, s1 + s2)` whose operands are a
 * matrix-plus-matrix and a scalar-plus-scalar, neither of which classifies as a
 * matrix-plus-scalar broadcast, so the pattern cannot match its own result.
 */
struct BroadcastMinimizePattern : public OpRewritePattern<daphne::EwAddOp> {
    using OpRewritePattern<daphne::EwAddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::EwAddOp op, PatternRewriter &rewriter) const override {
        auto add1 = op.getLhs().getDefiningOp<daphne::EwAddOp>();
        auto add2 = op.getRhs().getDefiningOp<daphne::EwAddOp>();
        if (!add1 || !add2)
            return failure();

        // Split an inner add into its matrix and scalar operand, in either order.
        // Fails when the add is not exactly one matrix and one scalar, which is
        // also what keeps the rewrite from matching its own matrix+matrix output.
        auto classify = [](daphne::EwAddOp a, Value &mat, Value &sca) -> bool {
            Value l = a.getLhs(), r = a.getRhs();
            if (getMatrixElementTypeOrNull(l) && CompilerUtils::hasScaType(r)) {
                mat = l;
                sca = r;
                return true;
            }
            if (getMatrixElementTypeOrNull(r) && CompilerUtils::hasScaType(l)) {
                mat = r;
                sca = l;
                return true;
            }
            return false;
        };

        Value m1, s1, m2, s2;
        if (!classify(add1, m1, s1) || !classify(add2, m2, s2))
            return failure();

        // Tie all four element types to one integer type: the two matrices and
        // the two scalars must share it, and it must be the outer result's element
        // type. This makes the regrouping a pure reassociation with no width or
        // sign change, and the integer check rules out non-associative floats and
        // the non-numeric scalars hasScaType would otherwise admit.
        Type elem = getMatrixElementTypeOrNull(m1);
        if (!llvm::isa<IntegerType>(elem))
            return failure();
        if (getMatrixElementTypeOrNull(m2) != elem)
            return failure();
        if (s1.getType() != elem || s2.getType() != elem)
            return failure();
        auto outMat = llvm::dyn_cast<daphne::MatrixType>(op.getResult().getType());
        if (!outMat || outMat.getElementType() != elem)
            return failure();

        // The two matrices must have the same statically-known shape so the matrix
        // add needs no broadcasting; fail closed on any unknown or mismatched
        // extent.
        std::optional<ssize_t> rows1 = daphne::knownNumRows(m1);
        std::optional<ssize_t> cols1 = daphne::knownNumCols(m1);
        std::optional<ssize_t> rows2 = daphne::knownNumRows(m2);
        std::optional<ssize_t> cols2 = daphne::knownNumCols(m2);
        if (!rows1 || !cols1 || !rows2 || !cols2)
            return failure();
        if (*rows1 != *rows2 || *cols1 != *cols2)
            return failure();

        // s1 + s2 as a scalar add; M1 + M2 as a matrix add with sparsity and
        // symmetry reset (the regrouped sum's are not M1's). The outer add reuses
        // the original result type: it is again a matrix-plus-scalar broadcast, so
        // it keeps whatever the original produced.
        Value scalarSum = rewriter.create<daphne::EwAddOp>(op.getLoc(), elem, s1, s2);
        auto m1Ty = llvm::cast<daphne::MatrixType>(m1.getType());
        Type matAddTy = m1Ty.withSparsity(-1.0).withSymmetric(daphne::BoolOrUnknown::Unknown);
        Value matrixSum = rewriter.create<daphne::EwAddOp>(op.getLoc(), matAddTy, m1, m2);
        rewriter.replaceOpWithNewOp<daphne::EwAddOp>(op, op.getResult().getType(), matrixSum, scalarSum);
        return success();
    }
};

} // namespace

namespace mlir::daphne {

void populateLinearAlgebraRewritePatterns(RewritePatternSet &patterns) {
    MLIRContext *ctx = patterns.getContext();
    patterns.add<TraceIdiomPattern, RowScalePattern, ColScalePattern, SumScalarFactorPattern>(ctx);
    patterns.add<RowAggDim1IdentityPattern<RowAggSumOp>, RowAggDim1IdentityPattern<RowAggMinOp>,
                 RowAggDim1IdentityPattern<RowAggMaxOp>, ColAggDim1IdentityPattern<ColAggSumOp>,
                 ColAggDim1IdentityPattern<ColAggMinOp>, ColAggDim1IdentityPattern<ColAggMaxOp>>(ctx);
    patterns.add<SelfAddPattern, AddScaledLeafPattern, AddTwoScaledPattern>(ctx);
    patterns.add<BroadcastMinimizePattern>(ctx);
}

} // namespace mlir::daphne
