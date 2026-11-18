// RUN: daphne-opt --daphne-algebraic-simplify %s | FileCheck %s

// Locks in that ops annotated with an algebraic trait are simplified by
// --daphne-algebraic-simplify. One representative per trait suffices, since the
// trait mechanism is the invariant, not the individual op.

// Involutive: EwMinusOp collapses --a to a.
// CHECK-LABEL: func.func @double_negate
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewMinus
func.func @double_negate(%arg0: f64) -> f64 {
    %0 = "daphne.ewMinus"(%arg0) : (f64) -> f64
    %1 = "daphne.ewMinus"(%0) : (f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// Involutive negative case: a single ewMinus must survive.
// CHECK-LABEL: func.func @single_negate
// CHECK: daphne.ewMinus
func.func @single_negate(%arg0: f64) -> f64 {
    %0 = "daphne.ewMinus"(%arg0) : (f64) -> f64
    "daphne.return"(%0) : (f64) -> ()
}

// IdempotentUnary: EwAbs collapses abs(abs(x)) to a single abs(x). Unlike
// Involutive the inner result is kept, so exactly one ewAbs must survive and be
// returned directly. An integer arg makes a wrong involutive-style rewrite
// (returning the bare %arg0) visible as a missing ewAbs.
// CHECK-LABEL: func.func @double_abs
// CHECK-SAME: (%[[ARG:.*]]: si64) -> si64
// CHECK-NEXT: %[[R:.*]] = "daphne.ewAbs"(%[[ARG]])
// CHECK-NEXT: "daphne.return"(%[[R]])
// CHECK-NOT: daphne.ewAbs
func.func @double_abs(%arg0: si64) -> si64 {
    %0 = "daphne.ewAbs"(%arg0) : (si64) -> si64
    %1 = "daphne.ewAbs"(%0) : (si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// IdempotentUnary negative case: a lone ewAbs must survive.
// CHECK-LABEL: func.func @single_abs
// CHECK: daphne.ewAbs
func.func @single_abs(%arg0: si64) -> si64 {
    %0 = "daphne.ewAbs"(%arg0) : (si64) -> si64
    "daphne.return"(%0) : (si64) -> ()
}

// IdempotentUnary second adopter: sign(sign(x)) collapses to a single sign(x),
// catching a broken attachment on EwSignOp independently of EwAbsOp.
// CHECK-LABEL: func.func @double_sign
// CHECK-SAME: (%[[ARG:.*]]: si64) -> si64
// CHECK-NEXT: %[[R:.*]] = "daphne.ewSign"(%[[ARG]])
// CHECK-NEXT: "daphne.return"(%[[R]])
// CHECK-NOT: daphne.ewSign
func.func @double_sign(%arg0: si64) -> si64 {
    %0 = "daphne.ewSign"(%arg0) : (si64) -> si64
    %1 = "daphne.ewSign"(%0) : (si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// IdentityOnIntegerElementType: floor of a signed integer scalar is a no-op.
// CHECK-LABEL: func.func @floor_int_scalar
// CHECK-SAME: (%[[ARG:.*]]: si64) -> si64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewFloor
func.func @floor_int_scalar(%arg0: si64) -> si64 {
    %0 = "daphne.ewFloor"(%arg0) : (si64) -> si64
    "daphne.return"(%0) : (si64) -> ()
}

// IdentityOnIntegerElementType negative case: floor of f64 must survive.
// CHECK-LABEL: func.func @floor_float_scalar
// CHECK: daphne.ewFloor
func.func @floor_float_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.ewFloor"(%arg0) : (f64) -> f64
    "daphne.return"(%0) : (f64) -> ()
}

// IdentityOnIntegerElementType also fires on matrix operands with integer
// element type, checked here for ceil to exercise a second adopter.
// CHECK-LABEL: func.func @ceil_int_matrix
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewCeil
func.func @ceil_int_matrix(%arg0: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64> {
    %0 = "daphne.ewCeil"(%arg0) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%0) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Round on a float matrix must survive: third adopter, exercised on the
// negative side to confirm the trait's element-type gate is honored.
// CHECK-LABEL: func.func @round_float_matrix
// CHECK: daphne.ewRound
func.func @round_float_matrix(%arg0: !daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64> {
    %0 = "daphne.ewRound"(%arg0) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64>
    "daphne.return"(%0) : (!daphne.Matrix<2x3xf64>) -> ()
}

// NeutralOnZeroRHS on EwAddOp: x + 0 collapses to x for an integer operand.
// Gated to integers: over floats (-0.0) + 0.0 = +0.0, so x + 0.0 -> x would
// wrongly preserve a negative zero.
// CHECK-LABEL: func.func @add_zero_int
// CHECK-SAME: (%[[ARG:.*]]: si64) -> si64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewAdd
func.func @add_zero_int(%arg0: si64) -> si64 {
    %0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %1 = "daphne.ewAdd"(%arg0, %0) : (si64, si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// NeutralOnZeroRHS on EwAddOp negative case: over a float element type the
// rewrite must not fire (signed-zero fidelity). The ewAdd must survive.
// CHECK-LABEL: func.func @add_zero_float
// CHECK: daphne.ewAdd
func.func @add_zero_float(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %1 = "daphne.ewAdd"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnZeroRHS negative case: a non-zero RHS must survive.
// CHECK-LABEL: func.func @add_nonzero_scalar
// CHECK: daphne.ewAdd
func.func @add_nonzero_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %1 = "daphne.ewAdd"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnZeroRHS on EwSubOp: x - 0 collapses to x. This stays valid over
// floats for a positive zero RHS (x - 0.0 = x is exact for every IEEE value),
// so unlike ewAdd it is not gated to integers. Exercising the second adopter
// also catches a broken trait attachment on EwSubOp independently.
// CHECK-LABEL: func.func @sub_zero_scalar
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewSub
func.func @sub_zero_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %1 = "daphne.ewSub"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnZeroRHS negative case: a negative zero RHS must NOT be dropped.
// x - (-0.0) = x + 0.0, which flips a -0.0 argument to +0.0, so the identity
// fails for negative zero and the ewSub must survive.
// CHECK-LABEL: func.func @sub_neg_zero_scalar
// CHECK: daphne.ewSub
func.func @sub_neg_zero_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = -0.0 : f64} : () -> f64
    %1 = "daphne.ewSub"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnOneRHS on EwMulOp: x * 1 collapses to x.
// CHECK-LABEL: func.func @mul_one_scalar
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewMul
func.func @mul_one_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %1 = "daphne.ewMul"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnOneRHS on EwDivOp: x / 1 collapses to x. Exercises the second
// adopter of NeutralOnOneRHS so a broken attachment on EwDivOp is caught
// independently of EwMulOp.
// CHECK-LABEL: func.func @div_one_scalar
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewDiv
func.func @div_one_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %1 = "daphne.ewDiv"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnOneRHS negative case: x / 2 must survive.
// CHECK-LABEL: func.func @div_two_scalar
// CHECK: daphne.ewDiv
func.func @div_two_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 2.0 : f64} : () -> f64
    %1 = "daphne.ewDiv"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// RightAbsorbingOnZero on EwMulOp: x * 0 collapses to 0, but only for integer
// element types. Over IEEE floating point the identity is false (NaN * 0 = NaN,
// Inf * 0 = NaN), so the rewrite is gated on an integer element type.
// CHECK-LABEL: func.func @mul_zero_rhs_int
// CHECK: %[[Z:.*]] = "daphne.constant"() <{value = 0 : si64}>
// CHECK-NEXT: "daphne.return"(%[[Z]])
// CHECK-NOT: daphne.ewMul
func.func @mul_zero_rhs_int(%arg0: si64) -> si64 {
    %0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %1 = "daphne.ewMul"(%arg0, %0) : (si64, si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// LeftAbsorbingOnZero on EwMulOp: 0 * x collapses to 0 for an integer operand.
// CHECK-LABEL: func.func @mul_zero_lhs_int
// CHECK: %[[Z:.*]] = "daphne.constant"() <{value = 0 : si64}>
// CHECK-NEXT: "daphne.return"(%[[Z]])
// CHECK-NOT: daphne.ewMul
func.func @mul_zero_lhs_int(%arg0: si64) -> si64 {
    %0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %1 = "daphne.ewMul"(%0, %arg0) : (si64, si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// Absorbing-zero negative case: over a float element type the rewrite must not
// fire, because NaN * 0.0 = NaN and Inf * 0.0 = NaN. The ewMul must survive.
// CHECK-LABEL: func.func @mul_zero_float
// CHECK: daphne.ewMul
func.func @mul_zero_float(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %1 = "daphne.ewMul"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// RightAbsorbingOnZero on EwAndOp: x AND 0 collapses to 0. Sound over the whole
// integer domain because logical AND is truthiness-based (x && 0 = 0 for every
// integer x); ewAnd is integer-typed by construction, so no float caveat
// applies. A second adopter of the absorbing trait, exercised once to catch a
// broken attachment independently of EwMulOp.
// CHECK-LABEL: func.func @and_zero_rhs_int
// CHECK: %[[Z:.*]] = "daphne.constant"() <{value = 0 : si64}>
// CHECK-NEXT: "daphne.return"(%[[Z]])
// CHECK-NOT: daphne.ewAnd
func.func @and_zero_rhs_int(%arg0: si64) -> si64 {
    %0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %1 = "daphne.ewAnd"(%arg0, %0) : (si64, si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// Absorbing negative case: a scalar zero AND a matrix must NOT collapse. The
// result is an MxN matrix of zeros, not the scalar constant, so the
// type-equality guard declines. The only matrix-typed absorbing case in the
// suite, covering the guard that keeps a broadcasting rewrite from shrinking
// the result to a scalar.
// CHECK-LABEL: func.func @and_zero_rhs_matrix
// CHECK: daphne.ewAnd
func.func @and_zero_rhs_matrix(%arg0: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64> {
    %0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %1 = "daphne.ewAnd"(%arg0, %0) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%1) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Involutive on TransposeOp: t(t(X)) collapses to X. The element type is
// unchanged, but the shape swaps back, so this also confirms the pattern's
// type-equality guard admits a shape-swapping involution on a rectangular
// matrix (the inner input type equals the outer result type).
// CHECK-LABEL: func.func @transpose_double_rectangular
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.transpose
func.func @transpose_double_rectangular(%arg0: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %0 = "daphne.transpose"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<4x3xf64>
    %1 = "daphne.transpose"(%0) : (!daphne.Matrix<4x3xf64>) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%1) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Involutive negative case: a single transpose must survive.
// CHECK-LABEL: func.func @transpose_single
// CHECK: daphne.transpose
func.func @transpose_single(%arg0: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<4x3xf64> {
    %0 = "daphne.transpose"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<4x3xf64>
    "daphne.return"(%0) : (!daphne.Matrix<4x3xf64>) -> ()
}

// Involutive on ReverseOp: reverse(reverse(X)) collapses to X. Exercises a
// second Involutive adopter so a broken attachment on ReverseOp is caught
// independently of TransposeOp.
// CHECK-LABEL: func.func @reverse_double
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.reverse
func.func @reverse_double(%arg0: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %0 = "daphne.reverse"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    %1 = "daphne.reverse"(%0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%1) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Involutive negative case: a single reverse must survive.
// CHECK-LABEL: func.func @reverse_single
// CHECK: daphne.reverse
func.func @reverse_single(%arg0: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %0 = "daphne.reverse"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%0) : (!daphne.Matrix<3x4xf64>) -> ()
}

// OrderAgnosticAggregate over OnlyReordersElements: sum(t(X)) folds to sum(X),
// dropping the transpose the total sum never needed. Uses a dynamic shape to
// show the fold does not depend on statically-known extents.
// CHECK-LABEL: func.func @agg_reorder_transpose
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<?x?xf64>)
// CHECK: "daphne.sumAll"(%[[ARG]])
// CHECK-NOT: daphne.transpose
func.func @agg_reorder_transpose(%arg0: !daphne.Matrix<?x?xf64>) -> f64 {
    %0 = "daphne.transpose"(%arg0) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %1 = "daphne.sumAll"(%0) : (!daphne.Matrix<?x?xf64>) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// Second OnlyReordersElements producer: sum(reverse(X)) folds to sum(X) too,
// catching a broken trait attachment on ReverseOp independently of transpose.
// CHECK-LABEL: func.func @agg_reorder_reverse
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<3x4xf64>)
// CHECK: "daphne.sumAll"(%[[ARG]])
// CHECK-NOT: daphne.reverse
func.func @agg_reorder_reverse(%arg0: !daphne.Matrix<3x4xf64>) -> f64 {
    %0 = "daphne.reverse"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    %1 = "daphne.sumAll"(%0) : (!daphne.Matrix<3x4xf64>) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// A chain of reorders peels off entirely: sum(t(reverse(X))) folds to sum(X)
// as the greedy driver re-examines each new sumAll.
// CHECK-LABEL: func.func @agg_reorder_chain
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<3x4xf64>)
// CHECK: "daphne.sumAll"(%[[ARG]])
// CHECK-NOT: daphne.transpose
// CHECK-NOT: daphne.reverse
func.func @agg_reorder_chain(%arg0: !daphne.Matrix<3x4xf64>) -> f64 {
    %0 = "daphne.reverse"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    %1 = "daphne.transpose"(%0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<4x3xf64>
    %2 = "daphne.sumAll"(%1) : (!daphne.Matrix<4x3xf64>) -> f64
    "daphne.return"(%2) : (f64) -> ()
}

// meanAll over a transpose folds to meanAll on the original. Locks in the trait
// token on meanAll and that the generic clone rebuilds a meanAll (not a sumAll):
// the CHECK for "daphne.meanAll" fails if a regressed clone changed the op class.
// CHECK-LABEL: func.func @agg_reorder_mean_transpose
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<?x?xf64>)
// CHECK: "daphne.meanAll"(%[[ARG]])
// CHECK-NOT: daphne.transpose
func.func @agg_reorder_mean_transpose(%arg0: !daphne.Matrix<?x?xf64>) -> f64 {
    %0 = "daphne.transpose"(%arg0) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %1 = "daphne.meanAll"(%0) : (!daphne.Matrix<?x?xf64>) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// Third adopter and a second reorder producer: stddevAll over a reverse folds,
// catching a broken trait attachment on stddevAll independently of the others.
// CHECK-LABEL: func.func @agg_reorder_stddev_reverse
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<3x4xf64>)
// CHECK: "daphne.stddevAll"(%[[ARG]])
// CHECK-NOT: daphne.reverse
func.func @agg_reorder_stddev_reverse(%arg0: !daphne.Matrix<3x4xf64>) -> f64 {
    %0 = "daphne.reverse"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    %1 = "daphne.stddevAll"(%0) : (!daphne.Matrix<3x4xf64>) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// Negative case: single-axis reductions are not order-agnostic. meanRow(t(X))
// equals meanCol(X), not meanRow(X), so the transpose must survive. Only the
// absent trait token keeps meanRow untouched here; the old sum-specialised
// pattern made the fold structurally impossible.
// CHECK-LABEL: func.func @agg_reorder_meanrow_not_agnostic
// CHECK: daphne.transpose
func.func @agg_reorder_meanrow_not_agnostic(%arg0: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<4x1xf64> {
    %0 = "daphne.transpose"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<4x3xf64>
    %1 = "daphne.meanRow"(%0) : (!daphne.Matrix<4x3xf64>) -> !daphne.Matrix<4x1xf64>
    "daphne.return"(%1) : (!daphne.Matrix<4x1xf64>) -> ()
}

// Negative case: minAll is deliberately left untagged. Its value is
// permutation-invariant too, but min/max are intentionally out of scope, so the
// trait gate never fires and the reorder survives.
// CHECK-LABEL: func.func @agg_reorder_min_not_agnostic
// CHECK: daphne.transpose
func.func @agg_reorder_min_not_agnostic(%arg0: !daphne.Matrix<3x4xf64>) -> f64 {
    %0 = "daphne.transpose"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<4x3xf64>
    %1 = "daphne.minAll"(%0) : (!daphne.Matrix<4x3xf64>) -> f64
    "daphne.return"(%1) : (f64) -> ()
}
