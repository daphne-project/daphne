// RUN: daphne-opt --canonicalize %s | FileCheck %s

// Checks that each rewrite still fires after being moved to a different
// canonicalization mechanism. Folders, DRR patterns and C++ patterns all run
// inside --canonicalize, so one pass covers every case here.

// Fold: numRows(X) folds to an index constant when the operand's row extent is
// statically known. The unknown-extent case must survive.
// CHECK-LABEL: func.func @num_rows_known
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 3 : index}>
// CHECK-NEXT: "daphne.return"(%[[C]])
// CHECK-NOT: daphne.numRows
func.func @num_rows_known(%arg0: !daphne.Matrix<3x4xf64>) -> index {
    %0 = "daphne.numRows"(%arg0) : (!daphne.Matrix<3x4xf64>) -> index
    "daphne.return"(%0) : (index) -> ()
}

// Negative case: an unknown row extent leaves numRows in place.
// CHECK-LABEL: func.func @num_rows_unknown
// CHECK: daphne.numRows
func.func @num_rows_unknown(%arg0: !daphne.Matrix<?x4xf64>) -> index {
    %0 = "daphne.numRows"(%arg0) : (!daphne.Matrix<?x4xf64>) -> index
    "daphne.return"(%0) : (index) -> ()
}

// Fold: sparsity(X) folds to an f64 constant when the operand's sparsity is
// statically known. The unknown-sparsity case must survive.
// CHECK-LABEL: func.func @sparsity_known
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 2.500000e-01 : f64}>
// CHECK-NEXT: "daphne.return"(%[[C]])
// CHECK-NOT: daphne.sparsity
func.func @sparsity_known(%arg0: !daphne.Matrix<8x8xf64:sp[2.500000e-01]>) -> f64 {
    %0 = "daphne.sparsity"(%arg0) : (!daphne.Matrix<8x8xf64:sp[2.500000e-01]>) -> f64
    "daphne.return"(%0) : (f64) -> ()
}

// Negative case: an unknown sparsity leaves sparsity in place.
// CHECK-LABEL: func.func @sparsity_unknown
// CHECK: daphne.sparsity
func.func @sparsity_unknown(%arg0: !daphne.Matrix<8x8xf64>) -> f64 {
    %0 = "daphne.sparsity"(%arg0) : (!daphne.Matrix<8x8xf64>) -> f64
    "daphne.return"(%0) : (f64) -> ()
}

// DRR: a rename is a predicate-free pass-through, so a declarative rewrite
// erases it and forwards its operand to the users.
// CHECK-LABEL: func.func @rename_erased
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<3x4xf64>)
// CHECK-NOT: daphne.rename
// CHECK: "daphne.return"(%[[ARG]])
func.func @rename_erased(%arg0: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %0 = "daphne.rename"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%0) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Fold: a trivial cast whose result type equals its operand type is a pure
// pass-through, folded away to the operand. The folder owns this case, so the
// canonicalize method carries no branch for it; --canonicalize still erases the
// cast because the driver folds before it applies canonicalize patterns.
// CHECK-LABEL: func.func @trivial_cast_folded
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<3x4xf64>)
// CHECK-NOT: daphne.cast
// CHECK: "daphne.return"(%[[ARG]])
func.func @trivial_cast_folded(%arg0: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%0) : (!daphne.Matrix<3x4xf64>) -> ()
}

// C++ pattern: normalizing a scalar left-hand side of an ewAdd to the
// right-hand side needs an operand rewrite gated on operand data types, which a
// folder or DRR cannot express. It lives as a hand-written OpRewritePattern.
// After canonicalization the scalar has moved to the second operand position.
// CHECK-LABEL: func.func @ewadd_scalar_lhs_normalized
// CHECK-SAME: (%[[M:.*]]: !daphne.Matrix<3x4xf64>, %[[S:.*]]: f64)
// CHECK: "daphne.ewAdd"(%[[M]], %[[S]])
func.func @ewadd_scalar_lhs_normalized(%m: !daphne.Matrix<3x4xf64>, %s: f64) -> !daphne.Matrix<3x4xf64> {
    %0 = "daphne.ewAdd"(%s, %m) : (f64, !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%0) : (!daphne.Matrix<3x4xf64>) -> ()
}
