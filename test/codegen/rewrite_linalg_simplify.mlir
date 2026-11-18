// RUN: daphne-opt --daphne-algebraic-simplify %s | FileCheck %s

// Locks in the linear-algebra simplification rewrites driven by
// --daphne-algebraic-simplify.

// Trace idiom: sum(diagVector(X @ Y)) with X @ Y square is rewritten to
// sum(X * t(Y)): the matMul and diagVector are gone, replaced by a transpose
// and an ewMul feeding the sumAll.
// CHECK-LABEL: func.func @trace_idiom
// CHECK-NOT: daphne.matMul
// CHECK-NOT: daphne.diagVector
// CHECK: daphne.transpose
// CHECK: daphne.ewMul
// CHECK: daphne.sumAll
func.func @trace_idiom(%x: !daphne.Matrix<3x4xf64>, %y: !daphne.Matrix<4x3xf64>) -> f64 {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %false) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x3xf64>, i1, i1) -> !daphne.Matrix<3x3xf64>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<3x3xf64>) -> !daphne.Matrix<3x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<3x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Negative case: unknown shape. Without static dimensions the rewrite cannot
// prove X @ Y is square, so it fails closed and the matMul survives.
// CHECK-LABEL: func.func @trace_idiom_unknown_shape
// CHECK: daphne.matMul
// CHECK: daphne.diagVector
func.func @trace_idiom_unknown_shape(%x: !daphne.Matrix<?x?xf64>, %y: !daphne.Matrix<?x?xf64>) -> f64 {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %false) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<?x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Negative case: a transposed matMul flag is a different chain the rewrite does
// not handle, so the matMul survives.
// CHECK-LABEL: func.func @trace_idiom_transposed
// CHECK: daphne.matMul
// CHECK: daphne.diagVector
func.func @trace_idiom_transposed(%x: !daphne.Matrix<3x4xf64>, %y: !daphne.Matrix<3x4xf64>) -> f64 {
    %true = "daphne.constant"() <{value = true}> : () -> i1
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %true) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<3x4xf64>, i1, i1) -> !daphne.Matrix<3x3xf64>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<3x3xf64>) -> !daphne.Matrix<3x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<3x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Negative case: X and Y have different element types, so the emitted ewMul
// would be ill-typed. The rewrite fails closed and the matMul survives.
// CHECK-LABEL: func.func @trace_idiom_elem_mismatch
// CHECK: daphne.matMul
// CHECK: daphne.diagVector
func.func @trace_idiom_elem_mismatch(%x: !daphne.Matrix<3x4xf64>, %y: !daphne.Matrix<4x3xsi64>) -> f64 {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %false) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x3xsi64>, i1, i1) -> !daphne.Matrix<3x3xf64>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<3x3xf64>) -> !daphne.Matrix<3x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<3x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Row scaling: diag(v) @ X with v an n x 1 column vector and X n x m is
// rewritten to X * v: the matMul and the materialized n x n diagonal are
// gone, replaced by a single ewMul broadcasting v down the rows of X. The
// operand order is X (matrix) first, v (vector) second.
// CHECK-LABEL: func.func @row_scale
// CHECK-NOT: daphne.matMul
// CHECK-NOT: daphne.diagMatrix
// CHECK: daphne.ewMul
func.func @row_scale(%v: !daphne.Matrix<3x1xf64>, %x: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64>
    %p = "daphne.matMul"(%d, %x, %false, %false) : (!daphne.Matrix<3x3xf64>, !daphne.Matrix<3x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Negative case: unknown shape. Without a statically known row extent the
// rewrite cannot prove the broadcast branch will fire, so it fails closed and
// the matMul survives.
// CHECK-LABEL: func.func @row_scale_unknown_shape
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @row_scale_unknown_shape(%v: !daphne.Matrix<?x1xf64>, %x: !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64>
    %p = "daphne.matMul"(%d, %x, %false, %false) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    "daphne.return"(%p) : (!daphne.Matrix<?x?xf64>) -> ()
}

// Negative case: a transposed matMul flag is a different chain the rewrite does
// not handle, so the matMul survives.
// CHECK-LABEL: func.func @row_scale_transposed
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @row_scale_transposed(%v: !daphne.Matrix<3x1xf64>, %x: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %true = "daphne.constant"() <{value = true}> : () -> i1
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64>
    %p = "daphne.matMul"(%d, %x, %true, %false) : (!daphne.Matrix<3x3xf64>, !daphne.Matrix<3x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Negative case: v and X have different element types, so the emitted ewMul
// would be ill-typed. The rewrite fails closed and the matMul survives.
// CHECK-LABEL: func.func @row_scale_elem_mismatch
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @row_scale_elem_mismatch(%v: !daphne.Matrix<3x1xf64>, %x: !daphne.Matrix<3x4xsi64>) -> !daphne.Matrix<3x4xsi64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64>
    %p = "daphne.matMul"(%d, %x, %false, %false) : (!daphne.Matrix<3x3xf64>, !daphne.Matrix<3x4xsi64>, i1, i1) -> !daphne.Matrix<3x4xsi64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xsi64>) -> ()
}

// Column scaling: X @ diag(v) with X n x m and v an m x 1 column vector is
// rewritten to X * t(v): the matMul and the materialized m x m diagonal are
// gone, replaced by a transpose of v to a 1 x m row vector and a single ewMul
// broadcasting it across the columns of X. The operand order is X (matrix)
// first, t(v) (row vector) second.
// CHECK-LABEL: func.func @col_scale
// CHECK-NOT: daphne.matMul
// CHECK-NOT: daphne.diagMatrix
// CHECK: daphne.transpose
// CHECK: daphne.ewMul
func.func @col_scale(%x: !daphne.Matrix<3x4xf64>, %v: !daphne.Matrix<4x1xf64>) -> !daphne.Matrix<3x4xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<4x1xf64>) -> !daphne.Matrix<4x4xf64>
    %p = "daphne.matMul"(%x, %d, %false, %false) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Negative case: unknown shape. Without a statically known column extent the
// rewrite cannot prove the broadcast branch will fire, so it fails closed and
// the matMul survives.
// CHECK-LABEL: func.func @col_scale_unknown_shape
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @col_scale_unknown_shape(%x: !daphne.Matrix<?x?xf64>, %v: !daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64>
    %p = "daphne.matMul"(%x, %d, %false, %false) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    "daphne.return"(%p) : (!daphne.Matrix<?x?xf64>) -> ()
}

// Negative case: a transposed matMul flag is a different chain the rewrite does
// not handle, so the matMul survives.
// CHECK-LABEL: func.func @col_scale_transposed
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @col_scale_transposed(%x: !daphne.Matrix<3x4xf64>, %v: !daphne.Matrix<4x1xf64>) -> !daphne.Matrix<3x4xf64> {
    %true = "daphne.constant"() <{value = true}> : () -> i1
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<4x1xf64>) -> !daphne.Matrix<4x4xf64>
    %p = "daphne.matMul"(%x, %d, %false, %true) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Negative case: X and v have different element types, so the emitted ewMul
// would be ill-typed. The rewrite fails closed and the matMul survives.
// CHECK-LABEL: func.func @col_scale_elem_mismatch
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @col_scale_elem_mismatch(%x: !daphne.Matrix<3x4xf64>, %v: !daphne.Matrix<4x1xsi64>) -> !daphne.Matrix<3x4xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<4x1xsi64>) -> !daphne.Matrix<4x4xsi64>
    %p = "daphne.matMul"(%x, %d, %false, %false) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x4xsi64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}

// The emitted ewMul must NOT inherit the sparsity/symmetry inferred for the ops
// it replaces: those properties describe a matrix product (or, for the trace
// idiom, the left factor), not the element-wise result, so the rewrites reset
// them to unknown for a later inference pass to re-derive. Inference does not
// re-run on the default pipeline after this pass, so a leaked value would
// persist, so these cases lock the reset in.

// Row scaling: the matMul result carries a concrete sparsity, and the emitted
// ewMul must drop it (result type is a plain 3x4xf64, no sp[...] annotation).
// CHECK-LABEL: func.func @row_scale_resets_sparsity
// CHECK: %[[M:.*]] = "daphne.ewMul"
// CHECK-SAME: -> !daphne.Matrix<3x4xf64>
func.func @row_scale_resets_sparsity(%v: !daphne.Matrix<3x1xf64>, %x: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64:sp[5.000000e-01]> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64>
    %p = "daphne.matMul"(%d, %x, %false, %false) : (!daphne.Matrix<3x3xf64>, !daphne.Matrix<3x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64:sp[5.000000e-01]>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64:sp[5.000000e-01]>) -> ()
}

// Column scaling: same invariant, the concrete matMul sparsity must not reach
// the ewMul result.
// CHECK-LABEL: func.func @col_scale_resets_sparsity
// CHECK: %[[M:.*]] = "daphne.ewMul"
// CHECK-SAME: -> !daphne.Matrix<3x4xf64>
func.func @col_scale_resets_sparsity(%x: !daphne.Matrix<3x4xf64>, %v: !daphne.Matrix<4x1xf64>) -> !daphne.Matrix<3x4xf64:sp[2.500000e-01]> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<4x1xf64>) -> !daphne.Matrix<4x4xf64>
    %p = "daphne.matMul"(%x, %d, %false, %false) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64:sp[2.500000e-01]>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64:sp[2.500000e-01]>) -> ()
}

// Trace idiom: the left factor X carries a concrete sparsity; the emitted ewMul
// result must reset it (X remains its own type as an operand, only the product's
// result type is reset). Also exercises a symmetric matMul result being dropped.
// CHECK-LABEL: func.func @trace_idiom_resets_props
// CHECK: %[[M:.*]] = "daphne.ewMul"
// CHECK-SAME: -> !daphne.Matrix<3x3xf64>
func.func @trace_idiom_resets_props(%x: !daphne.Matrix<3x3xf64:sp[3.000000e-01]>, %y: !daphne.Matrix<3x3xf64>) -> f64 {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %false) : (!daphne.Matrix<3x3xf64:sp[3.000000e-01]>, !daphne.Matrix<3x3xf64>, i1, i1) -> !daphne.Matrix<3x3xf64:symmetric[true]>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<3x3xf64:symmetric[true]>) -> !daphne.Matrix<3x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<3x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Scalar-factor hoisting: sum(s * X) is rewritten to s * sum(X), the scalar
// multiply moves out of the element-wise product to after the aggregate. The
// aggregate now runs directly on the matrix X, and the surviving ewMul is a
// scalar-by-scalar product (si64, si64) -> si64. The rewrite fires only over
// integers, where regrouping the factor is exact; here s and X share si64.
// CHECK-LABEL: func.func @sum_scalar_factor
// CHECK: %[[S:.*]] = "daphne.sumAll"(%{{.*}}) : (!daphne.Matrix<3x4xsi64>) -> si64
// CHECK: "daphne.ewMul"(%{{.*}}, %[[S]]) : (si64, si64) -> si64
func.func @sum_scalar_factor(%s: si64, %x: !daphne.Matrix<3x4xsi64>) -> si64 {
    %m = "daphne.ewMul"(%s, %x) : (si64, !daphne.Matrix<3x4xsi64>) -> !daphne.Matrix<3x4xsi64>
    %r = "daphne.sumAll"(%m) : (!daphne.Matrix<3x4xsi64>) -> si64
    "daphne.return"(%r) : (si64) -> ()
}

// Negative case: a float matrix. Regrouping the factor rounds differently over
// floating point, so the rewrite fails closed and the ewMul over the matrix
// survives.
// CHECK-LABEL: func.func @sum_scalar_factor_float
// CHECK: daphne.ewMul
// CHECK: daphne.sumAll
func.func @sum_scalar_factor_float(%s: f64, %x: !daphne.Matrix<3x4xf64>) -> f64 {
    %m = "daphne.ewMul"(%s, %x) : (f64, !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    %r = "daphne.sumAll"(%m) : (!daphne.Matrix<3x4xf64>) -> f64
    "daphne.return"(%r) : (f64) -> ()
}

// Negative case: the scalar promotes the matrix element type (s : f64 over an
// si64 matrix, so the product is f64). Moving the sum inside would accumulate in
// si64, a different accumulation type that can overflow, so the rewrite fails
// closed and the element-wise product over the matrix survives.
// CHECK-LABEL: func.func @sum_scalar_factor_promoting
// CHECK: daphne.ewMul
// CHECK: daphne.sumAll
func.func @sum_scalar_factor_promoting(%s: f64, %x: !daphne.Matrix<3x4xsi64>) -> f64 {
    %m = "daphne.ewMul"(%s, %x) : (f64, !daphne.Matrix<3x4xsi64>) -> !daphne.Matrix<3x4xf64>
    %r = "daphne.sumAll"(%m) : (!daphne.Matrix<3x4xf64>) -> f64
    "daphne.return"(%r) : (f64) -> ()
}

// Same rewrite with the scalar on the ewMul rhs (X * s), the form EwMul's
// canonicalizer produces, so the shape the pass most often sees. The match has a
// second arm for this operand order; the emitted ewMul(s, sum(X)) is identical to
// the scalar-lhs case, so only the input order distinguishes the two.
// CHECK-LABEL: func.func @sum_scalar_factor_scalar_rhs
// CHECK: %[[S:.*]] = "daphne.sumAll"(%{{.*}}) : (!daphne.Matrix<3x4xsi64>) -> si64
// CHECK: "daphne.ewMul"(%{{.*}}, %[[S]]) : (si64, si64) -> si64
func.func @sum_scalar_factor_scalar_rhs(%s: si64, %x: !daphne.Matrix<3x4xsi64>) -> si64 {
    %m = "daphne.ewMul"(%x, %s) : (!daphne.Matrix<3x4xsi64>, si64) -> !daphne.Matrix<3x4xsi64>
    %r = "daphne.sumAll"(%m) : (!daphne.Matrix<3x4xsi64>) -> si64
    "daphne.return"(%r) : (si64) -> ()
}

// Row-aggregate identity: a row-wise sum of an n x 1 matrix touches one element
// per row, so sumRow(X) = X and the aggregate is dropped.
// CHECK-LABEL: func.func @row_agg_dim1
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<3x1xf64>)
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.sumRow
func.func @row_agg_dim1(%x: !daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x1xf64> {
    %r = "daphne.sumRow"(%x) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x1xf64>
    "daphne.return"(%r) : (!daphne.Matrix<3x1xf64>) -> ()
}

// Column-aggregate identity: a column-wise sum of a 1 x m matrix is the
// identity, sumCol(X) = X.
// CHECK-LABEL: func.func @col_agg_dim1
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<1x4xf64>)
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.sumCol
func.func @col_agg_dim1(%x: !daphne.Matrix<1x4xf64>) -> !daphne.Matrix<1x4xf64> {
    %r = "daphne.sumCol"(%x) : (!daphne.Matrix<1x4xf64>) -> !daphne.Matrix<1x4xf64>
    "daphne.return"(%r) : (!daphne.Matrix<1x4xf64>) -> ()
}

// Min/max adopt the same identity over a singleton reduced dimension. Exercised
// on an integer minRow so a broken instantiation of the min/max variant is
// caught independently of the sum variant.
// CHECK-LABEL: func.func @row_agg_dim1_min
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<2x1xsi64>)
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.minRow
func.func @row_agg_dim1_min(%x: !daphne.Matrix<2x1xsi64>) -> !daphne.Matrix<2x1xsi64> {
    %r = "daphne.minRow"(%x) : (!daphne.Matrix<2x1xsi64>) -> !daphne.Matrix<2x1xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x1xsi64>) -> ()
}

// Negative case: a row-wise sum of a multi-column matrix genuinely reduces, so
// the aggregate must survive.
// CHECK-LABEL: func.func @row_agg_multicol
// CHECK: daphne.sumRow
func.func @row_agg_multicol(%x: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x1xf64> {
    %r = "daphne.sumRow"(%x) : (!daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x1xf64>
    "daphne.return"(%r) : (!daphne.Matrix<3x1xf64>) -> ()
}

// Negative case: with the column count unknown the rewrite fails closed (it
// cannot prove the reduced dimension is 1), so the aggregate survives.
// CHECK-LABEL: func.func @row_agg_unknown
// CHECK: daphne.sumRow
func.func @row_agg_unknown(%x: !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x1xf64> {
    %r = "daphne.sumRow"(%x) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x1xf64>
    "daphne.return"(%r) : (!daphne.Matrix<?x1xf64>) -> ()
}

// Negative case: meanRow promotes to a floating-point result and a singleton
// mean is not the element in general, so the FP-promoting variants are not
// instantiated and the aggregate survives even over an n x 1 matrix.
// CHECK-LABEL: func.func @row_agg_mean_promote
// CHECK: daphne.meanRow
func.func @row_agg_mean_promote(%x: !daphne.Matrix<3x1xsi64>) -> !daphne.Matrix<3x1xf64> {
    %r = "daphne.meanRow"(%x) : (!daphne.Matrix<3x1xsi64>) -> !daphne.Matrix<3x1xf64>
    "daphne.return"(%r) : (!daphne.Matrix<3x1xf64>) -> ()
}

// Repeated-add self-doubling: X + X is rewritten to X * 2. Fires for floats too:
// doubling a single value is an exact exponent increment, unlike regrouping
// distinct addends, so this exercises the float case.
// CHECK-LABEL: func.func @repeated_add_self_float
// CHECK-NOT: daphne.ewAdd
// CHECK: daphne.ewMul
func.func @repeated_add_self_float(%x: !daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64> {
    %r = "daphne.ewAdd"(%x, %x) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xf64>) -> ()
}

// A chain X + X + X collapses to X * 3: the greedy driver rewrites the inner
// X + X to X * 2, re-examines the result, then folds (X * 2) + X to X * 3. The
// two ewAdds are gone, leaving one ewMul by the coefficient 3.
// CHECK-LABEL: func.func @repeated_add_chain_int
// CHECK-NOT: daphne.ewAdd
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 3 : si64}>
// CHECK: "daphne.ewMul"(%{{.*}}, %[[C]])
func.func @repeated_add_chain_int(%x: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64> {
    %a = "daphne.ewAdd"(%x, %x) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    %b = "daphne.ewAdd"(%a, %x) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%b) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Two scaled copies merge: (X * 2) + (X * 5) is rewritten to X * 7.
// CHECK-LABEL: func.func @add_two_scaled_int
// CHECK-NOT: daphne.ewAdd
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 7 : si64}>
// CHECK: "daphne.ewMul"(%{{.*}}, %[[C]])
func.func @add_two_scaled_int(%x: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64> {
    %c2 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %c5 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %m1 = "daphne.ewMul"(%x, %c2) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %m2 = "daphne.ewMul"(%x, %c5) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%m1, %m2) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// EwMul is commutative, so the scaling constant may sit on either side of the
// inner product. Here the first product is written constant-first (2 * X) and
// the second matrix-first (X * 5); both are extracted and merged to X * 7.
// CHECK-LABEL: func.func @add_two_scaled_const_lhs
// CHECK-NOT: daphne.ewAdd
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 7 : si64}>
// CHECK: "daphne.ewMul"(%{{.*}}, %[[C]])
func.func @add_two_scaled_const_lhs(%x: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64> {
    %c2 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %c5 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %m1 = "daphne.ewMul"(%c2, %x) : (si64, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    %m2 = "daphne.ewMul"(%x, %c5) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%m1, %m2) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// EwAdd is not commutative in DAPHNE, so the scaled-leaf fold must match the
// scaled term on either side. Here the bare copy is the lhs: X + (X * 4) folds
// to X * 5.
// CHECK-LABEL: func.func @add_scaled_leaf_commuted_int
// CHECK-NOT: daphne.ewAdd
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 5 : si64}>
// CHECK: "daphne.ewMul"(%{{.*}}, %[[C]])
func.func @add_scaled_leaf_commuted_int(%x: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64> {
    %c4 = "daphne.constant"() {value = 4 : si64} : () -> si64
    %m = "daphne.ewMul"(%x, %c4) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%x, %m) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Negative case: the scaled-leaf fold is integer-only, because over floats
// (c*x) + x and (c+1)*x round differently. With a float element type the ewAdd
// must survive.
// CHECK-LABEL: func.func @add_scaled_leaf_float
// CHECK: daphne.ewAdd
func.func @add_scaled_leaf_float(%x: !daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64> {
    %c4 = "daphne.constant"() {value = 4.0 : f64} : () -> f64
    %m = "daphne.ewMul"(%x, %c4) : (!daphne.Matrix<2x3xf64>, f64) -> !daphne.Matrix<2x3xf64>
    %r = "daphne.ewAdd"(%m, %x) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xf64>) -> ()
}

// Negative case: self-doubling emits an ewMul, which rejects strings, so a
// string element type must not be rewritten. The ewAdd (string concatenation)
// must survive.
// CHECK-LABEL: func.func @self_add_string
// CHECK: daphne.ewAdd
func.func @self_add_string(%x: !daphne.Matrix<2x3x!daphne.String>) -> !daphne.Matrix<2x3x!daphne.String> {
    %r = "daphne.ewAdd"(%x, %x) : (!daphne.Matrix<2x3x!daphne.String>, !daphne.Matrix<2x3x!daphne.String>) -> !daphne.Matrix<2x3x!daphne.String>
    "daphne.return"(%r) : (!daphne.Matrix<2x3x!daphne.String>) -> ()
}

// Negative case: distinct operands are not a self-addition, so X + Y survives.
// CHECK-LABEL: func.func @add_distinct_leaves
// CHECK: daphne.ewAdd
func.func @add_distinct_leaves(%x: !daphne.Matrix<2x3xsi64>, %y: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64> {
    %r = "daphne.ewAdd"(%x, %y) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Negative case: the scaled-leaf fold needs a constant factor. Here the factor
// is a runtime value, so it cannot be folded and the ewAdd survives.
// CHECK-LABEL: func.func @add_scaled_leaf_nonconst
// CHECK: daphne.ewAdd
func.func @add_scaled_leaf_nonconst(%x: !daphne.Matrix<2x3xsi64>, %c: si64) -> !daphne.Matrix<2x3xsi64> {
    %m = "daphne.ewMul"(%x, %c) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%m, %x) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Negative case: the scaled term promotes the element width (an si32 leaf times
// an si64 constant infers an si64 product and add). The coefficient fold rebuilds
// from the narrow leaf type, which would truncate the coefficient and mistype the
// result, so it must fail closed and the ewAdd survive.
// CHECK-LABEL: func.func @add_scaled_leaf_promoting
// CHECK: daphne.ewAdd
func.func @add_scaled_leaf_promoting(%x: !daphne.Matrix<2x3xsi32>) -> !daphne.Matrix<2x3xsi64> {
    %c4 = "daphne.constant"() {value = 4 : si64} : () -> si64
    %m = "daphne.ewMul"(%x, %c4) : (!daphne.Matrix<2x3xsi32>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%m, %x) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi32>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Negative case: the two-scaled fold has the same width-promotion guard. Two
// si32-leaf products scaled by si64 constants infer an si64 add, so the fold must
// fail closed rather than rebuild at the narrow width.
// CHECK-LABEL: func.func @add_two_scaled_promoting
// CHECK: daphne.ewAdd
func.func @add_two_scaled_promoting(%x: !daphne.Matrix<2x3xsi32>) -> !daphne.Matrix<2x3xsi64> {
    %c2 = "daphne.constant"() {value = 2 : si64} : () -> si64
    %c5 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %m1 = "daphne.ewMul"(%x, %c2) : (!daphne.Matrix<2x3xsi32>, si64) -> !daphne.Matrix<2x3xsi64>
    %m2 = "daphne.ewMul"(%x, %c5) : (!daphne.Matrix<2x3xsi32>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%m1, %m2) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// The emitted ewMul must reset the leaf's sparsity: the scaled product's density
// is not the leaf's, and inference does not re-run after this pass, so a leaked
// value would persist into representation selection. The result type must be a
// plain 2x3xf64 with no sp[...] annotation.
// CHECK-LABEL: func.func @self_add_resets_sparsity
// CHECK: %[[M:.*]] = "daphne.ewMul"
// CHECK-SAME: -> !daphne.Matrix<2x3xf64>
func.func @self_add_resets_sparsity(%x: !daphne.Matrix<2x3xf64:sp[3.000000e-01]>)
        -> !daphne.Matrix<2x3xf64:sp[3.000000e-01]> {
    %r = "daphne.ewAdd"(%x, %x)
        : (!daphne.Matrix<2x3xf64:sp[3.000000e-01]>, !daphne.Matrix<2x3xf64:sp[3.000000e-01]>)
        -> !daphne.Matrix<2x3xf64:sp[3.000000e-01]>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xf64:sp[3.000000e-01]>) -> ()
}

// Broadcast minimization: (M1 + s1) + (M2 + s2) is regrouped to
// (M1 + M2) + (s1 + s2), summing the scalars once instead of broadcasting twice.
// The result is one matrix add feeding one matrix-plus-scalar broadcast; the two
// inner scalar broadcasts are gone.
// CHECK-LABEL: func.func @broadcast_min
// CHECK: %[[SS:.*]] = "daphne.ewAdd"(%{{.*}}, %{{.*}}) : (si64, si64) -> si64
// CHECK: %[[MS:.*]] = "daphne.ewAdd"(%{{.*}}, %{{.*}}) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>)
// CHECK: "daphne.ewAdd"(%[[MS]], %[[SS]])
func.func @broadcast_min(%m1: !daphne.Matrix<2x3xsi64>, %s1: si64, %m2: !daphne.Matrix<2x3xsi64>, %s2: si64)
        -> !daphne.Matrix<2x3xsi64> {
    %a = "daphne.ewAdd"(%m1, %s1) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %b = "daphne.ewAdd"(%m2, %s2) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%a, %b) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// A scalar may sit on either side of an inner add (EwAdd is not commutative, so
// both positions are matched). Here the first inner add has the scalar as its
// lhs; the regrouping still fires.
// CHECK-LABEL: func.func @broadcast_min_scalar_lhs
// CHECK: "daphne.ewAdd"(%{{.*}}, %{{.*}}) : (si64, si64) -> si64
// CHECK: "daphne.ewAdd"(%{{.*}}, %{{.*}}) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>)
func.func @broadcast_min_scalar_lhs(%m1: !daphne.Matrix<2x3xsi64>, %s1: si64, %m2: !daphne.Matrix<2x3xsi64>, %s2: si64)
        -> !daphne.Matrix<2x3xsi64> {
    %a = "daphne.ewAdd"(%s1, %m1) : (si64, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    %b = "daphne.ewAdd"(%m2, %s2) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%a, %b) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Negative case: integer-only, because float addition is not associative.
// Regrouping distinct float addends would change rounding, so the ewAdds must
// survive over a float element type.
// CHECK-LABEL: func.func @broadcast_min_float
// CHECK-COUNT-3: daphne.ewAdd
func.func @broadcast_min_float(%m1: !daphne.Matrix<2x3xf64>, %s1: f64, %m2: !daphne.Matrix<2x3xf64>, %s2: f64)
        -> !daphne.Matrix<2x3xf64> {
    %a = "daphne.ewAdd"(%m1, %s1) : (!daphne.Matrix<2x3xf64>, f64) -> !daphne.Matrix<2x3xf64>
    %b = "daphne.ewAdd"(%m2, %s2) : (!daphne.Matrix<2x3xf64>, f64) -> !daphne.Matrix<2x3xf64>
    %r = "daphne.ewAdd"(%a, %b) : (!daphne.Matrix<2x3xf64>, !daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xf64>) -> ()
}

// Negative case: all four element types must match. Here one scalar is si32 over
// si64 matrices, which would promote the accumulation width, so the rewrite fails
// closed and the ewAdds survive.
// CHECK-LABEL: func.func @broadcast_min_promoting
// CHECK-COUNT-3: daphne.ewAdd
func.func @broadcast_min_promoting(%m1: !daphne.Matrix<2x3xsi64>, %s1: si32, %m2: !daphne.Matrix<2x3xsi64>, %s2: si32)
        -> !daphne.Matrix<2x3xsi64> {
    %a = "daphne.ewAdd"(%m1, %s1) : (!daphne.Matrix<2x3xsi64>, si32) -> !daphne.Matrix<2x3xsi64>
    %b = "daphne.ewAdd"(%m2, %s2) : (!daphne.Matrix<2x3xsi64>, si32) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%a, %b) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Negative case: the two matrices must have the same statically-known shape.
// With shapes unknown the rewrite cannot prove the matrix add needs no
// broadcasting, so it fails closed.
// CHECK-LABEL: func.func @broadcast_min_unknown_shape
// CHECK-COUNT-3: daphne.ewAdd
func.func @broadcast_min_unknown_shape(%m1: !daphne.Matrix<?x?xsi64>, %s1: si64, %m2: !daphne.Matrix<?x?xsi64>, %s2: si64)
        -> !daphne.Matrix<?x?xsi64> {
    %a = "daphne.ewAdd"(%m1, %s1) : (!daphne.Matrix<?x?xsi64>, si64) -> !daphne.Matrix<?x?xsi64>
    %b = "daphne.ewAdd"(%m2, %s2) : (!daphne.Matrix<?x?xsi64>, si64) -> !daphne.Matrix<?x?xsi64>
    %r = "daphne.ewAdd"(%a, %b) : (!daphne.Matrix<?x?xsi64>, !daphne.Matrix<?x?xsi64>) -> !daphne.Matrix<?x?xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<?x?xsi64>) -> ()
}

// Negative case and termination witness: an inner add of two matrices is not a
// matrix-plus-scalar broadcast, so it does not classify, which is exactly the
// shape of this pattern's own output, confirming it cannot re-fire on its result.
// CHECK-LABEL: func.func @broadcast_min_two_matrix_inner
// CHECK-COUNT-3: daphne.ewAdd
func.func @broadcast_min_two_matrix_inner(%m1: !daphne.Matrix<2x3xsi64>, %m2: !daphne.Matrix<2x3xsi64>,
        %m3: !daphne.Matrix<2x3xsi64>, %s2: si64) -> !daphne.Matrix<2x3xsi64> {
    %a = "daphne.ewAdd"(%m1, %m2) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    %b = "daphne.ewAdd"(%m3, %s2) : (!daphne.Matrix<2x3xsi64>, si64) -> !daphne.Matrix<2x3xsi64>
    %r = "daphne.ewAdd"(%a, %b) : (!daphne.Matrix<2x3xsi64>, !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%r) : (!daphne.Matrix<2x3xsi64>) -> ()
}
