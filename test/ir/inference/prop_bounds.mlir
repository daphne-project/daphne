// RUN: split-file %s %t
// RUN: daphne-opt %t/inference.mlir --inference --verify-each | FileCheck %s --check-prefix=INFER
// RUN: daphne-opt %t/inference.mlir --verify-each | FileCheck %s --check-prefix=BASELINE
// RUN: daphne-opt %t/identity.mlir --canonicalize --verify-each | FileCheck %s --check-prefix=IDENTITY

// RUN: daphne-opt %t/transfer.mlir --transfer-data-props --verify-each | FileCheck %s --check-prefix=TRANSFER

//--- inference.mlir
// A constant fill establishes equal bounds and a nonconstant fill stays unknown
func.func @fill(%arg0: f64) {
  %0 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %1 = "daphne.constant"() <{value = 3.5 : f64}> : () -> f64
  %2 = "daphne.fill"(%1, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
  %3 = "daphne.fill"(%arg0, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
  return
}
// INFER-LABEL: func.func @fill(
// INFER: %2 = "daphne.fill"(%1, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64{{[^>]*}}:lowerBound[3.500000e+00]{{[^>]*}}:upperBound[3.500000e+00]{{[^>]*}}>
// INFER: %3 = "daphne.fill"(%arg0, %0, %0) : (f64, index, index) -> !daphne.Matrix<2x2xf64
// BASELINE-LABEL: func.func @fill(
// BASELINE: %2 = "daphne.fill"(%1, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
// BASELINE: %3 = "daphne.fill"(%arg0, %0, %0) : (f64, index, index) -> !daphne.Matrix<?x?xf64>

// Shape inference reconstructs the matrix type without losing bounds
func.func @shape(%arg0: !daphne.Matrix<2x3xf64:lowerBound[-2.0]:upperBound[4.25]>) {
  %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x3xf64:lowerBound[-2.0]:upperBound[4.25]>) -> !daphne.Matrix<?x?xf64:lowerBound[-2.0]:upperBound[4.25]>
  return
}
// INFER-LABEL: func.func @shape(
// INFER: %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x3xf64:lowerBound[-2.000000e+00]:upperBound[4.250000e+00]>) -> !daphne.Matrix<2x3xf64{{[^>]*}}:lowerBound[-2.000000e+00]{{[^>]*}}:upperBound[4.250000e+00]{{[^>]*}}>

// Constant random limits establish bounds, a dynamic limit leaves them unknown
// Density is one so implicit zeros cannot fall outside the requested interval
func.func @random_bounds(%arg0: f64) {
  %0 = "daphne.constant"() <{value = 3 : index}> : () -> index
  %1 = "daphne.constant"() <{value = -2.0 : f64}> : () -> f64
  %2 = "daphne.constant"() <{value = 4.0 : f64}> : () -> f64
  %3 = "daphne.constant"() <{value = 1.0 : f64}> : () -> f64
  %4 = "daphne.constant"() <{value = 42 : si64}> : () -> si64
  %5 = "daphne.randMatrix"(%0, %0, %1, %2, %3, %4) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
  %6 = "daphne.randMatrix"(%0, %0, %arg0, %2, %3, %4) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
  return
}
// INFER-LABEL: func.func @random_bounds(
// INFER: %5 = "daphne.randMatrix"(%0, %0, %1, %2, %3, %4) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<3x3xf64{{[^>]*}}:lowerBound[-2.000000e+00]{{[^>]*}}:upperBound[4.000000e+00]{{[^>]*}}>
// INFER: %6 = "daphne.randMatrix"(%0, %0, %arg0, %2, %3, %4) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<3x3xf64

// Sparse or unknown density must include zero, even outside the generation interval
func.func @random_zeros(%arg0: f64) {
  %0 = "daphne.constant"() <{value = 3 : index}> : () -> index
  %1 = "daphne.constant"() <{value = 2.0 : f64}> : () -> f64
  %2 = "daphne.constant"() <{value = 4.0 : f64}> : () -> f64
  %3 = "daphne.constant"() <{value = -4.0 : f64}> : () -> f64
  %4 = "daphne.constant"() <{value = -2.0 : f64}> : () -> f64
  %5 = "daphne.constant"() <{value = 0.5 : f64}> : () -> f64
  %6 = "daphne.constant"() <{value = 42 : si64}> : () -> si64
  %7 = "daphne.randMatrix"(%0, %0, %1, %2, %5, %6) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
  %8 = "daphne.randMatrix"(%0, %0, %3, %4, %5, %6) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
  %9 = "daphne.randMatrix"(%0, %0, %1, %2, %arg0, %6) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<?x?xf64>
  return
}
// INFER-LABEL: func.func @random_zeros(
// INFER: %7 = "daphne.randMatrix"(%0, %0, %1, %2, %5, %6) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<3x3xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[4.000000e+00]{{[^>]*}}>
// INFER: %8 = "daphne.randMatrix"(%0, %0, %3, %4, %5, %6) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<3x3xf64{{[^>]*}}:lowerBound[-4.000000e+00]{{[^>]*}}:upperBound[0.000000e+00]{{[^>]*}}>
// INFER: %9 = "daphne.randMatrix"(%0, %0, %1, %2, %arg0, %6) : (index, index, f64, f64, f64, si64) -> !daphne.Matrix<3x3xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[4.000000e+00]{{[^>]*}}>

// Sorting values preserves known bounds, unknown bounds remain unknown
// Preserve input bounds only when returning sorted values, row indices have bounds derived from the row count
func.func @order_bounds(%arg0: !daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, %arg1: !daphne.Matrix<3x1xf64>, %arg2: i1) {
  %0 = "daphne.constant"() <{value = 0 : index}> : () -> index
  %1 = "daphne.constant"() <{value = true}> : () -> i1
  %2 = "daphne.constant"() <{value = false}> : () -> i1
  %3 = "daphne.order"(%arg0, %0, %1, %2) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, index, i1, i1) -> !daphne.Matrix<?x?xf64>
  %4 = "daphne.order"(%arg1, %0, %1, %2) : (!daphne.Matrix<3x1xf64>, index, i1, i1) -> !daphne.Matrix<?x?xf64>
  %5 = "daphne.order"(%arg0, %0, %1, %1) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, index, i1, i1) -> !daphne.Matrix<?x?xindex>
  %6 = "daphne.order"(%arg0, %0, %1, %arg2) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, index, i1, i1) -> !daphne.Matrix<?x?xindex>
  return
}
// INFER-LABEL: func.func @order_bounds(
// INFER: %3 = "daphne.order"(%arg0, %0, %1, %2) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, index, i1, i1) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[-2.000000e+00]{{[^>]*}}:upperBound[4.000000e+00]{{[^>]*}}>
// INFER: %4 = "daphne.order"(%arg1, %0, %1, %2) : (!daphne.Matrix<3x1xf64>, index, i1, i1) -> !daphne.Matrix<3x1xf64
// INFER: %5 = "daphne.order"(%arg0, %0, %1, %1) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, index, i1, i1) -> !daphne.Matrix<3x1xindex{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[2.000000e+00]{{[^>]*}}>
// INFER: %6 = "daphne.order"(%arg0, %0, %1, %arg2) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, index, i1, i1) -> !daphne.Matrix<3x?xindex

// Finite bounds: all true, all false (including equality at the lower bound),
// potentially mixed, and [0, 1] bounds for unknown inputs or scalars
func.func @comparison_bounds(%arg0: !daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, %arg1: !daphne.Matrix<3x1xf64:lowerBound[4.0]:upperBound[4.0]>, %arg2: !daphne.Matrix<3x1xf64>, %arg3: f64) {
  %0 = "daphne.constant"() <{value = -2.0 : f64}> : () -> f64
  %1 = "daphne.constant"() <{value = 4.0 : f64}> : () -> f64
  %2 = "daphne.constant"() <{value = 5.0 : f64}> : () -> f64
  %3 = "daphne.ewLt"(%arg0, %2) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, f64) -> !daphne.Matrix<?x?xf64>
  %4 = "daphne.ewLt"(%arg0, %0) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, f64) -> !daphne.Matrix<?x?xf64>
  %5 = "daphne.ewLt"(%arg0, %1) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, f64) -> !daphne.Matrix<?x?xf64>
  %6 = "daphne.ewNeq"(%arg0, %2) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, f64) -> !daphne.Matrix<?x?xf64>
  %7 = "daphne.ewNeq"(%arg1, %1) : (!daphne.Matrix<3x1xf64:lowerBound[4.0]:upperBound[4.0]>, f64) -> !daphne.Matrix<?x?xf64>
  %8 = "daphne.ewNeq"(%arg0, %1) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, f64) -> !daphne.Matrix<?x?xf64>
  %9 = "daphne.ewLt"(%arg2, %1) : (!daphne.Matrix<3x1xf64>, f64) -> !daphne.Matrix<?x?xf64>
  %10 = "daphne.ewNeq"(%arg2, %1) : (!daphne.Matrix<3x1xf64>, f64) -> !daphne.Matrix<?x?xf64>
  %11 = "daphne.ewLt"(%arg0, %arg3) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, f64) -> !daphne.Matrix<?x?xf64>
  %12 = "daphne.ewNeq"(%arg0, %arg3) : (!daphne.Matrix<3x1xf64:lowerBound[-2.0]:upperBound[4.0]>, f64) -> !daphne.Matrix<?x?xf64>
  return
}
// INFER-LABEL: func.func @comparison_bounds(
// INFER: %3 = "daphne.ewLt"(%arg0, %2) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[1.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER: %4 = "daphne.ewLt"(%arg0, %0) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[0.000000e+00]{{[^>]*}}>
// INFER: %5 = "daphne.ewLt"(%arg0, %1) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER: %6 = "daphne.ewNeq"(%arg0, %2) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[1.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER: %7 = "daphne.ewNeq"(%arg1, %1) : (!daphne.Matrix<3x1xf64:lowerBound[4.000000e+00]:upperBound[4.000000e+00]>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[0.000000e+00]{{[^>]*}}>
// INFER: %8 = "daphne.ewNeq"(%arg0, %1) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER: %9 = "daphne.ewLt"(%arg2, %1) : (!daphne.Matrix<3x1xf64>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER: %10 = "daphne.ewNeq"(%arg2, %1) : (!daphne.Matrix<3x1xf64>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER: %11 = "daphne.ewLt"(%arg0, %arg3) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER: %12 = "daphne.ewNeq"(%arg0, %arg3) : (!daphne.Matrix<3x1xf64:lowerBound[-2.000000e+00]:upperBound[4.000000e+00]>, f64) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>

// Matrix comparisons: separated ranges, equal boundaries, overlap, and partial bounds.
func.func @matrix_comparisons(%arg0: !daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>, %arg1: !daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]:upperBound[8.000000e+00]>, %arg2: !daphne.Matrix<3x1xf64:lowerBound[3.000000e+00]:upperBound[6.000000e+00]>, %arg3: !daphne.Matrix<3x1xf64>, %arg4: !daphne.Matrix<3x1xf64:upperBound[3.000000e+00]>, %arg5: !daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]>, %arg6: !daphne.Matrix<3x1xf64:lowerBound[3.000000e+00]>) {
  %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]:upperBound[8.000000e+00]>) -> !daphne.Matrix<?x?xf64>
  %1 = "daphne.ewLt"(%arg1, %arg0) : (!daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]:upperBound[8.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>) -> !daphne.Matrix<?x?xf64>
  %2 = "daphne.ewLt"(%arg2, %arg0) : (!daphne.Matrix<3x1xf64:lowerBound[3.000000e+00]:upperBound[6.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>) -> !daphne.Matrix<?x?xf64>
  %3 = "daphne.ewLt"(%arg0, %arg2) : (!daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[3.000000e+00]:upperBound[6.000000e+00]>) -> !daphne.Matrix<?x?xf64>
  %4 = "daphne.ewLt"(%arg0, %arg3) : (!daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64>) -> !daphne.Matrix<?x?xf64>
  %5 = "daphne.ewLt"(%arg4, %arg5) : (!daphne.Matrix<3x1xf64:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]>) -> !daphne.Matrix<?x?xf64>
  %6 = "daphne.ewLt"(%arg5, %arg4) : (!daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]>, !daphne.Matrix<3x1xf64:upperBound[3.000000e+00]>) -> !daphne.Matrix<?x?xf64>
  %7 = "daphne.ewLt"(%arg4, %arg6) : (!daphne.Matrix<3x1xf64:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[3.000000e+00]>) -> !daphne.Matrix<?x?xf64>
  return
}
// INFER-LABEL: func.func @matrix_comparisons(
// INFER-NEXT: %0 = "daphne.ewLt"(%arg0, %arg1) : (!daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]:upperBound[8.000000e+00]>) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[1.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER-NEXT: %1 = "daphne.ewLt"(%arg1, %arg0) : (!daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]:upperBound[8.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[0.000000e+00]{{[^>]*}}>
// INFER-NEXT: %2 = "daphne.ewLt"(%arg2, %arg0) : (!daphne.Matrix<3x1xf64:lowerBound[3.000000e+00]:upperBound[6.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[0.000000e+00]{{[^>]*}}>
// INFER-NEXT: %3 = "daphne.ewLt"(%arg0, %arg2) : (!daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[3.000000e+00]:upperBound[6.000000e+00]>) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER-NEXT: %4 = "daphne.ewLt"(%arg0, %arg3) : (!daphne.Matrix<3x1xf64:lowerBound[1.000000e+00]:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER-NEXT: %5 = "daphne.ewLt"(%arg4, %arg5) : (!daphne.Matrix<3x1xf64:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]>) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[1.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER-NEXT: %6 = "daphne.ewLt"(%arg5, %arg4) : (!daphne.Matrix<3x1xf64:lowerBound[5.000000e+00]>, !daphne.Matrix<3x1xf64:upperBound[3.000000e+00]>) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[0.000000e+00]{{[^>]*}}>
// INFER-NEXT: %7 = "daphne.ewLt"(%arg4, %arg6) : (!daphne.Matrix<3x1xf64:upperBound[3.000000e+00]>, !daphne.Matrix<3x1xf64:lowerBound[3.000000e+00]>) -> !daphne.Matrix<3x1xf64{{[^>]*}}:lowerBound[0.000000e+00]{{[^>]*}}:upperBound[1.000000e+00]{{[^>]*}}>
// INFER-NEXT: return
// INFER-NEXT: }

//--- identity.mlir
// Large bounds must hash safely and identical types must be uniqued
func.func @identity(%arg0: !daphne.Matrix<2x3xf64:lowerBound[-1.0e100]:upperBound[1.0e100]>) -> (!daphne.Matrix<2x3xf64:lowerBound[-1.0e100]:upperBound[1.0e100]>) {
  %0 = "daphne.cast"(%arg0) : (!daphne.Matrix<2x3xf64:lowerBound[-1.0e100]:upperBound[1.0e100]>) -> !daphne.Matrix<2x3xf64:lowerBound[-1.0e100]:upperBound[1.0e100]>
  return %0 : !daphne.Matrix<2x3xf64:lowerBound[-1.0e100]:upperBound[1.0e100]>
}
// IDENTITY-LABEL: func.func @identity(
// IDENTITY-NEXT: return %arg0 : !daphne.Matrix<2x3xf64:lowerBound[-1.000000e+100]:upperBound[1.000000e+100]>

//--- transfer.mlir
// Transfer a known lower endpoint and an unknown upper endpoint independently.
func.func @transfer_bounds() {
  %0 = "daphne.constant"() <{value = 1.0 : f64}> : () -> f64
  %1 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %2 = "daphne.fill"(%0, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:lowerBound[-2.0]>
  return
}
// TRANSFER-LABEL: func.func @transfer_bounds() {
// TRANSFER-NEXT: %0 = "daphne.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
// TRANSFER-NEXT: %1 = "daphne.constant"() <{value = 2 : index}> : () -> index
// TRANSFER-NEXT: %2 = "daphne.fill"(%0, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:lowerBound[-2.000000e+00]>
// TRANSFER: %6 = "daphne.constant"() <{value = true}> : () -> i1
// TRANSFER-NEXT: %7 = "daphne.constant"() <{value = -2.000000e+00 : f64}> : () -> f64
// TRANSFER-NEXT: %8 = "daphne.constant"() <{value = false}> : () -> i1
// TRANSFER-NEXT: %9 = "daphne.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
// TRANSFER-NEXT: "daphne.transferProperties"(%2, %3, %4, %5, %6, %7, %8, %9) : (!daphne.Matrix<2x2xf64:lowerBound[-2.000000e+00]>, f64, si64, si64, i1, f64, i1, f64) -> ()
// TRANSFER-NEXT: return
// TRANSFER-NEXT: }
