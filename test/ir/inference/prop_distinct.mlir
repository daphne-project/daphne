// RUN: split-file %s %t
// RUN: daphne-opt --inference %t/inference.mlir --verify-each | FileCheck %s
// RUN: daphne-opt %t/parse_print.mlir --verify-each | daphne-opt --verify-each | FileCheck %s --check-prefix=PARSE-PRINT
// RUN: daphne-opt %t/invalid.mlir --split-input-file --verify-diagnostics
// RUN: daphne-opt %t/transfer.mlir --transfer-data-props --verify-each | FileCheck %s --check-prefix=TRANSFER


//--- inference.mlir

// Original sequence inference: integer directions, floating steps, unknown operands.
func.func @seq_rules(%unknown: f64) {
  %c0 = "daphne.constant"() <{value = 0 : si64}> : () -> si64
  %c1 = "daphne.constant"() <{value = 10 : si64}> : () -> si64
  %c2 = "daphne.constant"() <{value = 2 : si64}> : () -> si64
  %ascending = "daphne.seq"(%c0, %c1, %c2) : (si64, si64, si64) -> !daphne.Matrix<?x1xsi64>
  %c3 = "daphne.constant"() <{value = -2 : si64}> : () -> si64
  %descending = "daphne.seq"(%c1, %c0, %c3) : (si64, si64, si64) -> !daphne.Matrix<?x1xsi64>
  %f0 = "daphne.constant"() <{value = 0.0 : f64}> : () -> f64
  %f1 = "daphne.constant"() <{value = 1.0 : f64}> : () -> f64
  %f2 = "daphne.constant"() <{value = 0.25 : f64}> : () -> f64
  %fractional = "daphne.seq"(%f0, %f1, %f2) : (f64, f64, f64) -> !daphne.Matrix<?x1xf64>
  %unknown_sequence = "daphne.seq"(%f0, %unknown, %f2) : (f64, f64, f64) -> !daphne.Matrix<?x1xf64>
  %s0 = "daphne.constant"() <{value = 0.0 : f32}> : () -> f32
  %s1 = "daphne.constant"() <{value = 1.0 : f32}> : () -> f32
  %s2 = "daphne.constant"() <{value = 0.25 : f32}> : () -> f32
  %single_precision = "daphne.seq"(%s0, %s1, %s2) : (f32, f32, f32) -> !daphne.Matrix<?x1xf32>
  return
}
// CHECK-LABEL: func.func @seq_rules(%arg0: f64) {
// CHECK-NEXT: %0 = "daphne.constant"() <{value = 0 : si64}> : () -> si64
// CHECK-NEXT: %1 = "daphne.constant"() <{value = 10 : si64}> : () -> si64
// CHECK-NEXT: %2 = "daphne.constant"() <{value = 2 : si64}> : () -> si64
// CHECK-NEXT: %3 = "daphne.seq"(%0, %1, %2) : (si64, si64, si64) -> !daphne.Matrix<6x1xsi64:sp[1.000000e+00]:numDistinct[6]>
// CHECK-NEXT: %4 = "daphne.constant"() <{value = -2 : si64}> : () -> si64
// CHECK-NEXT: %5 = "daphne.seq"(%1, %0, %4) : (si64, si64, si64) -> !daphne.Matrix<6x1xsi64:sp[1.000000e+00]:numDistinct[6]>
// CHECK-NEXT: %6 = "daphne.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
// CHECK-NEXT: %7 = "daphne.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
// CHECK-NEXT: %8 = "daphne.constant"() <{value = 2.500000e-01 : f64}> : () -> f64
// CHECK-NEXT: %9 = "daphne.seq"(%6, %7, %8) : (f64, f64, f64) -> !daphne.Matrix<5x1xf64:sp[1.000000e+00]:numDistinct[5]>
// CHECK-NEXT: %10 = "daphne.seq"(%6, %arg0, %8) : (f64, f64, f64) -> !daphne.Matrix<?x1xf64:sp[1.000000e+00]>
// CHECK-NEXT: %11 = "daphne.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
// CHECK-NEXT: %12 = "daphne.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
// CHECK-NEXT: %13 = "daphne.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
// CHECK-NEXT: %14 = "daphne.seq"(%11, %12, %13) : (f32, f32, f32) -> !daphne.Matrix<5x1xf32:sp[1.000000e+00]:numDistinct[5]>
// CHECK-NEXT: return
// CHECK-NEXT: }

func.func @fill_rules(%n: index, %unknown: f64) {
  // negative zero
  %c2 = "daphne.constant"() <{value = -0.0 : f64}> : () -> f64
  %c1 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %fill_negative_zero = "daphne.fill"(%c2, %c1, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  // nan
  %c4 = "daphne.constant"() <{value = 0x7FF8000000000000 : f64}> : () -> f64
  %fill_nan = "daphne.fill"(%c4, %c1, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  // empty rows
  %c13 = "daphne.constant"() <{value = 2.0 : f64}> : () -> f64
  %c14 = "daphne.constant"() <{value = 0 : index}> : () -> index
  %fill_empty_rows = "daphne.fill"(%c13, %c14, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  // unknown rows
  %fill_unknown_rows = "daphne.fill"(%c13, %n, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  // nonconstant
  %fill_nonconstant = "daphne.fill"(%unknown, %c1, %c1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
  return
}

// CHECK-LABEL: func.func @fill_rules(%arg0: index, %arg1: f64) {
// CHECK-NEXT: %0 = "daphne.constant"() <{value = -0.000000e+00 : f64}> : () -> f64
// CHECK-NEXT: %1 = "daphne.constant"() <{value = 2 : index}> : () -> index
// CHECK-NEXT: %2 = "daphne.fill"(%0, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]:numDistinct[1]>
// CHECK-NEXT: %3 = "daphne.constant"() <{value = 0x7FF8000000000000 : f64}> : () -> f64
// CHECK-NEXT: %4 = "daphne.fill"(%3, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]>
// CHECK-NEXT: %5 = "daphne.constant"() <{value = 2.000000e+00 : f64}> : () -> f64
// CHECK-NEXT: %6 = "daphne.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %7 = "daphne.fill"(%5, %6, %1) : (f64, index, index) -> !daphne.Matrix<0x2xf64:symmetric[false]:numDistinct[0]>
// CHECK-NEXT: %8 = "daphne.fill"(%5, %arg0, %1) : (f64, index, index) -> !daphne.Matrix<?x2xf64>
// CHECK-NEXT: %9 = "daphne.fill"(%arg1, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]>
// CHECK-NEXT: return
// CHECK-NEXT: }

// Both constant dialects must give the same fill inference, including NaN handling.
func.func @fill_arith_constants() {
  %size = arith.constant 2 : index
  %value = arith.constant 3.5 : f64
  %nan = arith.constant 0x7FF8000000000000 : f64
  %filled = "daphne.fill"(%value, %size, %size) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
  %nan_filled = "daphne.fill"(%nan, %size, %size) : (f64, index, index) -> !daphne.Matrix<?x?xf64>
  return
}
// CHECK-LABEL: func.func @fill_arith_constants() {
// CHECK-NEXT: %c2 = arith.constant 2 : index
// CHECK-NEXT: %cst = arith.constant 3.500000e+00 : f64
// CHECK-NEXT: %cst_0 = arith.constant 0x7FF8000000000000 : f64
// CHECK-NEXT: %0 = "daphne.fill"(%cst, %c2, %c2) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]:numDistinct[1]>
// CHECK-NEXT: %1 = "daphne.fill"(%cst_0, %c2, %c2) : (f64, index, index) -> !daphne.Matrix<2x2xf64:symmetric[true]>
// CHECK-NEXT: return
// CHECK-NEXT: }

func.func @fill_constant_types() {
  %size = "daphne.constant"() <{value = 2 : index}> : () -> index
  %text = "daphne.constant"() <{value = "hello"}> : () -> !daphne.String
  %strings = "daphne.fill"(%text, %size, %size) : (!daphne.String, index, index) -> !daphne.Matrix<?x?x!daphne.String>
  %indices = "daphne.fill"(%size, %size, %size) : (index, index, index) -> !daphne.Matrix<?x?xindex>
  %integer = "daphne.constant"() <{value = 7 : si32}> : () -> si32
  %integers = "daphne.fill"(%integer, %size, %size) : (si32, index, index) -> !daphne.Matrix<?x?xsi32>
  return
}
// CHECK-LABEL: func.func @fill_constant_types() {
// CHECK-NEXT: %0 = "daphne.constant"() <{value = 2 : index}> : () -> index
// CHECK-NEXT: %1 = "daphne.constant"() <{value = "hello"}> : () -> !daphne.String
// CHECK-NEXT: %2 = "daphne.fill"(%1, %0, %0) : (!daphne.String, index, index) -> !daphne.Matrix<2x2x!daphne.String:symmetric[true]:numDistinct[1]>
// CHECK-NEXT: %3 = "daphne.fill"(%0, %0, %0) : (index, index, index) -> !daphne.Matrix<2x2xindex:symmetric[true]:numDistinct[1]>
// CHECK-NEXT: %4 = "daphne.constant"() <{value = 7 : si32}> : () -> si32
// CHECK-NEXT: %5 = "daphne.fill"(%4, %0, %0) : (si32, index, index) -> !daphne.Matrix<2x2xsi32:symmetric[true]:numDistinct[1]>
// CHECK-NEXT: return
// CHECK-NEXT: }

//--- transfer.mlir
func.func @transfer_count() {
  %0 = "daphne.constant"() <{value = 1.0 : f64}> : () -> f64
  %1 = "daphne.constant"() <{value = 2 : index}> : () -> index
  %2 = "daphne.fill"(%0, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:numDistinct[1]>
  return
}
// TRANSFER-LABEL: func.func @transfer_count() {
// TRANSFER-NEXT: %0 = "daphne.constant"() <{value = 1.000000e+00 : f64}> : () -> f64
// TRANSFER-NEXT: %1 = "daphne.constant"() <{value = 2 : index}> : () -> index
// TRANSFER-NEXT: %2 = "daphne.fill"(%0, %1, %1) : (f64, index, index) -> !daphne.Matrix<2x2xf64:numDistinct[1]>
// TRANSFER-NEXT: %3 = "daphne.constant"() <{value = -1.000000e+00 : f64}> : () -> f64
// TRANSFER-NEXT: %4 = "daphne.constant"() <{value = -1 : si64}> : () -> si64
// TRANSFER-NEXT: %5 = "daphne.constant"() <{value = 1 : si64}> : () -> si64
// TRANSFER-NEXT: "daphne.transferProperties"(%2, %3, %4, %5) : (!daphne.Matrix<2x2xf64:numDistinct[1]>, f64, si64, si64) -> ()
// TRANSFER-NEXT: return
// TRANSFER-NEXT: }

//--- parse_print.mlir
func.func @distinct_parse_print(%arg0: !daphne.Matrix<2x3xf64:sp[5.000000e-01]:numDistinct[2]>, %arg1: !daphne.Matrix<?x?xf64:numDistinct[0]>) {
  return
}
// PARSE-PRINT-LABEL: func.func @distinct_parse_print(%arg0: !daphne.Matrix<2x3xf64:sp[5.000000e-01]:numDistinct[2]>, %arg1: !daphne.Matrix<?x?xf64:numDistinct[0]>) {
// PARSE-PRINT-NEXT: return
// PARSE-PRINT-NEXT: }

//--- invalid.mlir
// expected-error@+1 {{distinct count must be nonnegative}}
func.func @negative(%arg0: !daphne.Matrix<2x3xf64:numDistinct[-1]>) { return }
// -----
// expected-error@+1 {{duplicate distinct count}}
func.func @duplicate(%arg0: !daphne.Matrix<2x3xf64:numDistinct[0]:numDistinct[1]>) { return }
// -----
// expected-error@+1 {{expected integer value}}
func.func @fractional(%arg0: !daphne.Matrix<2x3xf64:numDistinct[1.5]>) { return }
