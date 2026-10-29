// RUN: daphne-opt --cse %s | FileCheck %s

// COM: Two identical transpose ops on the same operand must collapse to one under CSE.
// COM: This relies on transpose being marked Pure; CSE skips ops with side effects.

module {
  func.func @duplicate_transpose() {
    %0 = "daphne.constant"() {value = 2 : index} : () -> index
    %1 = "daphne.constant"() {value = 3 : index} : () -> index
    %2 = "daphne.constant"() {value = false} : () -> i1
    %3 = "daphne.constant"() {value = true} : () -> i1
    %4 = "daphne.constant"() {value = 1.000000e+00 : f64} : () -> f64
    %5 = "daphne.fill"(%4, %1, %0) : (f64, index, index) -> !daphne.Matrix<3x2xf64>
    // CHECK-COUNT-1: daphne.transpose
    // CHECK-NOT: daphne.transpose
    %6 = "daphne.transpose"(%5) : (!daphne.Matrix<3x2xf64>) -> !daphne.Matrix<2x3xf64>
    %7 = "daphne.transpose"(%5) : (!daphne.Matrix<3x2xf64>) -> !daphne.Matrix<2x3xf64>
    "daphne.print"(%6, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    "daphne.print"(%7, %3, %2) : (!daphne.Matrix<2x3xf64>, i1, i1) -> ()
    "daphne.return"() : () -> ()
  }
}
