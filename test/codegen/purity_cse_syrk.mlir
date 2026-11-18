// RUN: daphne-opt --cse %s | FileCheck %s

module {
  // Two identical syrk ops on the same operand. Now that syrk is Pure, CSE
  // folds them into one; without the trait CSE skips it and both survive.
  func.func @dup_syrk(%a: !daphne.Matrix<8x4xf64>) -> !daphne.Matrix<4x4xf64> {
    %t = "daphne.constant"() {value = true} : () -> i1
    %0 = "daphne.syrk"(%a, %t) : (!daphne.Matrix<8x4xf64>, i1) -> !daphne.Matrix<4x4xf64>
    %1 = "daphne.syrk"(%a, %t) : (!daphne.Matrix<8x4xf64>, i1) -> !daphne.Matrix<4x4xf64>
    %r = "daphne.ewAdd"(%0, %1) : (!daphne.Matrix<4x4xf64>, !daphne.Matrix<4x4xf64>) -> !daphne.Matrix<4x4xf64>
    return %r : !daphne.Matrix<4x4xf64>
    // CHECK-LABEL: func.func @dup_syrk
    // CHECK: "daphne.syrk"
    // CHECK-NOT: "daphne.syrk"
  }
}
