// RUN: daphne-opt --canonicalize %s | FileCheck %s

module {
  func.func @hoist_slicerow_above_ewsqrt(%arg0: !daphne.Matrix<?x?xf64>, %lb: si64, %ub: si64) -> !daphne.Matrix<?x?xf64> {
    %0 = "daphne.ewSqrt"(%arg0) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64>
    %1 = "daphne.sliceRow"(%0, %lb, %ub) : (!daphne.Matrix<?x?xf64>, si64, si64) -> !daphne.Matrix<?x?xf64>
    "daphne.return"(%1) : (!daphne.Matrix<?x?xf64>) -> ()
    // CHECK-LABEL: func.func @hoist_slicerow_above_ewsqrt
    // CHECK: %[[SLICE:.*]] = "daphne.sliceRow"(%arg0, %arg1, %arg2)
    // CHECK: %[[SQRT:.*]] = "daphne.ewSqrt"(%[[SLICE]])
    // CHECK: "daphne.return"(%[[SQRT]])
  }
}
