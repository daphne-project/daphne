// RUN: daphne-opt --canonicalize %s | FileCheck %s

// Locks in that scalar comparison ops constant-fold through the shared
// ScalarConstantFoldable driver, and that the folded truth value takes the op's
// result type: the most-general argument value type (si64/f64) for numeric
// inputs, or i1 for bool inputs. The individual cases pin down the value.

// Signed-integer comparison folds to 1 in the argument value type.
// CHECK-LABEL: func.func @lt_sint
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 1 : si64}>
// CHECK-NEXT: "daphne.return"(%[[C]])
// CHECK-NOT: daphne.ewLt
func.func @lt_sint() -> si64 {
    %0 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %2 = "daphne.ewLt"(%0, %1) : (si64, si64) -> si64
    "daphne.return"(%2) : (si64) -> ()
}

// The false case folds to 0.
// CHECK-LABEL: func.func @gt_sint
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 0 : si64}>
// CHECK-NEXT: "daphne.return"(%[[C]])
// CHECK-NOT: daphne.ewGt
func.func @gt_sint() -> si64 {
    %0 = "daphne.constant"() {value = 3 : si64} : () -> si64
    %1 = "daphne.constant"() {value = 5 : si64} : () -> si64
    %2 = "daphne.ewGt"(%0, %1) : (si64, si64) -> si64
    "daphne.return"(%2) : (si64) -> ()
}

// Float comparison folds to a float 0.0/1.0 in the (float) result type.
// CHECK-LABEL: func.func @eq_float
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 1.000000e+00 : f64}>
// CHECK-NEXT: "daphne.return"(%[[C]])
// CHECK-NOT: daphne.ewEq
func.func @eq_float() -> f64 {
    %0 = "daphne.constant"() {value = 2.0 : f64} : () -> f64
    %1 = "daphne.constant"() {value = 2.0 : f64} : () -> f64
    %2 = "daphne.ewEq"(%0, %1) : (f64, f64) -> f64
    "daphne.return"(%2) : (f64) -> ()
}

// An f32 comparison must fold too: the materialised 0.0/1.0 truth value has to
// be built in the result type's float semantics (a bare IEEEdouble truth value
// fails FloatAttr verification for f32 and would silently skip the fold).
// CHECK-LABEL: func.func @lt_f32
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 1.000000e+00 : f32}>
// CHECK-NEXT: "daphne.return"(%[[C]])
// CHECK-NOT: daphne.ewLt
func.func @lt_f32() -> f32 {
    %0 = "daphne.constant"() {value = 1.0 : f32} : () -> f32
    %1 = "daphne.constant"() {value = 2.0 : f32} : () -> f32
    %2 = "daphne.ewLt"(%0, %1) : (f32, f32) -> f32
    "daphne.return"(%2) : (f32) -> ()
}

// Bool operands are i1 (signless), so they fold through the bool comparison
// path rather than the signed/unsigned int paths. eq of true and false is
// false; the ewEq must be gone.
// CHECK-LABEL: func.func @eq_bool
// CHECK: "daphne.constant"() <{value = false}>
// CHECK-NOT: daphne.ewEq
func.func @eq_bool() -> i1 {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.ewEq"(%0, %1) : (i1, i1) -> i1
    "daphne.return"(%2) : (i1) -> ()
}

// neq of true and false is true.
// CHECK-LABEL: func.func @neq_bool
// CHECK: "daphne.constant"() <{value = true}>
// CHECK-NOT: daphne.ewNeq
func.func @neq_bool() -> i1 {
    %0 = "daphne.constant"() {value = true} : () -> i1
    %1 = "daphne.constant"() {value = false} : () -> i1
    %2 = "daphne.ewNeq"(%0, %1) : (i1, i1) -> i1
    "daphne.return"(%2) : (i1) -> ()
}
