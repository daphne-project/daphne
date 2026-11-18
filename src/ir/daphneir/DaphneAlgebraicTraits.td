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

#ifndef SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_TD
#define SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_TD

include "mlir/IR/OpBase.td"

// ****************************************************************************
// Non-parametric algebraic traits
// ****************************************************************************

// f(f(x)) = x. Unlike MLIR's upstream IsInvolution, this tolerates ops
// whose result type differs from the operand type (e.g. TransposeOp).
def Involutive : NativeOpTrait<"Involutive">;

// f(f(x)) = f(x): f's output is already a fixed point (e.g. abs is non-negative,
// sign is in {-1, 0, 1}). Named to avoid upstream Idempotent, whose IsIdempotent
// static_asserts SameOperandsAndResultType (DAPHNE ops lack it) and also covers
// the binary g(x, x) = x case.
def IdempotentUnary : NativeOpTrait<"IdempotentUnary">;

// f(X) = X when X's element type is any integer.
def IdentityOnIntegerElementType : NativeOpTrait<"IdentityOnIntegerElementType">;

// f(X) = X when X's inferred symmetric property is true.
def IdentityWhenSymmetric : NativeOpTrait<"IdentityWhenSymmetric">;

// f(X) permutes X's elements without changing their multiset (e.g. transpose,
// reverse). Marks the producer half of the agg(reorder(X)) = agg(X) rewrite.
def OnlyReordersElements : NativeOpTrait<"OnlyReordersElements">;

// f(X) computes an order-independent reduction over all of X's elements, so it
// is invariant under any OnlyReordersElements producer feeding it.
def OrderAgnosticAggregate : NativeOpTrait<"OrderAgnosticAggregate">;

// ****************************************************************************
// Parametric algebraic traits (enumerated)
// ****************************************************************************

// Enumerated rather than TableGen-parametric: the constant K and operand
// position i cover only two small cardinalities in practice, so discrete
// defs beat one ParamNativeOpTrait that would need runtime attribute parsing.

// f(x, 0) = x.
def NeutralOnZeroRHS : NativeOpTrait<"NeutralOnZeroRHS">;

// f(x, 1) = x.
def NeutralOnOneRHS : NativeOpTrait<"NeutralOnOneRHS">;

// f(0, x) = 0.
def LeftAbsorbingOnZero : NativeOpTrait<"LeftAbsorbingOnZero">;

// f(x, 0) = 0.
def RightAbsorbingOnZero : NativeOpTrait<"RightAbsorbingOnZero">;

#endif // SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_TD
