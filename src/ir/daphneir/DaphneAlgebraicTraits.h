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

#ifndef SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_H
#define SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_H

#include "mlir/IR/OpDefinition.h"

// ****************************************************************************
// Algebraic traits
// ****************************************************************************

namespace mlir::OpTrait {

template <class ConcreteOp> class Involutive : public TraitBase<ConcreteOp, Involutive> {};

template <class ConcreteOp> class IdempotentUnary : public TraitBase<ConcreteOp, IdempotentUnary> {};

template <class ConcreteOp>
class IdentityOnIntegerElementType : public TraitBase<ConcreteOp, IdentityOnIntegerElementType> {};

template <class ConcreteOp> class IdentityWhenSymmetric : public TraitBase<ConcreteOp, IdentityWhenSymmetric> {};

template <class ConcreteOp> class OnlyReordersElements : public TraitBase<ConcreteOp, OnlyReordersElements> {};

template <class ConcreteOp> class OrderAgnosticAggregate : public TraitBase<ConcreteOp, OrderAgnosticAggregate> {};

template <class ConcreteOp> class NeutralOnZeroRHS : public TraitBase<ConcreteOp, NeutralOnZeroRHS> {};

template <class ConcreteOp> class NeutralOnOneRHS : public TraitBase<ConcreteOp, NeutralOnOneRHS> {};

template <class ConcreteOp> class LeftAbsorbingOnZero : public TraitBase<ConcreteOp, LeftAbsorbingOnZero> {};

template <class ConcreteOp> class RightAbsorbingOnZero : public TraitBase<ConcreteOp, RightAbsorbingOnZero> {};

} // namespace mlir::OpTrait

#endif // SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_H
