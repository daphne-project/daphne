/*
 * Copyright 2024 The DAPHNE Consortium
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

// Declarative canonicalization patterns (DRR). A rewrite that only matches an
// op shape and rewires SSA values, with no C++ predicate, no op built by hand,
// and no constant folding, is expressed most directly as a TableGen Pat<>.

#ifndef SRC_IR_DAPHNEIR_DAPHNECANONICALIZEPATTERNS_TD
#define SRC_IR_DAPHNEIR_DAPHNECANONICALIZEPATTERNS_TD

include "mlir/IR/PatternBase.td"
include "ir/daphneir/DaphneOps.td"

// A rename only distinguishes two DaphneDSL variable names aliasing the same
// SSA value during parsing; it carries no runtime meaning. Erase it by
// forwarding its single operand.
def RenameOpErase : Pat<(Daphne_RenameOp $arg), (replaceWithValue $arg)>;

#endif // SRC_IR_DAPHNEIR_DAPHNECANONICALIZEPATTERNS_TD
