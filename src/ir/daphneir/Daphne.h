/*
 * Copyright 2021 The DAPHNE Consortium
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

#ifndef SRC_IR_DAPHNEIR_DAPHNE_H
#define SRC_IR_DAPHNEIR_DAPHNE_H

// The following includes are required by...
#include "llvm/ADT/StringRef.h"

// TODO Get rid of this workaround by removing the pragmas and the include
// within
//      (note that this header is also included transitively by FuncOps.h),
//      once the problem is fixed in MLIR/LLVM.
// As of MLIR llvm/llvm-project@20d454c79bbca7822eee88d188afb7a8747dac58,
// AttrTypeSubElements.h yields the following warnings, which are hereby
// ignored:
// - "... parameter 'derived' set but not used [-Wunused-but-set-parameter]"
// - "... parameter 'walkAttrsFn' set but not used [-Wunused-but-set-parameter]"
// - "... parameter 'walkTypesFn' set but not used [-Wunused-but-set-parameter]"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#include "mlir/IR/AttrTypeSubElements.h"
#pragma GCC diagnostic pop

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// TODO Get rid of this workaround by removing the pragmas,
//      once the problem is fixed in MLIR/LLVM.
// As of MLIR llvm/llvm-project@20d454c79bbca7822eee88d188afb7a8747dac58,
// PatternMatch.h yields the following warning, which is hereby ignored:
// - "... typedef 'using FnTraitsT = struct llvm::function_traits<PDLFnT>'
// locally defined but not used [-Wunused-local-typedefs]"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include "mlir/IR/PatternMatch.h"
#pragma GCC diagnostic pop

#include "mlir/Support/TypeID.h"

#include <ir/daphneir/DaphneAdaptTypesToKernelsTraits.h>
#include <ir/daphneir/DaphneDistributableOpInterface.h>
#include <ir/daphneir/DaphneOpsEnums.h.inc>
#include <ir/daphneir/DaphneTypeStorage.h>
#include <ir/daphneir/DaphneVectorizableOpInterface.h>

// Custom C++ types used to represent the data properties of DAPHNE data objects (e.g., matrices, frames) must be
// declared before including the data property inference interfaces.
#include <ir/daphneir/DataPropertyTypes.h>
namespace mlir::daphne {
std::string boolOrUnknownToString(BoolOrUnknown rep);
BoolOrUnknown stringToBoolOrUnknown(const std::string &str);
} // namespace mlir::daphne

#include <ir/daphneir/DaphneInferFrameLabelsOpInterface.h>
#include <ir/daphneir/DaphneInferShapeOpInterface.h>
#include <ir/daphneir/DaphneInferSparsityOpInterface.h>
#include <ir/daphneir/DaphneInferSymmetricOpInterface.h>
#include <ir/daphneir/DaphneInferMncSketchIdOpInterface.h>
#include <ir/daphneir/DaphneInferTypesOpInterface.h>

#include <string>
#include <utility>
#include <vector>

namespace mlir::OpTrait {
template <class ConcreteOp> class FPGAOPENCLSupport : public TraitBase<ConcreteOp, FPGAOPENCLSupport> {};
} // namespace mlir::OpTrait

namespace mlir::daphne {
enum class MatrixRepresentation {
    Dense = 0,
    // default is dense
    Default = MatrixRepresentation::Dense,
    Sparse = 1,
};

std::string matrixRepresentationToString(MatrixRepresentation rep);
std::ostream &operator<<(std::ostream &os, MatrixRepresentation rep);

MatrixRepresentation stringToMatrixRepresentation(const std::string &str);
} // namespace mlir::daphne

// ... the following tablegen'erated headers.
#define GET_TYPEDEF_CLASSES
#include "ir/daphneir/DaphneOpsDialect.h.inc"
#include <ir/daphneir/DaphneOpsTypes.h.inc>
#define GET_OP_CLASSES
#include "ir/daphneir/DaphneOps.h.inc"

#endif // SRC_IR_DAPHNEIR_DAPHNE_H
