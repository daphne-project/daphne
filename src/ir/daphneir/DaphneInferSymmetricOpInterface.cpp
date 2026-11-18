/*
 * Copyright 2025 The DAPHNE Consortium
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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

#include <vector>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferSymmetricOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

std::vector<BoolOrUnknown> daphne::FillOp::inferSymmetric() {
    // The result of FillOp is symmetric iff it is square.
    std::pair numRows = CompilerUtils::isConstant<ssize_t>(getNumRows());
    std::pair numCols = CompilerUtils::isConstant<ssize_t>(getNumCols());
    if (numRows.first && numCols.first) // the shape is known
        return {numRows.second == numCols.second ? BoolOrUnknown::True : BoolOrUnknown::False};
    // the shape is unknown
    return {BoolOrUnknown::Unknown};
}

std::vector<BoolOrUnknown> daphne::TransposeOp::inferSymmetric() {
    // TransposeOp retains the symmetry of its argument.
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(getArg().getType()))
        return {mt.getSymmetric()};
    return {BoolOrUnknown::Unknown};
}

std::vector<BoolOrUnknown> daphne::SyrkOp::inferSymmetric() {
    // SyrkOp computes t(A) @ A or A @ t(A), which is always symmetric.
    return {BoolOrUnknown::True};
}

std::vector<BoolOrUnknown> daphne::MatMulOp::inferSymmetric() {
    // A matrix product is symmetric when it has the "syrk" shape A @ t(A) or t(A) @ A: the two operands are the same
    // matrix and exactly one of them is transposed. Such a product is square and symmetric for any real A, regardless
    // of the (possibly unknown) static shape.
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();
    std::pair<bool, bool> transa = CompilerUtils::isConstant<bool>(getTransa());
    std::pair<bool, bool> transb = CompilerUtils::isConstant<bool>(getTransb());
    if (!transa.first || !transb.first)
        return {BoolOrUnknown::Unknown};
    // Folded form: matMul(X, X, false, true) / matMul(X, X, true, false).
    if (lhs == rhs && (transa.second != transb.second))
        return {BoolOrUnknown::True};
    // Un-folded form: matMul(X, transpose(X)) / matMul(transpose(X), X), which survives when the transpose was not
    // folded into a flag.
    if (!transa.second && !transb.second) {
        if (auto rhsT = rhs.getDefiningOp<daphne::TransposeOp>())
            if (rhsT.getArg() == lhs) // X @ t(X)
                return {BoolOrUnknown::True};
        if (auto lhsT = lhs.getDefiningOp<daphne::TransposeOp>())
            if (lhsT.getArg() == rhs) // t(X) @ X
                return {BoolOrUnknown::True};
    }
    return {BoolOrUnknown::Unknown};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<BoolOrUnknown> daphne::tryInferSymmetric(Operation *op) {
    if (auto inferSymmetricOp = llvm::dyn_cast<daphne::InferSymmetric>(op))
        // If the operation implements the inference interface, we apply that.
        return inferSymmetricOp.inferSymmetric();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<BoolOrUnknown> symmetrics;
        for (size_t i = 0; i < op->getNumResults(); i++)
            symmetrics.push_back(BoolOrUnknown::Unknown);
        return symmetrics;
    }
}