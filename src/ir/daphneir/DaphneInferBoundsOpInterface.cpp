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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferBoundsOpInterface.cpp.inc>
}

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

// TODO: Support integer constants once the bounds representation is settled.
std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::RandMatrixOp::inferBounds() {
    // Try to get the generation bounds as floating-point constants.
    auto minC = CompilerUtils::isConstant<double>(getMin());
    auto maxC = CompilerUtils::isConstant<double>(getMax());
    if (minC.first && maxC.first && !std::isnan(minC.second) && !std::isnan(maxC.second)) {
        auto sparsity = CompilerUtils::isConstant<double>(getSparsity());
        if (sparsity.first && sparsity.second == 1.0)
            return {{minC.second, maxC.second}};
        // Unless fully dense, generated matrices may also contain zeros.
        return {{std::min(minC.second, 0.0), std::max(maxC.second, 0.0)}};
    }
    return {{std::nullopt, std::nullopt}};
}

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::FillOp::inferBounds() {
    // A constant floating-point fill has equal lower and upper bounds.
    auto fillC = CompilerUtils::isConstant<double>(getArg());
    if (fillC.first && !std::isnan(fillC.second))
        return {{fillC.second, fillC.second}};
    return {{std::nullopt, std::nullopt}};
}

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::OrderOp::inferBounds() {
    auto [known, returnIndices] = CompilerUtils::isConstant<bool>(getReturnIdxs());
    if (!known)
        return {{std::nullopt, std::nullopt}};
    auto mt = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    if (!mt)
        return {{std::nullopt, std::nullopt}};
    if (returnIndices) {
        const auto rows = mt.getNumRows();
        if (rows == 0)
            return {{std::nullopt, std::nullopt}};
        if (rows > 0)
            return {{0.0, static_cast<double>(rows - 1)}};
        return {{0.0, std::nullopt}};
    }
    // Ordering values preserves the input bounds
    return {{mt.getLowerBound(), mt.getUpperBound()}};
}

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::EwLtOp::inferBounds() {
    // Comparisons produce 0/1. Known input bounds may tighten the result bounds.
    auto lhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());
    if (lhsMatrix && rhsMatrix) {
        auto lhsLower = lhsMatrix.getLowerBound();
        auto lhsUpper = lhsMatrix.getUpperBound();
        auto rhsLower = rhsMatrix.getLowerBound();
        auto rhsUpper = rhsMatrix.getUpperBound();

        if (lhsUpper && rhsLower && *lhsUpper < *rhsLower)
            return {{1.0, 1.0}};
        if (lhsLower && rhsUpper && *lhsLower >= *rhsUpper)
            return {{0.0, 0.0}};
        return {{0.0, 1.0}};
    }
    auto rhsC = CompilerUtils::isConstant<double>(getRhs());
    if (!lhsMatrix || !rhsC.first)
        return {{0.0, 1.0}};

    auto matrixMin = lhsMatrix.getLowerBound();
    auto matrixMax = lhsMatrix.getUpperBound();
    double scalarValue = rhsC.second;

    if (matrixMin.has_value() && matrixMax.has_value()) {
        if (scalarValue > matrixMax.value())
            return {{1.0, 1.0}}; // all true
        else if (scalarValue <= matrixMin.value())
            return {{0.0, 0.0}}; // all false (<= because x < min => always false)
        else
            return {{0.0, 1.0}}; // mixed
    }
    return {{0.0, 1.0}};
}

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::EwNeqOp::inferBounds() {
    // Comparisons produce 0/1. Known input bounds may tighten the result bounds.
    auto lhsMatrix = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsC = CompilerUtils::isConstant<double>(getRhs());
    if (!lhsMatrix || !rhsC.first)
        return {{0.0, 1.0}};

    auto matrixMin = lhsMatrix.getLowerBound();
    auto matrixMax = lhsMatrix.getUpperBound();
    double scalarValue = rhsC.second;

    if (matrixMin.has_value() && matrixMax.has_value()) {
        if (scalarValue > matrixMax.value() || scalarValue < matrixMin.value())
            return {{1.0, 1.0}}; // always not equal
        // If both min and max equal the scalar, then all entries equal scalar
        if (matrixMin.value() == scalarValue && matrixMax.value() == scalarValue)
            return {{0.0, 0.0}}; // always equal -> not equal is false
        return {{0.0, 1.0}};     // otherwise could be mixed
    }
    return {{0.0, 1.0}};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<std::pair<std::optional<double>, std::optional<double>>> daphne::tryInferBounds(Operation *op) {
    if (auto inferBoundsOp = llvm::dyn_cast<daphne::InferBounds>(op))
        // If the operation implements the inference interface, we apply that.
        return inferBoundsOp.inferBounds();
    else {
        // If the operation does not implement the inference interface
        // we return unknown bounds for each result.
        std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds;
        for (size_t i = 0; i < op->getNumResults(); i++)
            bounds.push_back({std::nullopt, std::nullopt});
        return bounds;
    }
}
