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

#include <ir/daphneir/Daphne.h>

namespace mlir::daphne {
#include <ir/daphneir/DaphneDataPropertyRelevanceOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::daphne;

namespace {
void getComparisonProperties(OpOperand &lhs, OpOperand &rhs, unsigned operandIndex,
                             llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    if ((operandIndex == lhs.getOperandNumber() && isa<MatrixType>(lhs.get().getType())) ||
        (operandIndex == rhs.getOperandNumber() && isa<MatrixType>(rhs.get().getType())))
        properties.push_back(DataPropertyKind::Bounds);
}

void getJoinProperties(OpOperand &lhsValue, OpOperand &rhsValue, unsigned operandIndex,
                       llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    if (operandIndex == rhsValue.getOperandNumber() && isa<ColumnType>(lhsValue.get().getType()) &&
        isa<ColumnType>(rhsValue.get().getType()))
        properties.push_back(DataPropertyKind::Bounds);
}
} // namespace

void MatMulOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    if (!isa<MatrixType>(getLhs().getType()) || !isa<MatrixType>(getRhs().getType()))
        return;
    if (operandIndex == getLhsMutable().getOperandNumber())
        properties.push_back(DataPropertyKind::Sparsity);
    if (operandIndex == getLhsMutable().getOperandNumber() || operandIndex == getRhsMutable().getOperandNumber()) {
        properties.push_back(DataPropertyKind::NumRows);
        properties.push_back(DataPropertyKind::NumCols);
    }
}

void MapOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    if (operandIndex == getArgMutable().getOperandNumber() && isa<MatrixType>(getArg().getType()))
        properties.push_back(DataPropertyKind::NumDistinct);
}

void ColJoinOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getJoinProperties(getLhsDataMutable(), getRhsDataMutable(), operandIndex, properties);
}

void ColSemiJoinOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getJoinProperties(getLhsDataMutable(), getRhsDataMutable(), operandIndex, properties);
}

void EwLtOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwLeOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwGtOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwGeOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwEqOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwNeqOp::getRelevantProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}
