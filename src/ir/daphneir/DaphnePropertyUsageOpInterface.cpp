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
#include <ir/daphneir/DaphnePropertyUsageOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::daphne;

namespace {
void getComparisonProperties(OpOperand &lhs, OpOperand &rhs, unsigned operandIndex,
                             llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    if ((operandIndex == lhs.getOperandNumber() && isa<MatrixType>(lhs.get().getType())) ||
        (operandIndex == rhs.getOperandNumber() && isa<MatrixType>(rhs.get().getType())))
        properties.push_back(DataPropertyKind::ValueBounds);
}

void getJoinProperties(OpOperand &lhsValue, OpOperand &rhsValue, unsigned operandIndex,
                       llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    if (operandIndex == rhsValue.getOperandNumber() && isa<ColumnType>(lhsValue.get().getType()) &&
        isa<ColumnType>(rhsValue.get().getType()))
        properties.push_back(DataPropertyKind::ValueBounds);
}
} // namespace

void MatMulOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    if (operandIndex == getLhsMutable().getOperandNumber() && isa<MatrixType>(getLhs().getType()) &&
        isa<MatrixType>(getRhs().getType()))
        properties.push_back(DataPropertyKind::NonzeroCount);
}

void MapOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    if (operandIndex == getArgMutable().getOperandNumber() && isa<MatrixType>(getArg().getType()))
        properties.push_back(DataPropertyKind::DistinctCount);
}

void ColJoinOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getJoinProperties(getLhsDataMutable(), getRhsDataMutable(), operandIndex, properties);
}

void ColSemiJoinOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getJoinProperties(getLhsDataMutable(), getRhsDataMutable(), operandIndex, properties);
}

void EwLtOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwLeOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwGtOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwGeOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwEqOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}

void EwNeqOp::getUsedProperties(unsigned operandIndex, llvm::SmallVectorImpl<DataPropertyKind> &properties) {
    getComparisonProperties(getLhsMutable(), getRhsMutable(), operandIndex, properties);
}
