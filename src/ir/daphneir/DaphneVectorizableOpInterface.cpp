/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>

#include <vector>

namespace mlir::daphne {
#include <ir/daphneir/DaphneVectorizableOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// Vector split and combine utility functions
// ****************************************************************************
// For families of operations.

// Elementwise binary
template <class EwBinaryOp> std::vector<daphne::VectorSplit> getVectorSplits_EwBinaryOp(EwBinaryOp *op) {
    auto isLhsMatrix = llvm::isa<daphne::MatrixType>(op->getLhs().getType());
    auto isRhsMatrix = llvm::isa<daphne::MatrixType>(op->getRhs().getType());

    auto lhsSplit = isLhsMatrix ? daphne::VectorSplit::ROWS : daphne::VectorSplit::NONE;
    auto rhsSplit = isRhsMatrix ? daphne::VectorSplit::ROWS : daphne::VectorSplit::NONE;
    return {lhsSplit, rhsSplit};
}
template <class EwBinaryOp> std::vector<daphne::VectorCombine> getVectorCombines_EwBinaryOp(EwBinaryOp *op) {
    return {daphne::VectorCombine::ROWS};
}
template <class EwBinaryOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_EwBinaryOp(EwBinaryOp *op, OpBuilder &builder) {
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto lhsRows = builder.create<daphne::NumRowsOp>(loc, sizeTy, op->getLhs());
    auto lhsCols = builder.create<daphne::NumColsOp>(loc, sizeTy, op->getLhs());
    // TODO: do max on #rows/#cols of lhs and rhs for broadcasting
    return {{lhsRows, lhsCols}};
}

// Outer binary
template <class OuterBinaryOp> std::vector<daphne::VectorSplit> getVectorSplits_OuterBinaryOp(OuterBinaryOp *op) {
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::NONE};
}
template <class OuterBinaryOp> std::vector<daphne::VectorCombine> getVectorCombines_OuterBinaryOp(OuterBinaryOp *op) {
    return {daphne::VectorCombine::ROWS};
}
template <class OuterBinaryOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_OuterBinaryOp(OuterBinaryOp *op, OpBuilder &builder) {
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto lhsRows = builder.create<daphne::NumRowsOp>(loc, sizeTy, op->getLhs());
    auto rhsCols = builder.create<daphne::NumColsOp>(loc, sizeTy, op->getRhs());
    return {{lhsRows, rhsCols}};
}

// Elementwise unary
template <class EwUnaryOp> std::vector<daphne::VectorSplit> getVectorSplits_EwUnaryOp(EwUnaryOp *op) {
    return {daphne::VectorSplit::ROWS};
}
template <class EwUnaryOp> std::vector<daphne::VectorCombine> getVectorCombines_EwUnaryOp(EwUnaryOp *op) {
    return {daphne::VectorCombine::ROWS};
}
template <class EwUnaryOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_EwUnaryOp(EwUnaryOp *op, OpBuilder &builder) {
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, op->getArg());
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, op->getArg());
    // TODO: do max on #rows/#cols of lhs and rhs for broadcasting
    return {{rows, cols}};
}

// Row-wise aggregation
template <class RowAggOp> std::vector<daphne::VectorSplit> getVectorSplits_RowAggOp(RowAggOp *op) {
    return {daphne::VectorSplit::ROWS};
}
template <class RowAggOp> std::vector<daphne::VectorCombine> getVectorCombines_RowAggOp(RowAggOp *op) {
    return {daphne::VectorCombine::ROWS};
}
template <class RowAggOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_RowAggOp(RowAggOp *op, OpBuilder &builder) {
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, op->getArg());
    auto cst1 = builder.create<daphne::ConstantOp>(loc, sizeTy, builder.getIndexAttr(1l));
    return {{rows, cst1}};
}

// Column-wise aggregation
template <class ColAggOp> std::vector<daphne::VectorSplit> getVectorSplits_ColAggOp(ColAggOp *op) {
    return {daphne::VectorSplit::ROWS};
}
template <class ColAggOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_ColAggOp(ColAggOp *op, OpBuilder &builder) {
    auto loc = op->getLoc();
    auto sizeTy = builder.getIndexType();
    auto cst1 = builder.create<daphne::ConstantOp>(loc, sizeTy, builder.getIndexAttr(1l));
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, op->getArg());
    return {{cst1, cols}};
}

// ****************************************************************************
// Vector split and combine implementations
// ****************************************************************************

// ----------------------------------------------------------------------------
// Matrix multiplication
// ----------------------------------------------------------------------------

std::vector<daphne::VectorSplit> daphne::MatMulOp::getVectorSplits() {
    return {
        daphne::VectorSplit::ROWS, // lhs
        daphne::VectorSplit::NONE, // rhs
        daphne::VectorSplit::NONE, // transa
        daphne::VectorSplit::NONE  // transb
    };
}
std::vector<daphne::VectorCombine> daphne::MatMulOp::getVectorCombines() { return {daphne::VectorCombine::ROWS}; }
std::vector<std::pair<Value, Value>> daphne::MatMulOp::createOpsOutputSizes(OpBuilder &builder) {
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();

    Value rows;
    bool ta = CompilerUtils::constantOrThrow<bool>(getTransa(), "VectorizableOpInterface::createOpsOutputSizes() for "
                                                                "MatMulOp cannot know the number "
                                                                "of rows of the result, because it is not known if the "
                                                                "lhs input is transposed");
    rows = ta ? builder.create<daphne::NumColsOp>(loc, sizeTy, getLhs()).getResult()
              : builder.create<daphne::NumRowsOp>(loc, sizeTy, getLhs()).getResult();

    Value cols;
    bool tb =
        CompilerUtils::constantOrThrow<bool>(getTransb(), "VectorizableOpInterface::createOpsOutputSizes() for "
                                                          "MatMulOp cannot know the number "
                                                          "of columns of the result, because it is not known if the "
                                                          "rhs input is transposed");
    cols = tb ? builder.create<daphne::NumRowsOp>(loc, sizeTy, getRhs()).getResult()
              : builder.create<daphne::NumColsOp>(loc, sizeTy, getRhs()).getResult();

    return {{rows, cols}};
}

// ----------------------------------------------------------------------------
// Elementwise binary
// ----------------------------------------------------------------------------

#define IMPL_SPLIT_COMBINE_EWBINARY(OP)                                                                                \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { return getVectorSplits_EwBinaryOp(this); }        \
    std::vector<daphne::VectorCombine> daphne::OP::getVectorCombines() { return getVectorCombines_EwBinaryOp(this); }  \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) {                        \
        return createOpsOutputSizes_EwBinaryOp(this, builder);                                                         \
    }

// Arithmetic
IMPL_SPLIT_COMBINE_EWBINARY(EwAddOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwSubOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwMulOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwDivOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwPowOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwModOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwLogOp)

// Min/max
IMPL_SPLIT_COMBINE_EWBINARY(EwMinOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwMaxOp)

// Logical
IMPL_SPLIT_COMBINE_EWBINARY(EwAndOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwOrOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwXorOp)

// Bitwise
IMPL_SPLIT_COMBINE_EWBINARY(EwBitwiseAndOp);

// Strings
IMPL_SPLIT_COMBINE_EWBINARY(EwConcatOp)

// Comparisons
IMPL_SPLIT_COMBINE_EWBINARY(EwEqOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwNeqOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwLtOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwLeOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwGtOp)
IMPL_SPLIT_COMBINE_EWBINARY(EwGeOp)

#undef IMPL_SPLIT_COMBINE_EWBINARY

// ----------------------------------------------------------------------------
// Outer binary (generalized outer product)
// ----------------------------------------------------------------------------

#define IMPL_SPLIT_COMBINE_OUTERBINARY(OP)                                                                             \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { return getVectorSplits_OuterBinaryOp(this); }     \
    std::vector<daphne::VectorCombine> daphne::OP::getVectorCombines() {                                               \
        return getVectorCombines_OuterBinaryOp(this);                                                                  \
    }                                                                                                                  \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) {                        \
        return createOpsOutputSizes_OuterBinaryOp(this, builder);                                                      \
    }

// Arithmetic
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterAddOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterSubOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterMulOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterDivOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterPowOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterModOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterLogOp)

// Min/max
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterMinOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterMaxOp)

// Logical
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterAndOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterOrOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterXorOp)

// Strings
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterConcatOp)

// Comparisons
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterEqOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterNeqOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterLtOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterLeOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterGtOp)
IMPL_SPLIT_COMBINE_OUTERBINARY(OuterGeOp)

#undef IMPL_SPLIT_COMBINE_OUTERBINARY

// ----------------------------------------------------------------------------
// Elementwise unary
// ----------------------------------------------------------------------------

#define IMPL_SPLIT_COMBINE_EWUNARY(OP)                                                                                 \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { return getVectorSplits_EwUnaryOp(this); }         \
    std::vector<daphne::VectorCombine> daphne::OP::getVectorCombines() { return getVectorCombines_EwUnaryOp(this); }   \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) {                        \
        return createOpsOutputSizes_EwUnaryOp(this, builder);                                                          \
    }

// Arithmetic/general math
IMPL_SPLIT_COMBINE_EWUNARY(EwMinusOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwAbsOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwSignOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwExpOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwLnOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwSqrtOp)

// Logical
IMPL_SPLIT_COMBINE_EWUNARY(EwNegOp)

// Rounding
IMPL_SPLIT_COMBINE_EWUNARY(EwRoundOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwFloorOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwCeilOp)

// Trigonometric
IMPL_SPLIT_COMBINE_EWUNARY(EwSinOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwCosOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwTanOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwSinhOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwCoshOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwTanhOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwAsinOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwAcosOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwAtanOp)

// Comparison
IMPL_SPLIT_COMBINE_EWUNARY(EwIsNanOp)

// Strings
IMPL_SPLIT_COMBINE_EWUNARY(EwLowerOp)
IMPL_SPLIT_COMBINE_EWUNARY(EwUpperOp)

#undef IMPL_SPLIT_COMBINE_EWUNARY

// ----------------------------------------------------------------------------
// Full aggregation
// ----------------------------------------------------------------------------

template <class AllAggOp> std::vector<daphne::VectorSplit> getVectorSplits_AllAggOp(AllAggOp *op) {
    return {daphne::VectorSplit::ROWS};
}
template <class AllAggOp>
std::vector<std::pair<Value, Value>> createOpsOutputSizes_AllAggOp(AllAggOp *op, OpBuilder &builder) {
    auto cst1 = builder.create<daphne::ConstantOp>(op->getLoc(), static_cast<uint64_t>(1));
    return {{cst1, cst1}};
}

#define IMPL_SPLIT_COMBINE_ALLAGG(OP)                                                                                  \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { return getVectorSplits_AllAggOp(this); }          \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) {                        \
        return createOpsOutputSizes_AllAggOp(this, builder);                                                           \
    }

IMPL_SPLIT_COMBINE_ALLAGG(AllAggSumOp)
std::vector<daphne::VectorCombine> daphne::AllAggSumOp::getVectorCombines() { return {daphne::VectorCombine::ADD}; }
IMPL_SPLIT_COMBINE_ALLAGG(AllAggMinOp)
std::vector<daphne::VectorCombine> daphne::AllAggMinOp::getVectorCombines() { return {daphne::VectorCombine::MIN}; }
IMPL_SPLIT_COMBINE_ALLAGG(AllAggMaxOp)
std::vector<daphne::VectorCombine> daphne::AllAggMaxOp::getVectorCombines() { return {daphne::VectorCombine::MAX}; }

#undef IMPL_SPLIT_COMBINE_ALLAGG

// ----------------------------------------------------------------------------
// Row/column-wise aggregation
// ----------------------------------------------------------------------------

#define IMPL_SPLIT_COMBINE_ROWAGG(OP)                                                                                  \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { return getVectorSplits_RowAggOp(this); }          \
    std::vector<daphne::VectorCombine> daphne::OP::getVectorCombines() { return getVectorCombines_RowAggOp(this); }    \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) {                        \
        return createOpsOutputSizes_RowAggOp(this, builder);                                                           \
    }

IMPL_SPLIT_COMBINE_ROWAGG(RowAggSumOp)
IMPL_SPLIT_COMBINE_ROWAGG(RowAggMinOp)
IMPL_SPLIT_COMBINE_ROWAGG(RowAggMaxOp)

#undef IMPL_SPLIT_COMBINE_ROWAGG

#define IMPL_SPLIT_COMBINE_COLAGG(OP)                                                                                  \
    std::vector<daphne::VectorSplit> daphne::OP::getVectorSplits() { return getVectorSplits_ColAggOp(this); }          \
    std::vector<std::pair<Value, Value>> daphne::OP::createOpsOutputSizes(OpBuilder &builder) {                        \
        return createOpsOutputSizes_ColAggOp(this, builder);                                                           \
    }

IMPL_SPLIT_COMBINE_COLAGG(ColAggSumOp)
std::vector<daphne::VectorCombine> daphne::ColAggSumOp::getVectorCombines() { return {daphne::VectorCombine::ADD}; }
IMPL_SPLIT_COMBINE_COLAGG(ColAggMinOp)
std::vector<daphne::VectorCombine> daphne::ColAggMinOp::getVectorCombines() { return {daphne::VectorCombine::MIN}; }
IMPL_SPLIT_COMBINE_COLAGG(ColAggMaxOp)
std::vector<daphne::VectorCombine> daphne::ColAggMaxOp::getVectorCombines() { return {daphne::VectorCombine::MAX}; }

#undef IMPL_SPLIT_COMBINE_COLAGG

// ----------------------------------------------------------------------------
// Left and right indexing
// ----------------------------------------------------------------------------

std::vector<daphne::VectorSplit> daphne::ExtractColOp::getVectorSplits() {
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::NONE};
}
std::vector<daphne::VectorCombine> daphne::ExtractColOp::getVectorCombines() { return {daphne::VectorCombine::ROWS}; }
std::vector<std::pair<Value, Value>> daphne::ExtractColOp::createOpsOutputSizes(OpBuilder &builder) {
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, getSource());
    // TODO: support scalar and maybe (based on definition of `ExtractColOp`)
    // apply some kind of `unique()` op
    auto cols = builder.create<daphne::NumRowsOp>(loc, sizeTy, getSelectedCols());
    return {{rows, cols}};
}

// ----------------------------------------------------------------------------
// Reorganization
// ----------------------------------------------------------------------------

std::vector<daphne::VectorSplit> daphne::TransposeOp::getVectorSplits() { return {daphne::VectorSplit::ROWS}; }
std::vector<daphne::VectorCombine> daphne::TransposeOp::getVectorCombines() { return {daphne::VectorCombine::COLS}; }
std::vector<std::pair<Value, Value>> daphne::TransposeOp::createOpsOutputSizes(OpBuilder &builder) {
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, getArg());
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, getArg());
    return {{cols, rows}};
}

std::vector<daphne::VectorSplit> daphne::ColBindOp::getVectorSplits() {
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::ROWS};
}
std::vector<daphne::VectorCombine> daphne::ColBindOp::getVectorCombines() { return {daphne::VectorCombine::ROWS}; }
std::vector<std::pair<Value, Value>> daphne::ColBindOp::createOpsOutputSizes(OpBuilder &builder) {
    auto loc = getLoc();
    auto i64Ty = builder.getIntegerType(64, true);
    auto sizeTy = builder.getIndexType();
    auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, getLhs());
    auto colsLhs = builder.create<daphne::NumColsOp>(loc, sizeTy, getLhs());
    auto colsRhs = builder.create<daphne::NumColsOp>(loc, sizeTy, getRhs());
    return {{rows, builder.create<daphne::CastOp>(
                       loc, sizeTy,
                       builder.create<daphne::EwAddOp>(loc, i64Ty, builder.create<daphne::CastOp>(loc, i64Ty, colsLhs),
                                                       builder.create<daphne::CastOp>(loc, i64Ty, colsRhs)))}};
}

// ----------------------------------------------------------------------------
// Other
// ----------------------------------------------------------------------------

std::vector<daphne::VectorSplit> daphne::SyrkOp::getVectorSplits() {
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::NONE};
}
std::vector<daphne::VectorCombine> daphne::SyrkOp::getVectorCombines() { return {daphne::VectorCombine::ADD}; }
std::vector<std::pair<Value, Value>> daphne::SyrkOp::createOpsOutputSizes(OpBuilder &builder) {
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();
    // TODO: do max on #rows/#cols of lhs and rhs for broadcasting
    if (CompilerUtils::constantOrThrow<bool>(getTransLeft(),
                                             "argument transLeft of SyrkOp must be a compile-time constant")) {
        // This SyrkOp calculates `t(X) @ X`.
        auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, getArg());
        return {{cols, cols}};
    } else {
        // This SyrkOp calculates `X @ t(X)`.
        auto rows = builder.create<daphne::NumRowsOp>(loc, sizeTy, getArg());
        return {{rows, rows}};
    }
}

std::vector<daphne::VectorSplit> daphne::GemvOp::getVectorSplits() {
    return {daphne::VectorSplit::ROWS, daphne::VectorSplit::ROWS};
}
std::vector<daphne::VectorCombine> daphne::GemvOp::getVectorCombines() { return {daphne::VectorCombine::ADD}; }
std::vector<std::pair<Value, Value>> daphne::GemvOp::createOpsOutputSizes(OpBuilder &builder) {
    auto loc = getLoc();
    auto sizeTy = builder.getIndexType();
    auto cols = builder.create<daphne::NumColsOp>(loc, sizeTy, getMat());
    auto one = builder.create<daphne::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(1));
    return {{cols, one}};
}
