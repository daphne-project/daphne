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

#pragma once

#include <limits>
#include <queue>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <runtime/local/kernels/analysis-fusion/AnalysisFlags.h>
#include <unordered_map>
#include <unordered_set>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs, typename TAnalysis> struct EwBinaryMatAnalAcc {
    static void apply(BinaryOpCode opCode, DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs, typename TAnalysis>
void ewBinaryMatAnalAcc(BinaryOpCode opCode, DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, DCTX(ctx)) {
    EwBinaryMatAnalAcc<DTRes, DTLhs, DTRhs, TAnalysis>::apply(opCode, res, lhs, rhs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTres, typename VTlhs, typename VTrhs, AnalysisFlag... Fs>
struct EwBinaryMatAnalAcc<DenseMatrix<VTres>, DenseMatrix<VTlhs>, DenseMatrix<VTrhs>, AnalysisFlags<Fs...>> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VTres> *&res, const DenseMatrix<VTlhs> *lhs,
                      const DenseMatrix<VTrhs> *rhs, DCTX(ctx)) {
        const size_t numRowsLhs = lhs->getNumRows();
        const size_t numColsLhs = lhs->getNumCols();
        const size_t numRowsRhs = rhs->getNumRows();
        const size_t numColsRhs = rhs->getNumCols();

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VTres, AccessPattern::Dense, OutputMatrix::Dense, Fs...> analAcc;
        analAcc.init(numRowsLhs, numColsLhs);

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTres>>(numRowsLhs, numColsLhs, false);

        const VTlhs *valuesLhs = lhs->getValues();
        const VTrhs *valuesRhs = rhs->getValues();
        VTres *valuesRes = res->getValues();

        EwBinaryScaFuncPtr<VTres, VTlhs, VTrhs> func = getEwBinaryScaFuncPtr<VTres, VTlhs, VTrhs>(opCode);

        if (numRowsLhs == numRowsRhs && numColsLhs == numColsRhs) {
            // matrix op matrix (same size)
            for (size_t r = 0; r < numRowsLhs; r++) {
                for (size_t c = 0; c < numColsLhs; c++) {
                    VTres tmp = func(valuesLhs[c], valuesRhs[c], ctx);
                    valuesRes[c] = tmp;
                    analAcc.accumulate(r, c, tmp);
                }
                valuesLhs += lhs->getRowSkip();
                valuesRhs += rhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        } else if (numColsLhs == numColsRhs && (numRowsRhs == 1 || numRowsLhs == 1)) {
            // matrix op row-vector
            for (size_t r = 0; r < numRowsLhs; r++) {
                for (size_t c = 0; c < numColsLhs; c++) {
                    VTres tmp = func(valuesLhs[c], valuesRhs[c], ctx);
                    valuesRes[c] = tmp;
                    analAcc.accumulate(r, c, tmp);
                }
                valuesLhs += lhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        } else if (numRowsLhs == numRowsRhs && (numColsRhs == 1 || numColsLhs == 1)) {
            // matrix op col-vector
            for (size_t r = 0; r < numRowsLhs; r++) {
                for (size_t c = 0; c < numColsLhs; c++) {
                    VTres tmp = func(valuesLhs[c], valuesRhs[0], ctx);
                    valuesRes[c] = tmp;
                    analAcc.accumulate(r, c, tmp);
                }
                valuesLhs += lhs->getRowSkip();
                valuesRhs += rhs->getRowSkip();
                valuesRes += res->getRowSkip();
            }
        } else {
            throw std::runtime_error("EwBinaryMat(Dense) - lhs and rhs must either "
                                     "have the same dimensions, or one of them must be a row/column "
                                     "vector "
                                     "with the width/height of the other, but lhs has shape (" +
                                     std::to_string(numRowsLhs) + " x " + std::to_string(numColsLhs) +
                                     ") and rhs has shape (" + std::to_string(numRowsRhs) + " x " +
                                     std::to_string(numColsRhs) + ")");
        }
        analAcc.writeResult(res);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT, AnalysisFlag... Fs>
struct EwBinaryMatAnalAcc<CSRMatrix<VT>, CSRMatrix<VT>, CSRMatrix<VT>, AnalysisFlags<Fs...>> {
    static void apply(BinaryOpCode opCode, CSRMatrix<VT> *&res, const CSRMatrix<VT> *lhs, const CSRMatrix<VT> *rhs,
                      DCTX(ctx)) {

        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        if (numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            throw std::runtime_error("EwBinaryMat(CSR) - lhs and rhs must have "
                                     "the same dimensions.");

        size_t maxNnz;
        switch (opCode) {
        case BinaryOpCode::ADD: // merge
            maxNnz = lhs->getNumNonZeros() + rhs->getNumNonZeros();
            break;
        case BinaryOpCode::MUL: // intersect
            maxNnz = std::min(lhs->getNumNonZeros(), rhs->getNumNonZeros());
            break;
        default:
            throw std::runtime_error("EwBinaryMat(CSR) - unknown BinaryOpCode");
        }

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VT, AccessPattern::Sparse, OutputMatrix::Sparse, Fs...> analAcc;
        analAcc.init(numRows, numCols);

        if (res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, maxNnz, false);

        size_t *rowOffsetsRes = res->getRowOffsets();

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        rowOffsetsRes[0] = 0;

        switch (opCode) {
        case BinaryOpCode::ADD: { // merge non-zero cells
            // there must exist a zero in the result if there aren't enough elements to cover the whole matrix.

            for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
                size_t nnzRowRhs = rhs->getNumNonZeros(rowIdx);
                if (nnzRowLhs && nnzRowRhs) {
                    // merge within row
                    const VT *valuesRowLhs = lhs->getValues(rowIdx);
                    const VT *valuesRowRhs = rhs->getValues(rowIdx);
                    VT *valuesRowRes = res->getValues(rowIdx);
                    const size_t *colIdxsRowLhs = lhs->getColIdxs(rowIdx);
                    const size_t *colIdxsRowRhs = rhs->getColIdxs(rowIdx);
                    size_t *colIdxsRowRes = res->getColIdxs(rowIdx);
                    size_t posLhs = 0;
                    size_t posRhs = 0;
                    size_t posRes = 0;
                    while (posLhs < nnzRowLhs && posRhs < nnzRowRhs) {
                        if (colIdxsRowLhs[posLhs] == colIdxsRowRhs[posRhs]) {
                            const size_t r = rowIdx;
                            const size_t c = colIdxsRowLhs[posLhs];

                            VT funcRes = func(valuesRowLhs[posLhs], valuesRowRhs[posRhs], ctx);
                            if (funcRes != VT(0)) {
                                analAcc.accumulate(r, c, funcRes);

                                valuesRowRes[posRes] = funcRes;
                                colIdxsRowRes[posRes] = colIdxsRowLhs[posLhs];
                                posRes++;
                            }
                            posLhs++;
                            posRhs++;
                        } else if (colIdxsRowLhs[posLhs] < colIdxsRowRhs[posRhs]) {
                            // the lhs had a value at an index where rhs doesn't have a value.
                            // the sum is simply the lhs at the index.
                            const size_t r = rowIdx;
                            const size_t c = colIdxsRowLhs[posLhs];

                            VT tmp = valuesRowLhs[posLhs];
                            valuesRowRes[posRes] = tmp;
                            colIdxsRowRes[posRes] = colIdxsRowLhs[posLhs];
                            posLhs++;
                            posRes++;
                            analAcc.accumulate(r, c, tmp);

                        } else {
                            // the rhs had a value at an index where lhs doesn't have a value.
                            // the sum is simply the rhs at the index.
                            const size_t r = rowIdx;
                            const size_t c = colIdxsRowRhs[posRhs];

                            VT tmp = valuesRowRhs[posRhs];
                            valuesRowRes[posRes] = tmp;
                            colIdxsRowRes[posRes] = colIdxsRowRhs[posRhs];
                            posRhs++;
                            posRes++;
                            analAcc.accumulate(r, c, tmp);
                        }
                    }
                    // we deal with numbers that are left over. one of the two must be empty, so it is ok to memcpy over
                    // both.
                    const size_t restRowLhs = nnzRowLhs - posLhs;
                    const size_t restRowRhs = nnzRowRhs - posRhs;
                    if constexpr (AnalysisFlags<Fs...>::template containsAny<
                                      AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::numDistinct,
                                      AnalysisFlag::sparsity, AnalysisFlag::symmetry>) {
                        for (size_t i = 0; i < restRowLhs; i++) {
                            VT tmp = valuesRowLhs[posLhs + i];
                            valuesRowRes[posRes + i] = tmp;

                            const size_t r = rowIdx;
                            const size_t c = colIdxsRowLhs[posLhs + i];

                            colIdxsRowRes[posRes + i] = c;

                            analAcc.accumulate(r, c, tmp);
                        }
                        for (size_t i = 0; i < restRowRhs; i++) {
                            VT tmp = valuesRowRhs[posRhs + i];
                            valuesRowRes[posRes + i] = tmp;
                            const size_t r = rowIdx;
                            const size_t c = colIdxsRowRhs[posRhs + i];
                            colIdxsRowRes[posRes + i] = c;
                            analAcc.accumulate(r, c, tmp);
                        }
                    } else {
                        // copy from left
                        memcpy(valuesRowRes + posRes, valuesRowLhs + posLhs, restRowLhs * sizeof(VT));
                        memcpy(colIdxsRowRes + posRes, colIdxsRowLhs + posLhs, restRowLhs * sizeof(size_t));
                        // copy from right
                        memcpy(valuesRowRes + posRes, valuesRowRhs + posRhs, restRowRhs * sizeof(VT));
                        memcpy(colIdxsRowRes + posRes, colIdxsRowRhs + posRhs, restRowRhs * sizeof(size_t));
                    }

                    rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + posRes + restRowLhs + restRowRhs;
                } else if (nnzRowLhs) {
                    // copy from left
                    if constexpr (AnalysisFlags<Fs...>::template containsAny<
                                      AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::numDistinct,
                                      AnalysisFlag::sparsity, AnalysisFlag::symmetry>) {
                        const VT *valuesRowLhs = lhs->getValues(rowIdx);
                        const size_t *colIdxsRowLhs = lhs->getColIdxs(rowIdx);
                        size_t *colIdxsRowRes = res->getColIdxs(rowIdx);
                        VT *valuesRowRes = res->getValues(rowIdx);

                        for (size_t i = 0; i < nnzRowLhs; i++) {
                            VT tmp = valuesRowLhs[i];
                            const size_t r = rowIdx;
                            const size_t c = colIdxsRowLhs[i];

                            valuesRowRes[i] = tmp;
                            colIdxsRowRes[i] = c;
                            analAcc.accumulate(r, c, tmp);
                        }

                    } else {
                        memcpy(res->getValues(rowIdx), lhs->getValues(rowIdx), nnzRowLhs * sizeof(VT));
                        memcpy(res->getColIdxs(rowIdx), lhs->getColIdxs(rowIdx), nnzRowLhs * sizeof(size_t));
                    }
                    // memcpy(res->getValues(rowIdx), lhs->getValues(rowIdx), nnzRowLhs * sizeof(VT));
                    // memcpy(res->getColIdxs(rowIdx), lhs->getColIdxs(rowIdx), nnzRowLhs * sizeof(size_t));
                    rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + nnzRowLhs;
                } else if (nnzRowRhs) {
                    // copy from right
                    if constexpr (AnalysisFlags<Fs...>::template containsAny<
                                      AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::numDistinct,
                                      AnalysisFlag::sparsity, AnalysisFlag::symmetry>) {
                        const VT *valuesRowRhs = rhs->getValues(rowIdx);
                        const size_t *colIdxsRowRhs = rhs->getColIdxs(rowIdx);
                        size_t *colIdxsRowRes = res->getColIdxs(rowIdx);
                        VT *valuesRowRes = res->getValues(rowIdx);

                        for (size_t i = 0; i < nnzRowRhs; i++) {
                            VT tmp = valuesRowRhs[i];
                            const size_t r = rowIdx;
                            const size_t c = colIdxsRowRhs[i];

                            valuesRowRes[i] = tmp;
                            colIdxsRowRes[i] = c;
                            analAcc.accumulate(r, c, tmp);
                        }
                    } else {
                        memcpy(res->getValues(rowIdx), rhs->getValues(rowIdx), nnzRowRhs * sizeof(VT));
                        memcpy(res->getColIdxs(rowIdx), rhs->getColIdxs(rowIdx), nnzRowRhs * sizeof(size_t));
                    }
                    rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + nnzRowRhs;
                } else
                    // empty row in result
                    rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx];
            }
            break;
        }
        case BinaryOpCode::MUL: { // intersect non-zero cells
            for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
                size_t nnzRowRhs = rhs->getNumNonZeros(rowIdx);
                if (nnzRowLhs && nnzRowRhs) {
                    // intersect within row
                    const VT *valuesRowLhs = lhs->getValues(rowIdx);
                    const VT *valuesRowRhs = rhs->getValues(rowIdx);
                    VT *valuesRowRes = res->getValues(rowIdx);
                    const size_t *colIdxsRowLhs = lhs->getColIdxs(rowIdx);
                    const size_t *colIdxsRowRhs = rhs->getColIdxs(rowIdx);
                    size_t *colIdxsRowRes = res->getColIdxs(rowIdx);
                    size_t posLhs = 0;
                    size_t posRhs = 0;
                    size_t posRes = 0;
                    while (posLhs < nnzRowLhs && posRhs < nnzRowRhs) {
                        if (colIdxsRowLhs[posLhs] == colIdxsRowRhs[posRhs]) {
                            const size_t colIdx = colIdxsRowLhs[posLhs];
                            VT tmp = func(valuesRowLhs[posLhs], valuesRowRhs[posRhs], ctx);
                            valuesRowRes[posRes] = tmp;
                            colIdxsRowRes[posRes] = colIdx;
                            posLhs++;
                            posRhs++;
                            posRes++;
                            analAcc.accumulate(rowIdx, colIdx, tmp);

                        } else if (colIdxsRowLhs[posLhs] < colIdxsRowRhs[posRhs])
                            posLhs++;
                        else
                            posRhs++;
                    }
                    rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + posRes;
                } else
                    // empty row in result
                    rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx];
            }
            break;
        }
        default:
            throw std::runtime_error("EwBinaryMat(CSR) - unknown BinaryOpCode");
        }
        analAcc.writeResult(res);
        // TODO Update number of non-zeros in result in the end.
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT, AnalysisFlag... Fs>
struct EwBinaryMatAnalAcc<CSRMatrix<VT>, CSRMatrix<VT>, DenseMatrix<VT>, AnalysisFlags<Fs...>> {
    static void apply(BinaryOpCode opCode, CSRMatrix<VT> *&res, const CSRMatrix<VT> *lhs, const DenseMatrix<VT> *rhs,
                      DCTX(ctx)) {

        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();
        // TODO: lhs broadcast
        if ((numRows != rhs->getNumRows() && rhs->getNumRows() != 1) ||
            (numCols != rhs->getNumCols() && rhs->getNumCols() != 1))
            throw std::runtime_error("EwBinaryMat(CSR) - lhs and rhs must have "
                                     "the same dimensions (or broadcast)");

        size_t maxNnz;
        switch (opCode) {
        case BinaryOpCode::MUL: // intersect
            maxNnz = lhs->getNumNonZeros();
            break;
        default:
            throw std::runtime_error("EwBinaryMat(CSR) - unknown BinaryOpCode");
        }

        if (res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, maxNnz, false);

        size_t *rowOffsetsRes = res->getRowOffsets();

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        rowOffsetsRes[0] = 0;

        // Initialize Analysis
        AnalAcc<VT, AccessPattern::Sparse, OutputMatrix::Sparse, Fs...> analAcc;
        analAcc.init(numRows, numCols);

        switch (opCode) {
        case BinaryOpCode::MUL: { // intersect non-zero cells
            for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
                size_t nnzRowLhs = lhs->getNumNonZeros(rowIdx);
                if (nnzRowLhs) {
                    // intersect within row
                    const VT *valuesRowLhs = lhs->getValues(rowIdx);
                    VT *valuesRowRes = res->getValues(rowIdx);
                    const size_t *colIdxsRowLhs = lhs->getColIdxs(rowIdx);
                    size_t *colIdxsRowRes = res->getColIdxs(rowIdx);
                    auto rhsRow = (rhs->getNumRows() == 1 ? 0 : rowIdx);
                    size_t posRes = 0;
                    for (size_t posLhs = 0; posLhs < nnzRowLhs; ++posLhs) {
                        auto rhsCol = (rhs->getNumCols() == 1 ? 0 : colIdxsRowLhs[posLhs]);
                        auto rVal = rhs->get(rhsRow, rhsCol);
                        if (rVal != 0) {
                            VT tmp = func(valuesRowLhs[posLhs], rVal, ctx);
                            VT colIdx = colIdxsRowLhs[posLhs];
                            valuesRowRes[posRes] = tmp;
                            colIdxsRowRes[posRes] = colIdx;
                            posRes++;
                            analAcc.accumulate(rowIdx, colIdx, tmp);
                        }
                    }
                    rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx] + posRes;
                } else
                    // empty row in result
                    rowOffsetsRes[rowIdx + 1] = rowOffsetsRes[rowIdx];
            }
            break;
        }
        default:
            throw std::runtime_error("EwBinaryMat(CSR) - unknown BinaryOpCode");
        }
        analAcc.writeResult(res);
        // TODO Update number of non-zeros in result in the end.
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT, AnalysisFlag... Fs>
struct EwBinaryMatAnalAcc<DenseMatrix<VT>, CSRMatrix<VT>, DenseMatrix<VT>, AnalysisFlags<Fs...>> {
    static void apply(BinaryOpCode opCode, DenseMatrix<VT> *&res, const CSRMatrix<VT> *lhs, const DenseMatrix<VT> *rhs,
                      DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        if (numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            throw std::runtime_error("EwBinaryMat(Dense<-CSR,Dense) - lhs and rhs must have the same dimensions");

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        VT *valuesRes = res->getValues();
        const VT *valuesRhs = rhs->getValues();
        const size_t rowSkipRes = res->getRowSkip();
        const size_t rowSkipRhs = rhs->getRowSkip();

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        // Initialize analysis
        AnalAcc<VT, AccessPattern::Dense, OutputMatrix::Dense, Fs...> analAcc;
        analAcc.init(numRows, numCols);

        for (size_t r = 0; r < numRows; r++) {
            const size_t nnzRow = lhs->getNumNonZeros(r);
            const VT *valuesRowLhs = lhs->getValues(r);
            const size_t *colIdxsRowLhs = lhs->getColIdxs(r);

            // the index of a nonzero value in the CSR internal representation
            size_t nzPos = 0;
            for (size_t c = 0; c < numCols; c++) {
                VT lhsVal;
                if (nzPos < nnzRow && colIdxsRowLhs[nzPos] == c) {
                    lhsVal = valuesRowLhs[nzPos];
                    nzPos++;
                } else {
                    lhsVal = VT(0);
                }

                VT tmp = func(lhsVal, valuesRhs[c], ctx);
                valuesRes[c] = tmp;
                analAcc.accumulate(r, c, tmp);
            }

            valuesRes += rowSkipRes;
            valuesRhs += rowSkipRhs;
        }
        analAcc.writeResult(res);
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, Matrix
// ----------------------------------------------------------------------------
template <typename VT, AnalysisFlag... Fs>
struct EwBinaryMatAnalAcc<Matrix<VT>, Matrix<VT>, Matrix<VT>, AnalysisFlags<Fs...>> {
    static void apply(BinaryOpCode opCode, Matrix<VT> *&res, const Matrix<VT> *lhs, const Matrix<VT> *rhs, DCTX(ctx)) {
        const size_t numRows = lhs->getNumRows();
        const size_t numCols = lhs->getNumCols();

        // Initialize Analysis
        AnalAcc<VT, AccessPattern::Dense, OutputMatrix::Matrix, Fs...> analAcc;
        analAcc.init(numRows, numCols);

        if (numRows != rhs->getNumRows() || numCols != rhs->getNumCols())
            throw std::runtime_error("EwBinaryMat - lhs and rhs must have the same dimensions.");

        // TODO Choose matrix implementation depending on expected number of
        // non-zeros.
        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        EwBinaryScaFuncPtr<VT, VT, VT> func = getEwBinaryScaFuncPtr<VT, VT, VT>(opCode);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c) {
                const VT tmp = func(lhs->get(r, c), rhs->get(r, c), ctx);
                res->append(r, c, tmp);
                analAcc.accumulate(r, c, tmp);
            }
        res->finishAppend();
        analAcc.writeResult(res);
    }
};
