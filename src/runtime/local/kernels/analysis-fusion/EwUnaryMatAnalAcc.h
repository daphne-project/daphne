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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMATANALYSISAcc_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMATANALYSISAcc_H

#include "ir/daphneir/DataPropertyTypes.h"
#include <limits>
#include <queue>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/EwUnarySca.h>
#include <runtime/local/kernels/UnaryOpCode.h>

#include <algorithm>
#include <type_traits>

#include <cstddef>

#include <runtime/local/kernels/analysis-fusion/AnalysisFlags.h>
#include <unordered_set>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg, typename TAnalysis> struct EwUnaryMatAnalysisAcc {
    static void apply(UnaryOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg, typename TAnalysis>
void ewUnaryMatAnalysisAcc(UnaryOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) {
    EwUnaryMatAnalysisAcc<DTRes, DTArg, TAnalysis>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT, AnalysisFlag... Fs>
struct EwUnaryMatAnalysisAcc<DenseMatrix<VT>, DenseMatrix<VT>, AnalysisFlags<Fs...>> {
    static void apply(UnaryOpCode opCode, DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        const VT *valuesArg = arg->getValues();
        VT *valuesRes = res->getValues();

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VT, AccessPattern::Dense, OutputMatrix::Dense, Fs...> analAcc;
        analAcc.init(numRows, numCols);

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                VT tmp = func(valuesArg[c], ctx);
                valuesRes[c] = tmp;
                analAcc.accumulate(r, c, tmp);
            }
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }
        analAcc.writeResult(res);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT, AnalysisFlag... Fs>
struct EwUnaryMatAnalysisAcc<CSRMatrix<VT>, CSRMatrix<VT>, AnalysisFlags<Fs...>> {
    static void apply(UnaryOpCode opCode, CSRMatrix<VT> *&res, const CSRMatrix<VT> *arg, DCTX(ctx)) {

        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);
        const VT zeroRes = func(0, ctx);
        const bool mapsZeroToZero = zeroRes == 0;

        const size_t nnzArg = arg->getNumNonZeros();
        const size_t maxNnzRes = mapsZeroToZero ? nnzArg : (numRows * numCols);

        if (res == nullptr)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, maxNnzRes, false);

        const VT *valuesArg = arg->getValues();
        const size_t *colIdxsArg = arg->getColIdxs();
        const size_t *rowOffsetsArg = arg->getRowOffsets();
        VT *valuesRes = res->getValues();
        size_t *colIdxsRes = res->getColIdxs();
        size_t *rowOffsetsRes = res->getRowOffsets();

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VT, AccessPattern::Sparse, OutputMatrix::Sparse, Fs...> analAccSparse;
        AnalAcc<VT, AccessPattern::Dense, OutputMatrix::Sparse, Fs...> analAccDense;
        if (mapsZeroToZero)
            analAccSparse.init(numRows, numCols);
        else
            analAccDense.init(numRows, numCols);
        if (mapsZeroToZero) {
            // Zeros in the argument are mapped to zeros in the result. We only need to process the non-zeros of the
            // argument, the colIdxs and rowOffsets stay as they are.
            if ((std::is_floating_point_v<VT> && UnaryOpCodeUtils::mapsNonZeroToNonZeroFloat(opCode)) ||
                (std::is_integral_v<VT> && UnaryOpCodeUtils::mapsNonZeroToNonZeroInt(opCode))) {
                // Non-zeros in the argument are mapped to non-zeros in the result. We don't need to check if func
                // yields a zero.
                for (size_t r = 0; r < numRows; r++) {
                    for (size_t i = rowOffsetsArg[r]; i < rowOffsetsArg[r + 1]; i++) {
                        VT tmp = func(valuesArg[i], ctx);
                        valuesRes[i] = tmp;
                        const size_t c = colIdxsArg[i];
                        analAccSparse.accumulate(r, c, tmp);
                    }
                }

                // TODO The result could share the colIdxs and rowOffsets with the arg.
                std::copy(colIdxsArg, colIdxsArg + nnzArg, colIdxsRes);
                std::copy(rowOffsetsArg, rowOffsetsArg + numRows + 1, rowOffsetsRes);
            } else {
                // Non-zeros in the argument may be mapped to zeros in the result. We need to check if func yields a
                // zero.
                size_t nnzRes = 0;
                rowOffsetsRes[0] = 0;
                for (size_t r = 0; r < numRows; r++) {
                    for (size_t i = rowOffsetsArg[r]; i < rowOffsetsArg[r + 1]; i++) {
                        VT tmp = func(valuesArg[i], ctx);
                        if (tmp != 0) {
                            valuesRes[nnzRes] = tmp;
                            const size_t c = colIdxsArg[i];
                            colIdxsRes[nnzRes] = colIdxsArg[i];
                            nnzRes++;
                            analAccSparse.accumulate(r, c, tmp);
                        }
                    }
                    rowOffsetsRes[r + 1] = nnzRes;
                }
            }
        } else {
            // Zeros in the argument are mapped to non-zeros in the result. We must also process the zeros of the
            // argument.
            size_t nnzRes = 0;
            rowOffsetsRes[0] = 0;
            for (size_t r = 0; r < numRows; r++) {
                size_t c = 0;
                size_t i = rowOffsetsArg[r];
                while (c < numCols && i < rowOffsetsArg[r + 1]) {
                    if (c == colIdxsArg[i]) {
                        VT tmp = func(valuesArg[i], ctx);
                        if (tmp) {
                            valuesRes[nnzRes] = tmp;
                            colIdxsRes[nnzRes] = c;
                            nnzRes++;
                        }
                        i++;
                        analAccDense.accumulate(r, c, tmp);

                    } else {
                        valuesRes[nnzRes] = zeroRes;
                        analAccDense.accumulate(r, c, zeroRes);
                        colIdxsRes[nnzRes] = c;
                        nnzRes++;
                    }
                    c++;
                }
                while (c < numCols) {
                    valuesRes[nnzRes] = zeroRes;
                    colIdxsRes[nnzRes] = c;
                    analAccDense.accumulate(r, c, zeroRes);
                    nnzRes++;
                    c++;
                }
                rowOffsetsRes[r + 1] = nnzRes;
            }
        }
        if (mapsZeroToZero)
            analAccSparse.writeResult(res);
        else
            analAccDense.writeResult(res);
    }
};

// // ----------------------------------------------------------------------------
// // Matrix <- Matrix
// // ----------------------------------------------------------------------------

template <typename VT, AnalysisFlag... Fs> struct EwUnaryMatAnalysisAcc<Matrix<VT>, Matrix<VT>, AnalysisFlags<Fs...>> {
    static void apply(UnaryOpCode opCode, Matrix<VT> *&res, const Matrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        // Prepare analysis
        AnalAcc<VT, AccessPattern::Dense, OutputMatrix::Matrix, Fs...> analAcc;
        analAcc.init(numRows, numCols);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c) {
                VT tmp = func(arg->get(r, c), ctx);
                res->append(r, c, tmp);
                analAcc.accumulate(r, c, tmp);
            }
        res->finishAppend();
        analAcc.writeResult(res);
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMATANALYSISAcc_H