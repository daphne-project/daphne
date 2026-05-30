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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H

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

#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/analysis-fusion/AnalysisResult.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg, typename TAnalysis> struct EwUnaryMatAnalysis {
    static void apply(UnaryOpCode opCode, DTRes *&res, const DTArg *arg, TAnalysis *&resAnal, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg, typename TAnalysis>
void ewUnaryMat(UnaryOpCode opCode, DTRes *&res, const DTArg *arg, TAnalysis *&resAnal, DCTX(ctx)) {
    EwUnaryMatAnalysis<DTRes, DTArg, TAnalysis>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT, typename... FAs>
struct EwUnaryMatAnalysis<DenseMatrix<VT>, DenseMatrix<VT>, AnalysisResult<FAs...>> {
    static void apply(UnaryOpCode opCode, DenseMatrix<VT> *&res, const DenseMatrix<VT> *arg,
                      AnalysisResult<FAs...> *&resAnal, DCTX(ctx)) {
        using anal_t = AnalysisResult<FAs...>;
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if constexpr (anal_t::template contains<AnalysisFlag::mean>) {
            resAnal->template get<AnalysisFlag::mean>() = 0;
        }
        if constexpr (anal_t::template contains<AnalysisFlag::max>) {
            using AVT = anal_t::template TypeOf<AnalysisFlag::max>;
            resAnal->template get<AnalysisFlag::max>() = std::numeric_limits<AVT>::lowest();
        }
        if constexpr (anal_t::template contains<AnalysisFlag::min>) {
            using AVT = anal_t::template TypeOf<AnalysisFlag::min>;
            resAnal->template get<AnalysisFlag::min>() = std::numeric_limits<AVT>::max();
        }

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        const VT *valuesArg = arg->getValues();
        VT *valuesRes = res->getValues();

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                VT tmp = func(valuesArg[c], ctx);
                valuesRes[c] = tmp;
                if constexpr (anal_t::template contains<AnalysisFlag::mean>) {
                    resAnal->template get<AnalysisFlag::mean>() += tmp;
                }
                if constexpr (anal_t::template contains<AnalysisFlag::max>) {
                    auto &maxVal = resAnal->template get<AnalysisFlag::max>();
                    maxVal = std::max(maxVal, tmp);
                }
                if constexpr (anal_t::template contains<AnalysisFlag::min>) {
                    auto &minVal = resAnal->template get<AnalysisFlag::min>();
                    minVal = std::min(minVal, tmp);
                }
            }
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }
        if constexpr (anal_t::template contains<AnalysisFlag::mean>) {
            resAnal->template get<AnalysisFlag::mean>() /= (float)res->getNumItems();
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT, typename... FAs>
struct EwUnaryMatAnalysis<CSRMatrix<VT>, CSRMatrix<VT>, AnalysisResult<FAs...>> {
    static void apply(UnaryOpCode opCode, CSRMatrix<VT> *&res, const CSRMatrix<VT> *arg,
                      AnalysisResult<FAs...> *&resAnal, DCTX(ctx)) {
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

        if (mapsZeroToZero) {
            // Zeros in the argument are mapped to zeros in the result. We only need to process the non-zeros of the
            // argument, the colIdxs and rowOffsets stay as they are.
            if ((std::is_floating_point_v<VT> && UnaryOpCodeUtils::mapsNonZeroToNonZeroFloat(opCode)) ||
                (std::is_integral_v<VT> && UnaryOpCodeUtils::mapsNonZeroToNonZeroInt(opCode))) {
                // Non-zeros in the argument are mapped to non-zeros in the result. We don't need to check if func
                // yields a zero.
                for (size_t i = 0; i < nnzArg; i++)
                    valuesRes[i] = func(valuesArg[i], ctx);
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
                            colIdxsRes[nnzRes] = colIdxsArg[i];
                            nnzRes++;
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
                    } else {
                        valuesRes[nnzRes] = zeroRes;
                        colIdxsRes[nnzRes] = c;
                        nnzRes++;
                    }
                    c++;
                }
                while (c < numCols) {
                    valuesRes[nnzRes] = zeroRes;
                    colIdxsRes[nnzRes] = c;
                    nnzRes++;
                    c++;
                }
                rowOffsetsRes[r + 1] = nnzRes;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template <typename VT, typename... FAs> struct EwUnaryMatAnalysis<Matrix<VT>, Matrix<VT>, AnalysisResult<FAs...>> {
    static void apply(UnaryOpCode opCode, Matrix<VT> *&res, const Matrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);

        res->prepareAppend();
        for (size_t r = 0; r < numRows; ++r)
            for (size_t c = 0; c < numCols; ++c)
                res->append(r, c, func(arg->get(r, c), ctx));
        res->finishAppend();
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
