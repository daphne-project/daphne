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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_AggColAnalysisAcc_H
#define SRC_RUNTIME_LOCAL_KERNELS_AggColAnalysisAcc_H

#include "runtime/local/kernels/analysis-fusion/EwUnaryMatAnalysis.h"
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <vector>

#include <cmath>
#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg, typename TAnalysis> struct AggColAnalysisAcc {
    static void apply(AggOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg, typename TAnalysis>
void aggColAnalysisAcc(AggOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) {
    AggColAnalysisAcc<DTRes, DTArg, TAnalysis>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg, AnalysisFlag... Fs>
struct AggColAnalysisAcc<DenseMatrix<VTRes>, DenseMatrix<VTArg>, AnalysisFlags<Fs...>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, false);

        const VTArg *valuesArg = arg->getValues();
        VTRes *valuesRes = res->getValues();

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VTRes, AccessPattern::Dense, OutputMatrix::Dense, Fs...> analAcc;
        analAcc.init(1, numCols);

        // TODO Merge the cases for IDXMIN and IDXMAX to avoid code duplication.
        if (opCode == AggOpCode::IDXMIN) {
            // Minimum values seen so far per column (initialize with first row
            // of argument).
            auto tmp = DataObjectFactory::create<DenseMatrix<VTArg>>(1, numCols, false);
            VTArg *valuesTmp = tmp->getValues();
            memcpy(valuesTmp, valuesArg, numCols * sizeof(VTArg));

            // Positions at which the minimum values were found (initialize with
            // zeros), stored directly in the result.
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = 0;

            // Scan over the remaining rows and update the minimum values and
            // their positions accordingly.
            valuesArg += arg->getRowSkip();
            for (size_t r = 1; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++)
                    if (valuesArg[c] < valuesTmp[c]) {
                        valuesTmp[c] = valuesArg[c];
                        valuesRes[c] = r;
                    }
                valuesArg += arg->getRowSkip();
            }

            // There is no sensible way to do analysis rather than to run it afterwards.
            for (size_t c = 0; c < numCols; c++)
                analAcc.accumulate(0, c, valuesRes[c]);
            analAcc.writeResult(res);

            // Free the temporary minimum values.
            DataObjectFactory::destroy(tmp);
        } else if (opCode == AggOpCode::IDXMAX) {
            // Maximum values seen so far per column (initialize with first row
            // of argument).
            auto tmp = DataObjectFactory::create<DenseMatrix<VTArg>>(1, numCols, false);
            VTArg *valuesTmp = tmp->getValues();
            memcpy(valuesTmp, valuesArg, numCols * sizeof(VTArg));

            // Positions at which the maximum values were found (initialize with
            // zeros), stored directly in the result.
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = 0;

            // Scan over the remaining rows and update the maximum values and
            // their positions accordingly.
            valuesArg += arg->getRowSkip();
            for (size_t r = 1; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++)
                    if (valuesArg[c] > valuesTmp[c]) {
                        valuesTmp[c] = valuesArg[c];
                        valuesRes[c] = r;
                    }
                valuesArg += arg->getRowSkip();
            }

            // There is no sensible way to do analysis rather than to run it afterwards.
            for (size_t c = 0; c < numCols; c++)
                analAcc.accumulate(0, c, valuesRes[c]);
            analAcc.writeResult(res);

            // Free the temporary maximum values.
            DataObjectFactory::destroy(tmp);
        } else {
            EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;
            if (AggOpCodeUtils::isPureBinaryReduction(opCode))
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
            else
                // TODO Setting the function pointer yields the correct result.
                // However, since MEAN and STDDEV are not sparse-safe, the
                // program does not take the same path for doing the summation,
                // and is less efficient. for MEAN and STDDDEV, we need to sum
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

            // memcpy(valuesRes, valuesArg, numCols * sizeof(VTRes));
            // Can't memcpy because we might have different result type
            for (size_t c = 0; c < numCols; c++)
                valuesRes[c] = static_cast<VTRes>(valuesArg[c]);
            for (size_t r = 1; r < numRows; r++) {
                valuesArg += arg->getRowSkip();
                for (size_t c = 0; c < numCols; c++) {
                    const VTRes val = func(valuesRes[c], static_cast<VTRes>(valuesArg[c]), ctx);
                    valuesRes[c] = val;
                }
            }

            if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
                // There is no sensible way to do analysis rather than to run it afterwards.
                for (size_t c = 0; c < numCols; c++)
                    analAcc.accumulate(0, c, valuesRes[c]);
                analAcc.writeResult(res);
                return;
            }

            // The op-code is either MEAN or STDDEV or VAR.

            for (size_t c = 0; c < numCols; c++) {
                const VTRes val = valuesRes[c] / numRows;
                valuesRes[c] = val;
                if (opCode == AggOpCode::MEAN)
                    analAcc.accumulate(0, c, val);
            }

            if (opCode == AggOpCode::MEAN) {
                analAcc.writeResult(res);
                return;
            }

            auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, true);
            VTRes *valuesT = tmp->getValues();
            valuesArg = arg->getValues();

            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++) {
                    VTRes val = static_cast<VTRes>(valuesArg[c]) - valuesRes[c];
                    valuesT[c] = valuesT[c] + val * val;
                }
                valuesArg += arg->getRowSkip();
            }

            for (size_t c = 0; c < numCols; c++) {
                VTRes val = valuesT[c] / numRows;
                if (opCode == AggOpCode::STDDEV)
                    val = sqrt(val);
                valuesT[c] = val;
                analAcc.accumulate(0, c, val);
            }

            analAcc.writeResult(res);
            // TODO We could avoid copying by returning tmp and destroying res.
            // But that might be wrong if res was not nullptr initially.
            memcpy(valuesRes, valuesT, numCols * sizeof(VTRes));
            DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------
template <typename VTRes, typename VTArg, AnalysisFlag... Fs>
struct AggColAnalysisAcc<DenseMatrix<VTRes>, CSRMatrix<VTArg>, AnalysisFlags<Fs...>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *&res, const CSRMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, true);

        VTRes *valuesRes = res->getValues();

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VTRes, AccessPattern::Dense, OutputMatrix::Dense, Fs...> analAcc;
        analAcc.init(1, numCols);

        // if (arg->getNumRows() > 1)

        EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;
        if (AggOpCodeUtils::isPureBinaryReduction(opCode))
            func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
        else
            // TODO Setting the function pointer yields the correct result.
            // However, since MEAN and STDDEV are not sparse-safe, the program
            // does not take the same path for doing the summation, and is less
            // efficient.
            // for MEAN and STDDDEV, we need to sum
            func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

        const VTArg *valuesArg = arg->getValues(0);
        const size_t *colIdxsArg = arg->getColIdxs(0);

        const size_t numNonZeros = arg->getNumNonZeros();

        if (AggOpCodeUtils::isSparseSafe(opCode)) {
            for (size_t i = 0; i < numNonZeros; i++) {
                const size_t colIdx = colIdxsArg[i];
                VTRes val = func(valuesRes[colIdx], static_cast<VTRes>(valuesArg[i]), ctx);
                valuesRes[colIdx] = val;
            }
        } else {
            size_t *hist = new size_t[numCols](); // initialized to zeros

            const size_t numNonZerosFirstRowArg = arg->getNumNonZeros(0);
            for (size_t i = 0; i < numNonZerosFirstRowArg; i++) {
                size_t colIdx = colIdxsArg[i];
                valuesRes[colIdx] = static_cast<VTRes>(valuesArg[i]);
                hist[colIdx]++;
            }

            if (arg->getNumRows() > 1) {
                for (size_t i = numNonZerosFirstRowArg; i < numNonZeros; i++) {
                    const size_t colIdx = colIdxsArg[i];
                    valuesRes[colIdx] = func(valuesRes[colIdx], static_cast<VTRes>(valuesArg[i]), ctx);
                    hist[colIdx]++;
                }
                for (size_t c = 0; c < numCols; c++) {
                    if (hist[c] < numRows)
                        valuesRes[c] = func(valuesRes[c], VTRes(0), ctx);
                }
            }
            delete[] hist;
        }

        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
            for (size_t c = 0; c < numCols; c++)
                analAcc.accumulate(0, c, valuesRes[c]);
            analAcc.writeResult(res);
            return;
        }

        // The op-code is either MEAN or STDDEV or VAR.

        for (size_t c = 0; c < numCols; c++) {
            VTRes val = valuesRes[c] / arg->getNumRows();
            valuesRes[c] = val;
            if (opCode == AggOpCode::MEAN)
                analAcc.accumulate(0, c, val);
        }
        if (opCode == AggOpCode::MEAN) {
            analAcc.writeResult(res);
            return;
        }

        auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, true);
        VTRes *valuesT = tmp->getValues();

        size_t *nnzCol = new size_t[numCols](); // initialized to zeros
        for (size_t i = 0; i < numNonZeros; i++) {
            const size_t colIdx = colIdxsArg[i];
            VTRes val = static_cast<VTRes>(valuesArg[i]) - valuesRes[colIdx];
            valuesT[colIdx] = valuesT[colIdx] + val * val;
            nnzCol[colIdx]++;
        }

        for (size_t c = 0; c < numCols; c++) {
            // Take all zeros in the column into account.
            valuesT[c] += (valuesRes[c] * valuesRes[c]) * (numRows - nnzCol[c]);
            // Finish computation of stddev.
            valuesT[c] /= numRows;
            if (opCode == AggOpCode::STDDEV)
                valuesT[c] = sqrt(valuesT[c]);
            analAcc.accumulate(0, c, valuesT[c]);
        }

        delete[] nnzCol;

        analAcc.writeResult(res);
        // TODO We could avoid copying by returning tmp and destroying res. But
        // that might be wrong if res was not nullptr initially.
        memcpy(valuesRes, valuesT, numCols * sizeof(VTRes));
        DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg, AnalysisFlag... Fs>
struct AggColAnalysisAcc<Matrix<VTRes>, Matrix<VTArg>, AnalysisFlags<Fs...>> {
    static void apply(AggOpCode opCode, Matrix<VTRes> *&res, const Matrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VTRes, AccessPattern::Dense, OutputMatrix::Matrix, Fs...> analAcc;
        analAcc.init(1, numCols);

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(1, numCols, false);

        if (opCode == AggOpCode::IDXMIN || opCode == AggOpCode::IDXMAX) {
            // Minimum/Maximum values seen so far per column (initialize with
            // first row of argument).
            std::vector<VTArg> tmp;
            tmp.reserve(numCols);
            for (size_t c = 0; c < numCols; ++c)
                tmp.emplace_back(arg->get(0, c));

            // Positions at which the minimum/maximum values were found
            // (initialize with zeros), stored directly in the result.
            res->prepareAppend();
            res->finishAppend();

            // Scan over the remaining rows and update the minimum values and
            // their positions accordingly.
            // TODO: reduce code duplication with lambda
            //       initial test seemed slower than separate loops but should
            //       be tested again
            if (opCode == AggOpCode::IDXMIN) {
                for (size_t r = 1; r < numRows; ++r) {
                    for (size_t c = 0; c < numCols; ++c) {
                        VTArg argVal = arg->get(r, c);
                        if (argVal < tmp[c]) {
                            tmp[c] = argVal;
                            res->set(0, c, r);
                        }
                    }
                }
            } else {
                for (size_t r = 1; r < numRows; ++r) {
                    for (size_t c = 0; c < numCols; ++c) {
                        VTArg argVal = arg->get(r, c);
                        if (argVal > tmp[c]) {
                            tmp[c] = argVal;
                            res->set(0, c, r);
                        }
                    }
                }
            }
            // There is no sensible way to do analysis rather than to run it afterwards.
            for (size_t c = 0; c < numCols; c++)
                analAcc.accumulate(0, c, res->get(0, c));
            analAcc.writeResult(res);

        } else {
            EwBinaryScaFuncPtr<VTRes, VTRes, VTRes> func;
            if (AggOpCodeUtils::isPureBinaryReduction(opCode))
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(opCode));
            else
                // TODO Setting the function pointer yields the correct result.
                // However, since MEAN and STDDEV are not sparse-safe, the
                // program does not take the same path for doing the summation,
                // and is less efficient. for MEAN and STDDDEV, we need to sum
                func = getEwBinaryScaFuncPtr<VTRes, VTRes, VTRes>(AggOpCodeUtils::getBinaryOpCode(AggOpCode::SUM));

            res->prepareAppend();
            for (size_t c = 0; c < numCols; ++c)
                res->append(0, c, static_cast<VTRes>(arg->get(0, c)));
            res->finishAppend();
            for (size_t r = 1; r < numRows; ++r) {
                for (size_t c = 0; c < numCols; ++c) {
                    res->set(0, c, func(res->get(0, c), static_cast<VTRes>(arg->get(r, c)), ctx));
                }
            }

            if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
                for (size_t c = 0; c < numCols; c++)
                    analAcc.accumulate(0, c, res->get(0, c));
                analAcc.writeResult(res);
                return;
            }

            // The op-code is either MEAN or STDDEV or VAR.

            for (size_t c = 0; c < numCols; ++c) {
                const VTRes val = res->get(0, c) / numRows;
                res->set(0, c, val);
                if (opCode == AggOpCode::MEAN)
                    analAcc.accumulate(0, c, val);
            }

            if (opCode == AggOpCode::MEAN) {
                analAcc.writeResult(res);
                return;
            }

            std::vector<VTRes> tmp(numCols);

            for (size_t r = 0; r < numRows; ++r) {
                for (size_t c = 0; c < numCols; ++c) {
                    VTRes val = static_cast<VTRes>(arg->get(r, c)) - res->get(0, c);
                    tmp[c] += val * val;
                }
            }

            res->prepareAppend();
            for (size_t c = 0; c < numCols; ++c) {
                tmp[c] /= numRows;
                if (opCode == AggOpCode::STDDEV)
                    tmp[c] = sqrt(tmp[c]);
                analAcc.accumulate(0, c, tmp[c]);
                res->append(0, c, tmp[c]);
            }
            analAcc.writeResult(res);
            res->finishAppend();
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_AggColAnalysisAcc_H
