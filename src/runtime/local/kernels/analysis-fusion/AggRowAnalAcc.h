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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_AGGROWANALYSISACC_H
#define SRC_RUNTIME_LOCAL_KERNELS_AGGROWANALYSISACC_H

#include "runtime/local/kernels/analysis-fusion/AnalysisFlags.h"
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <vector>

#include <cmath>
#include <cstddef>
#include <cstring>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTArg, typename TAnalysis> struct AggRowAnalAcc {
    static void apply(AggOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTArg, typename TAnalysis>
void aggRowAnalysisAcc(AggOpCode opCode, DTRes *&res, const DTArg *arg, DCTX(ctx)) {
    AggRowAnalAcc<DTRes, DTArg, TAnalysis>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg, AnalysisFlag... Fs>
struct AggRowAnalAcc<DenseMatrix<VTRes>, DenseMatrix<VTArg>, AnalysisFlags<Fs...>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *&res, const DenseMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);

        const VTArg *valuesArg = arg->getValues();
        VTRes *valuesRes = res->getValues();

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VTRes, AccessPattern::Dense, OutputMatrix::Dense, Fs...> analAcc;
        analAcc.init(numRows, 1);

        if (opCode == AggOpCode::IDXMIN) {
            for (size_t r = 0; r < numRows; r++) {
                VTArg minVal = valuesArg[0];
                size_t minValIdx = 0;
                for (size_t c = 1; c < numCols; c++)
                    if (valuesArg[c] < minVal) {
                        minVal = valuesArg[c];
                        minValIdx = c;
                    }
                VTRes val = static_cast<VTRes>(minValIdx);
                *valuesRes = val;
                analAcc.accumulate(r, 0, val);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }
            analAcc.writeResult(res);
        } else if (opCode == AggOpCode::IDXMAX) {
            for (size_t r = 0; r < numRows; r++) {
                VTArg maxVal = valuesArg[0];
                size_t maxValIdx = 0;
                for (size_t c = 1; c < numCols; c++)
                    if (valuesArg[c] > maxVal) {
                        maxVal = valuesArg[c];
                        maxValIdx = c;
                    }
                VTRes val = static_cast<VTRes>(maxValIdx);
                *valuesRes = val;
                analAcc.accumulate(r, 0, val);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }
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

            for (size_t r = 0; r < numRows; r++) {
                VTRes agg = static_cast<VTRes>(*valuesArg);
                for (size_t c = 1; c < numCols; c++) {
                    agg = func(agg, static_cast<VTRes>(valuesArg[c]), ctx);
                }
                VTRes val = static_cast<VTRes>(agg);
                *valuesRes = val;
                if (AggOpCodeUtils::isPureBinaryReduction(opCode))
                    analAcc.accumulate(r, 0, val);
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }

            if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
                analAcc.writeResult(res);
                return;
            }

            // The op-code is either MEAN or STDDEV or VAR
            valuesRes = res->getValues();
            // valuesArg = arg->getValues();
            for (size_t r = 0; r < numRows; r++) {
                VTRes val = (*valuesRes) / numCols;
                *valuesRes = val;
                if (opCode == AggOpCode::MEAN)
                    analAcc.accumulate(r, 0, val);

                valuesRes += res->getRowSkip();
            }

            if (opCode == AggOpCode::MEAN) {
                analAcc.writeResult(res);
                return;
            }

            // else op-code is STDDEV or VAR

            // Create a temporary matrix to store the resulting standard
            // deviations for each row
            auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, true);
            VTRes *valuesT = tmp->getValues();
            valuesArg = arg->getValues();
            valuesRes = res->getValues();
            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++) {
                    VTRes val = static_cast<VTRes>(valuesArg[c]) - (*valuesRes);
                    valuesT[r] = valuesT[r] + val * val;
                }
                valuesArg += arg->getRowSkip();
                valuesRes += res->getRowSkip();
            }
            valuesRes = res->getValues();
            for (size_t r = 0; r < numRows; r++) {
                valuesT[r] /= numCols;
                if (opCode == AggOpCode::STDDEV)
                    *valuesRes = sqrt(valuesT[r]);
                else
                    *valuesRes = valuesT[r];
                analAcc.accumulate(r, 0, *valuesRes);
                valuesRes += res->getRowSkip();
            }
            analAcc.writeResult(res);
            DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);
        }
    }
};

// ----------------------------------------------------------------------------
// DenseMatrix <- CSRMatrix
// ----------------------------------------------------------------------------

#define MAKE_CASE(aggOpCode, binOpCode)                                                                                \
    case aggOpCode:                                                                                                    \
        for (size_t r = 0; r < numRows; r++) {                                                                         \
            const size_t numNonZeros = arg->getNumNonZeros(r);                                                         \
            const bool includeExtraZero = !isSparseSafe && numNonZeros < numCols;                                      \
            *valuesRes = aggAllArray<EwBinarySca<binOpCode, VTRes, VTArg, VTArg>, VTRes>(                              \
                arg->getValues(r), numNonZeros, includeExtraZero, neutral, ctx);                                       \
            analAcc.accumulate(r, 0, *valuesRes);                                                                      \
            valuesRes += rowSkipRes;                                                                                   \
        }                                                                                                              \
        break;

template <typename VTRes, typename VTArg, AnalysisFlag... Fs>
struct AggRowAnalAcc<DenseMatrix<VTRes>, CSRMatrix<VTArg>, AnalysisFlags<Fs...>> {
    static void apply(AggOpCode opCode, DenseMatrix<VTRes> *&res, const CSRMatrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numCols = arg->getNumCols();
        const size_t numRows = arg->getNumRows();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);

        VTRes *valuesRes = res->getValues();
        size_t rowSkipRes = res->getRowSkip();

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VTRes, AccessPattern::Dense, OutputMatrix::Dense, Fs...> analAcc;
        analAcc.init(numRows, 1);

        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
            const bool isSparseSafe = AggOpCodeUtils::isSparseSafe(opCode);
            const VTRes neutral = AggOpCodeUtils::template getNeutral<VTRes>(opCode);

            switch (opCode) {
                MAKE_CASE(AggOpCode::SUM, BinaryOpCode::ADD)
                MAKE_CASE(AggOpCode::PROD, BinaryOpCode::MUL)
                MAKE_CASE(AggOpCode::MIN, BinaryOpCode::MIN)
                MAKE_CASE(AggOpCode::MAX, BinaryOpCode::MAX)
                MAKE_CASE(AggOpCode::MEAN, BinaryOpCode::ADD)
                MAKE_CASE(AggOpCode::STDDEV, BinaryOpCode::ADD)
                MAKE_CASE(AggOpCode::VAR, BinaryOpCode::ADD)
            default:
                throw std::runtime_error("unsupported AggOpCode");
            }
            analAcc.writeResult(res);
        } else { // The op-code is either MEAN or STDDEV or VAR
            // get sum for each row
            size_t ctr = 0;
            const VTRes neutral = VTRes(0);
            const bool isSparseSafe = true;
            auto tmp = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, true);
            VTRes *valuesT = tmp->getValues();
            for (size_t r = 0; r < numRows; r++) {
                const size_t numNonZeros = arg->getNumNonZeros(r);
                const bool includeExtraZero = !isSparseSafe && numNonZeros < numCols;
                *valuesRes = aggAllArray<EwBinarySca<BinaryOpCode::ADD, VTRes, VTArg, VTArg>, VTRes>(
                    arg->getValues(r), numNonZeros, includeExtraZero, neutral, ctx);
                const VTArg *valuesArg = arg->getValues(0);
                *valuesRes = *valuesRes / numCols;
                if (opCode != AggOpCode::MEAN) {
                    for (size_t i = ctr; i < ctr + numNonZeros; i++) {
                        VTRes val = static_cast<VTRes>((valuesArg[i])) - (*valuesRes);
                        valuesT[r] = valuesT[r] + val * val;
                    }

                    ctr += numNonZeros;
                    valuesT[r] += (numCols - numNonZeros) * (*valuesRes) * (*valuesRes);
                    valuesT[r] /= numCols;
                    if (opCode == AggOpCode::STDDEV)
                        *valuesRes = sqrt(valuesT[r]);
                    else
                        *valuesRes = valuesT[r];
                }
                analAcc.accumulate(r, 0, *valuesRes);
                valuesRes += rowSkipRes;
            }
            valuesRes = res->getValues();
            analAcc.writeResult(res);

            DataObjectFactory::destroy<DenseMatrix<VTRes>>(tmp);
        }
    }
};

#undef MAKE_CASE

// ----------------------------------------------------------------------------
// Matrix <- Matrix
// ----------------------------------------------------------------------------

template <typename VTRes, typename VTArg, AnalysisFlag... Fs>
struct AggRowAnalAcc<Matrix<VTRes>, Matrix<VTArg>, AnalysisFlags<Fs...>> {
    static void apply(AggOpCode opCode, Matrix<VTRes> *&res, const Matrix<VTArg> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(numRows, 1, false);

        // initialize runtime data properties. They should be compiled away if unused.
        AnalAcc<VTRes, AccessPattern::Dense, OutputMatrix::Matrix, Fs...> analAcc;
        analAcc.init(numRows, 1);

        if (opCode == AggOpCode::IDXMIN) {
            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r) {
                VTArg minVal = arg->get(r, 0);
                size_t minValIdx = 0;
                for (size_t c = 1; c < numCols; ++c) {
                    VTArg argVal = arg->get(r, c);
                    if (argVal < minVal) {
                        minVal = argVal;
                        minValIdx = c;
                    }
                }
                const VTRes val = static_cast<VTRes>(minValIdx);
                analAcc.accumulate(r, 0, val);
                res->append(r, 0, val);
            }
            res->finishAppend();
            analAcc.writeResult(res);
        } else if (opCode == AggOpCode::IDXMAX) {
            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r) {
                VTArg maxVal = arg->get(r, 0);
                size_t maxValIdx = 0;
                for (size_t c = 1; c < numCols; ++c) {
                    VTArg argVal = arg->get(r, c);
                    if (argVal > maxVal) {
                        maxVal = argVal;
                        maxValIdx = c;
                    }
                }
                const VTRes val = static_cast<VTRes>(maxValIdx);
                analAcc.accumulate(r, 0, val);
                res->append(r, 0, static_cast<VTRes>(val));
            }
            res->finishAppend();
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
            for (size_t r = 0; r < numRows; ++r) {
                VTRes agg = static_cast<VTRes>(arg->get(r, 0));
                for (size_t c = 1; c < numCols; ++c)
                    agg = func(agg, static_cast<VTRes>(arg->get(r, c)), ctx);
                const VTRes val = static_cast<VTRes>(agg);
                res->append(r, 0, static_cast<VTRes>(agg));
                if (AggOpCodeUtils::isPureBinaryReduction(opCode))
                    analAcc.accumulate(r, 0, val);
            }
            res->finishAppend();

            if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {
                analAcc.writeResult(res);
                return;
            }

            // The op-code is either MEAN or STDDEV or VAR
            for (size_t r = 0; r < numRows; ++r) {
                const VTRes val = res->get(r, 0) / numCols;
                res->set(r, 0, val);
                if (opCode == AggOpCode::MEAN)
                    analAcc.accumulate(r, 0, val);
            }

            if (opCode == AggOpCode::MEAN) {
                analAcc.writeResult(res);
                return;
            }

            // else op-code is STDDEV or VAR

            // Create a temporary matrix to store the resulting standard
            // deviations for each row
            std::vector<VTRes> tmp(numRows);

            for (size_t r = 0; r < numRows; ++r) {
                for (size_t c = 0; c < numCols; ++c) {
                    VTRes val = static_cast<VTRes>(arg->get(r, c)) - res->get(r, 0);
                    tmp[r] += val * val;
                }
            }

            res->prepareAppend();
            for (size_t r = 0; r < numRows; ++r) {
                tmp[r] /= numCols;
                VTRes val = tmp[r];
                if (opCode == AggOpCode::STDDEV)
                    val = sqrt(val);
                res->append(r, 0, val);
                analAcc.accumulate(r, 0, val);
            }
            analAcc.writeResult(res);
            res->finishAppend();
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_AGGROW_H