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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CTABLEAnalAcc_H
#define SRC_RUNTIME_LOCAL_KERNELS_CTABLEAnalAcc_H

#include "runtime/local/kernels/analysis-fusion/EwUnaryMatAnalysis.h"
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <cstdint>

template <typename VT, OutputMatrix OM, AnalysisFlag... Fs> struct CTableAnalAccumulator {

    // compute the number of elements once
    size_t numRows;
    size_t numCols;

    using flags = AnalysisFlags<Fs...>;

    static_assert(!(flags::template contains<AnalysisFlag::numDistinct> &&
                    flags::template contains<AnalysisFlag::numDistinctApprox>),
                  "Cannot use both numDistinct and numDistinctApprox simultaneously");

    // no_unique_address ensures 0 byte is used if this field is not used (std::monostate)
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::mean>, double, std::monostate>
        sumAcc{};
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::min>, VT, std::monostate> minAcc{};

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::max>, VT, std::monostate> maxAcc{};

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::sparsity>, size_t, std::monostate>
        numNZ{};

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::numDistinct>,
                                             std::unordered_set<VT>, std::monostate> distinct{};

    // K Value
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::numDistinctApprox>, int64_t,
                                             std::monostate> seed;
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::numDistinctApprox>,
                                             UniqueBoundedSet<uint32_t>, std::monostate> uBSet;
    // Dense: queue of values only
    using DenseSymVec = std::vector<std::queue<VT>>;
    // Sparse: queue of (col_index, value) pairs
    using SparseSymVec = std::vector<std::queue<std::pair<size_t, VT>>>;
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::symmetry>,
                                             std::conditional_t<OM == OutputMatrix::Sparse, SparseSymVec, DenseSymVec>,
                                             std::monostate> symVec{};

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::symmetry>, bool, std::monostate>
        isSymmetric;

    CTableAnalAccumulator()
        requires(flags::template contains<AnalysisFlag::numDistinctApprox>)
        : uBSet(KMV_K) {}
    CTableAnalAccumulator()
        requires(!flags::template contains<AnalysisFlag::numDistinctApprox>)
    = default;

    // accumulate order-independent statistics
    [[gnu::always_inline]] inline void accumulateScalar(VT before, VT weight) {
        // mean is computed by first summing up all values. It is possible to overflow.
        VT now = before + weight;
        if constexpr (flags::template contains<AnalysisFlag::mean>)
            sumAcc += weight;
        if constexpr (flags::template contains<AnalysisFlag::max>)
            maxAcc = std::max(maxAcc, now);
        if constexpr (flags::template contains<AnalysisFlag::sparsity>)
            numNZ += (before == static_cast<VT>(0));
    }

    [[gnu::always_inline]] inline void accumulateEnd(size_t r, size_t c, VT val) {
        if constexpr (flags::template contains<AnalysisFlag::min>)
            minAcc = std::min(minAcc, val);
        if constexpr (flags::template contains<AnalysisFlag::numDistinct>)
            distinct.insert(val);
        // Same as the NumDistinctApprox kernel, use K Minimum Values with MurmurHash.
        if constexpr (flags::template contains<AnalysisFlag::numDistinctApprox>) {
            uint32_t hashedValueOut = 0;
            MurmurHash3_x86_32(&val, sizeof(VT), seed, &hashedValueOut);
            uBSet.push(hashedValueOut);
        }

        /*
            For the upper left half, push the values into queues
            For the lower right half, pop the queues to check if a corresponding value was seen.
        */

        // all values are pushed in order
        // we keep track of the position because zeros are skipped.
        if constexpr (flags::template contains<AnalysisFlag::symmetry>) {
            if constexpr (OM == OutputMatrix::Sparse) {
                if (val == static_cast<VT>(0))
                    return; // Ignore resulting zeros like the other zeroes that were (implicitly) skipped.
                if (isSymmetric) {
                    if (r < c) {
                        symVec[c].push({r, val});
                    }
                    if (r > c) {
                        if (symVec[r].empty()) {
                            isSymmetric = false;
                        } else {
                            auto [ci, v] = symVec[r].front();
                            if (ci == c) {
                                // a value at a symmetric position was found
                                isSymmetric = v == val;
                                symVec[r].pop();
                            } else {
                                // an entry at a symmetric position was not pushed
                                isSymmetric = false;
                            }
                        }
                    }
                }
            } else {
                if (isSymmetric) {
                    if (r < c) {
                        symVec[c].push(val);
                    }
                    if (r > c) {
                        isSymmetric = symVec[r].front() == val;
                        symVec[r].pop();
                    }
                }
            }
        }
    }

    [[gnu::always_inline]] inline void init(size_t _numRows, size_t _numCols) {
        // There is no statistics to analyze in an empty matrix
        numRows = _numRows;
        numCols = _numCols;
        const size_t numElem = numRows * numCols;
        if (numElem == 0) {
            return;
        };
        // if constexpr (flags::template contains<AnalysisFlag::sparsity>)
        //     numNZ = numRows * numCols;

        if constexpr (flags::template contains<AnalysisFlag::min>)
            minAcc = std::numeric_limits<VT>::max();
        if constexpr (flags::template contains<AnalysisFlag::max>)
            maxAcc = static_cast<VT>(0);
        if constexpr (flags::template contains<AnalysisFlag::numDistinctApprox>) {
            seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            uBSet = UniqueBoundedSet<uint32_t>(KMV_K);
        }
        if constexpr (flags::template contains<AnalysisFlag::symmetry>) {
            bool isSquare = numRows == numCols;
            isSymmetric = isSquare;
            symVec.resize(isSquare ? numRows : 0);
        }
    }

// List of flags other than symmetricity that needs to be computed at the end.
#define END_FLAGS_NOT_SYM AnalysisFlag::min, AnalysisFlag::numDistinct, AnalysisFlag::numDistinctApprox
    // Write the result back to the metadata of the resulting matrix
    [[gnu::always_inline]] inline void writeResult(Matrix<VT> *res) {
        // There is no statistics to analyze in an empty matrix
        const auto KMV_K = 1024;
        const size_t numElem = numRows * numCols;
        if (numElem == 0)
            return;

        if constexpr (flags::template contains<AnalysisFlag::sparsity>)
            res->sparsity = numNZ / (double)(numElem);

        if constexpr (flags::template contains<AnalysisFlag::mean>)
            res->mean = sumAcc / (double)(numElem);

        if constexpr (flags::template contains<AnalysisFlag::max>)
            res->max = maxAcc;

        // All of this code will not execute if none of the relevant flags are enabled.
        if constexpr (OM == OutputMatrix::Dense) {
            const auto *denseRes = static_cast<DenseMatrix<VT> *>(res);
            const VT *values = denseRes->getValues();
            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++) {
                    const VT val = values[c];
                    accumulateEnd(r, c, val);
                    if constexpr (!flags::template containsAny<END_FLAGS_NOT_SYM> &&
                                  flags::template contains<AnalysisFlag::symmetry>)
                        if (!isSymmetric)
                            break;
                }
                if constexpr (!flags::template containsAny<END_FLAGS_NOT_SYM> &&
                              flags::template contains<AnalysisFlag::symmetry>)
                    if (!isSymmetric)
                        break;
                values += denseRes->getRowSkip();
            }
            if constexpr (flags::template contains<AnalysisFlag::symmetry>)
                res->symmetric = isSymmetric ? BoolOrUnknown::True : BoolOrUnknown::False;
        } else if constexpr (OM == OutputMatrix::Matrix) {
            for (size_t r = 0; r < numRows; r++) {
                for (size_t c = 0; c < numCols; c++) {
                    const VT val = res->get(r, c);
                    accumulateEnd(r, c, val);
                    if constexpr (!flags::template containsAny<END_FLAGS_NOT_SYM> &&
                                  flags::template contains<AnalysisFlag::symmetry>)
                        if (!isSymmetric)
                            break;
                }
                if constexpr (!flags::template containsAny<END_FLAGS_NOT_SYM> &&
                              flags::template contains<AnalysisFlag::symmetry>)
                    if (!isSymmetric)
                        break;
            }
            if constexpr (flags::template contains<AnalysisFlag::symmetry>)

                res->symmetric = isSymmetric ? BoolOrUnknown::True : BoolOrUnknown::False;
        } else if constexpr (OM == OutputMatrix::Sparse) {
            const auto *csrRes = static_cast<CSRMatrix<VT> *>(res);

            const size_t *colIdxs = csrRes->getColIdxs();
            const size_t *rowOffsets = csrRes->getRowOffsets();
            const VT *values = csrRes->getValues();

            for (size_t r = 0; r < numRows; r++) {
                for (size_t i = rowOffsets[r]; i < rowOffsets[r + 1]; i++) {
                    if constexpr (flags::template contains<AnalysisFlag::symmetry>)
                        if (!isSymmetric)
                            break;
                    const auto val = values[i];
                    const size_t c = colIdxs[i];
                    accumulateEnd(r, c, val);
                    if constexpr (!flags::template containsAny<END_FLAGS_NOT_SYM> &&
                                  flags::template contains<AnalysisFlag::symmetry>)
                        if (!isSymmetric)
                            break;
                }
                if constexpr (!flags::template containsAny<END_FLAGS_NOT_SYM> &&
                              flags::template contains<AnalysisFlag::symmetry>)
                    if (!isSymmetric)
                        break;
            }
            if constexpr (flags::template contains<AnalysisFlag::symmetry>) {
                if (isSymmetric) {
                    for (auto &q : symVec) {
                        if (!q.empty()) {
                            isSymmetric = false;
                            break;
                        }
                    }
                    res->symmetric = isSymmetric ? BoolOrUnknown::True : BoolOrUnknown::False;
                } else {
                    res->symmetric = BoolOrUnknown::False;
                }
            }
            if (csrRes->getNumNonZeros() < numElem) {
                if constexpr (flags::template contains<AnalysisFlag::min>)
                    minAcc = std::min(minAcc, static_cast<VT>(0));
                if constexpr (flags::template contains<AnalysisFlag::numDistinct>)
                    distinct.insert(static_cast<VT>(0));
                if constexpr (flags::template contains<AnalysisFlag::numDistinctApprox>) {
                    uint32_t hashedValueOut = 0;
                    VT zero = static_cast<VT>(0);
                    MurmurHash3_x86_32(&zero, sizeof(VT), seed, &hashedValueOut);
                    uBSet.push(hashedValueOut);
                }
            }
        }

        if constexpr (flags::template contains<AnalysisFlag::numDistinct>)
            res->numDistinct = distinct.size();
        if constexpr (flags::template contains<AnalysisFlag::numDistinctApprox>) {
            if (uBSet.size() < KMV_K) {
                res->numDistinct = uBSet.size();
            } else {
                size_t kMinVal = uBSet.top();
                const size_t maxVal = std::numeric_limits<uint32_t>::max();
                double kMinValNormed = static_cast<double>(kMinVal) / static_cast<double>(maxVal);
                res->numDistinct = static_cast<size_t>(static_cast<double>(KMV_K - 1) / kMinValNormed);
            }
        }
        if constexpr (flags::template contains<AnalysisFlag::min>)
            res->min = minAcc;
    };
};
#undef END_FLAGS_NOT_SYM

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs, class VTWeight, typename TAnalysis> struct CTableAnalAcc {
    static void apply(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, VTWeight weight, int64_t resNumRows,
                      int64_t resNumCols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs, class VTWeight, typename TAnalysis>
void ctableAnalAcc(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, VTWeight weight, int64_t resNumRows,
                   int64_t resNumCols, DCTX(ctx)) {
    CTableAnalAcc<DTRes, DTLhs, DTRhs, VTWeight, TAnalysis>::apply(res, lhs, rhs, weight, resNumRows, resNumCols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTCoord, class VTWeight, AnalysisFlag... Fs>
struct CTableAnalAcc<DenseMatrix<VTWeight>, DenseMatrix<VTCoord>, DenseMatrix<VTCoord>, VTWeight,
                     AnalysisFlags<Fs...>> {
    static void apply(DenseMatrix<VTWeight> *&res, const DenseMatrix<VTCoord> *lhs, const DenseMatrix<VTCoord> *rhs,
                      VTWeight weight, int64_t resNumRows, int64_t resNumCols, DCTX(ctx)) {
        const size_t lhsNumRows = lhs->getNumRows();
        const size_t lhsNumCols = lhs->getNumCols();
        const size_t rhsNumRows = rhs->getNumRows();
        const size_t rhsNumCols = rhs->getNumCols();

        auto lhsVals = lhs->getValues();
        auto rhsVals = rhs->getValues();

        if ((lhsNumCols != 1) || (rhsNumCols != 1))
            throw std::runtime_error("ctableAnalAcc: lhs and rhs must have only one column");
        if (lhsNumRows != rhsNumRows)
            throw std::runtime_error("ctableAnalAcc: lhs and rhs must have the same number of rows");

        const bool isResNumRowsFromLhs = resNumRows < 0;
        const bool isResNumColsFromRhs = resNumCols < 0;
        if (res == nullptr) {
            if (isResNumRowsFromLhs)
                resNumRows = *std::max_element(lhsVals, &lhsVals[lhsNumRows]) + 1;
            if (isResNumColsFromRhs)
                resNumCols = *std::max_element(rhsVals, &rhsVals[rhsNumRows]) + 1;
            res = DataObjectFactory::create<DenseMatrix<VTWeight>>(resNumRows, resNumCols, true);
        }
        // initialize runtime data properties. They should be compiled away if unused.
        CTableAnalAccumulator<VTWeight, OutputMatrix::Dense, Fs...> analAcc;
        analAcc.init(resNumRows, resNumCols);

        // res[i, j] = |{ k | lhs[k] = i and rhs[k] = j, 0 ≤ k ≤ n-1 }|.
        auto resVals = res->getValues();
        const size_t resRowSkip = res->getRowSkip();
        if (isResNumRowsFromLhs && isResNumColsFromRhs) {
            // The number of rows and columns of the result were derived from
            // the left-hand-side and right-hand-side arguments. Thus, all
            // positions are in-bounds.
            for (size_t i = 0; i < lhsNumRows; i++) {
                const ssize_t r = lhsVals[i];
                const ssize_t c = rhsVals[i];
                const auto before = resVals[static_cast<size_t>(r * resRowSkip + c)];
                analAcc.accumulateScalar(before, weight);
                resVals[static_cast<size_t>(r * resRowSkip + c)] += weight;
            }
        } else {
            // The number of rows and/or columns of the result were given by the
            // caller. Thus, positions might be out-of-bounds. If that is the
            // case, they shall be silently ignored.
            for (size_t i = 0; i < lhsNumRows; i++) {
                const ssize_t r = lhsVals[i];
                const ssize_t c = rhsVals[i];
                if (r < resNumRows && c < resNumCols) {
                    const auto before = resVals[static_cast<size_t>(r * resRowSkip + c)];
                    analAcc.accumulateScalar(before, weight);
                    resVals[static_cast<size_t>(r * resRowSkip + c)] += weight;
                }
            }
        }
        analAcc.writeResult(res);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------
template <typename VTCoord, class VTWeight, AnalysisFlag... Fs>
struct CTableAnalAcc<CSRMatrix<VTWeight>, DenseMatrix<VTCoord>, DenseMatrix<VTCoord>, VTWeight, AnalysisFlags<Fs...>> {
    static void apply(CSRMatrix<VTWeight> *&res, const DenseMatrix<VTCoord> *lhs, const DenseMatrix<VTCoord> *rhs,
                      VTWeight weight, int64_t resNumRows, int64_t resNumCols, DCTX(ctx)) {
        const size_t lhsNumRows = lhs->getNumRows();
        const size_t lhsNumCols = lhs->getNumCols();
        const size_t rhsNumRows = rhs->getNumRows();
        const size_t rhsNumCols = rhs->getNumCols();

        auto lhsVals = lhs->getValues();
        auto rhsVals = rhs->getValues();

        if ((lhsNumCols != 1) || (rhsNumCols != 1))
            throw std::runtime_error("ctableAnalAcc: lhs and rhs must have only one column");
        if (lhsNumRows != rhsNumRows)
            throw std::runtime_error("ctableAnalAcc: lhs and rhs must have the same number of rows");

        const bool isResNumRowsFromLhs = resNumRows < 0;
        const bool isResNumColsFromRhs = resNumCols < 0;
        if (res == nullptr) {
            if (isResNumRowsFromLhs)
                resNumRows = *std::max_element(lhsVals, &lhsVals[lhsNumRows]) + 1;
            if (isResNumColsFromRhs)
                resNumCols = *std::max_element(rhsVals, &rhsVals[rhsNumRows]) + 1;
            res = DataObjectFactory::create<CSRMatrix<VTWeight>>(
                resNumRows, resNumCols, std::min(static_cast<ssize_t>(lhsNumRows), resNumRows * resNumCols), true);
        }

        // initialize runtime data properties. They should be compiled away if unused.
        CTableAnalAccumulator<VTWeight, OutputMatrix::Sparse, Fs...> analAcc;
        analAcc.init(resNumRows, resNumCols);

        if (isResNumRowsFromLhs && isResNumColsFromRhs) {
            // The number of rows and columns of the result were derived from
            // the left-hand-side and right-hand-side arguments. Thus, all
            // positions are in-bounds.
            for (size_t i = 0; i < lhsNumRows; i++) {
                const ssize_t r = lhsVals[i];
                const ssize_t c = rhsVals[i];
                const auto before = res->get(r, c);
                analAcc.accumulateScalar(before, weight);
                res->set(r, c, before + weight);
            }
        } else {
            // The number of rows and/or columns of the result were given by the
            // caller. Thus, positions might be out-of-bounds. If that is the
            // case, they shall be silently ignored.
            for (size_t i = 0; i < lhsNumRows; i++) {
                const ssize_t r = lhsVals[i];
                const ssize_t c = rhsVals[i];
                if (r < resNumRows && c < resNumCols) {
                    const auto before = res->get(r, c);
                    analAcc.accumulateScalar(before, weight);
                    res->set(r, c, before + weight);
                }
            }
        }
        analAcc.writeResult(res);
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, Matrix
// ----------------------------------------------------------------------------

template <typename VTCoord, class VTWeight, AnalysisFlag... Fs>
struct CTableAnalAcc<Matrix<VTWeight>, Matrix<VTCoord>, Matrix<VTCoord>, VTWeight, AnalysisFlags<Fs...>> {
    static void apply(Matrix<VTWeight> *&res, const Matrix<VTCoord> *lhs, const Matrix<VTCoord> *rhs, VTWeight weight,
                      int64_t resNumRows, int64_t resNumCols, DCTX(ctx)) {
        const size_t lhsNumRows = lhs->getNumRows();

        if ((lhs->getNumCols() != 1) || (rhs->getNumCols() != 1))
            throw std::runtime_error("ctableAnalAcc: lhs and rhs must have only one column");
        if (lhsNumRows != rhs->getNumRows())
            throw std::runtime_error("ctableAnalAcc: lhs and rhs must have the same number of rows");

        const bool isResNumRowsFromLhs = resNumRows < 0;
        const bool isResNumColsFromRhs = resNumCols < 0;

        if (res == nullptr) {
            auto getMaxVal = [](const Matrix<VTCoord> *mat) {
                const size_t numRows = mat->getNumRows();
                VTCoord maxVal = mat->get(0, 0);
                for (size_t r = 1; r < numRows; ++r) {
                    VTCoord val = mat->get(r, 0);
                    if (val > maxVal)
                        maxVal = val;
                }
                return maxVal;
            };

            if (isResNumRowsFromLhs)
                resNumRows = static_cast<int64_t>(getMaxVal(lhs)) + 1;
            if (isResNumColsFromRhs)
                resNumCols = static_cast<int64_t>(getMaxVal(rhs)) + 1;
            res = DataObjectFactory::create<DenseMatrix<VTWeight>>(resNumRows, resNumCols, true);
        }
        // initialize runtime data properties. They should be compiled away if unused.
        CTableAnalAccumulator<VTWeight, OutputMatrix::Matrix, Fs...> analAcc;
        analAcc.init(resNumRows, resNumCols);

        // res[i, j] = |{ k | lhs[k] = i and rhs[k] = j, 0 ≤ k ≤ n-1 }|.
        if (isResNumRowsFromLhs && isResNumColsFromRhs) {
            // The number of rows and columns of the result were derived from
            // the left-hand-side and right-hand-side arguments. Thus, all
            // positions are in-bounds.
            for (size_t i = 0; i < lhsNumRows; ++i) {
                const ssize_t r = lhs->get(i, 0);
                const ssize_t c = rhs->get(i, 0);
                const auto before = res->get(r, c);
                analAcc.accumulateScalar(before, weight);
                res->set(r, c, before + weight);
            }
        } else {
            // The number of rows and/or columns of the result were given by the
            // caller. Thus, positions might be out-of-bounds. If that is the
            // case, they shall be silently ignored.
            for (size_t i = 0; i < lhsNumRows; ++i) {
                const ssize_t r = lhs->get(i, 0);
                const ssize_t c = rhs->get(i, 0);
                if (r < resNumRows && c < resNumCols) {
                    const auto before = res->get(r, c);
                    analAcc.accumulateScalar(before, weight);
                    res->set(r, c, before + weight);
                }
            }
        }
        analAcc.writeResult(res);
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_CTABLEAnalAcc_H
