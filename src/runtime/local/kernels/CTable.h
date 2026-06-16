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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_CTABLE_H
#define SRC_RUNTIME_LOCAL_KERNELS_CTABLE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs, class VTWeight> struct CTable {
    static void apply(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, VTWeight weight, int64_t resNumRows,
                      int64_t resNumCols, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes, class DTLhs, class DTRhs, class VTWeight>
void ctable(DTRes *&res, const DTLhs *lhs, const DTRhs *rhs, VTWeight weight, int64_t resNumRows, int64_t resNumCols,
            DCTX(ctx)) {
    CTable<DTRes, DTLhs, DTRhs, VTWeight>::apply(res, lhs, rhs, weight, resNumRows, resNumCols, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------

template <typename VTCoord, class VTWeight>
struct CTable<DenseMatrix<VTWeight>, DenseMatrix<VTCoord>, DenseMatrix<VTCoord>, VTWeight> {
    static void apply(DenseMatrix<VTWeight> *&res, const DenseMatrix<VTCoord> *lhs, const DenseMatrix<VTCoord> *rhs,
                      VTWeight weight, int64_t resNumRows, int64_t resNumCols, DCTX(ctx)) {
        const size_t lhsNumRows = lhs->getNumRows();
        const size_t lhsNumCols = lhs->getNumCols();
        const size_t rhsNumRows = rhs->getNumRows();
        const size_t rhsNumCols = rhs->getNumCols();

        auto lhsVals = lhs->getValues();
        auto rhsVals = rhs->getValues();

        if ((lhsNumCols != 1) || (rhsNumCols != 1))
            throw std::runtime_error("ctable: lhs and rhs must have only one column");
        if (lhsNumRows != rhsNumRows)
            throw std::runtime_error("ctable: lhs and rhs must have the same number of rows");

        const bool isResNumRowsFromLhs = resNumRows < 0;
        const bool isResNumColsFromRhs = resNumCols < 0;
        if (res == nullptr) {
            if (isResNumRowsFromLhs)
                resNumRows = *std::max_element(lhsVals, &lhsVals[lhsNumRows]) + 1;
            if (isResNumColsFromRhs)
                resNumCols = *std::max_element(rhsVals, &rhsVals[rhsNumRows]) + 1;
            res = DataObjectFactory::create<DenseMatrix<VTWeight>>(resNumRows, resNumCols, true);
        }

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
                resVals[static_cast<size_t>(r * resRowSkip + c)] += weight;
            }
        } else {
            // The number of rows and/or columns of the result were given by the
            // caller. Thus, positions might be out-of-bounds. If that is the
            // case, they shall be silently ignored.
            for (size_t i = 0; i < lhsNumRows; i++) {
                const ssize_t r = lhsVals[i];
                const ssize_t c = rhsVals[i];
                if (r < resNumRows && c < resNumCols)
                    resVals[static_cast<size_t>(r * resRowSkip + c)] += weight;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix <- DenseMatrix, DenseMatrix
// ----------------------------------------------------------------------------
template <typename VTCoord, class VTWeight>
struct CTable<CSRMatrix<VTWeight>, DenseMatrix<VTCoord>, DenseMatrix<VTCoord>, VTWeight> {
    static void apply(CSRMatrix<VTWeight> *&res, const DenseMatrix<VTCoord> *lhs, const DenseMatrix<VTCoord> *rhs,
                      VTWeight weight, int64_t resNumRows, int64_t resNumCols, DCTX(ctx)) {
        const size_t lhsNumRows = lhs->getNumRows();
        const size_t lhsNumCols = lhs->getNumCols();
        const size_t rhsNumRows = rhs->getNumRows();
        const size_t rhsNumCols = rhs->getNumCols();

        auto lhsVals = lhs->getValues();
        auto rhsVals = rhs->getValues();

        if ((lhsNumCols != 1) || (rhsNumCols != 1))
            throw std::runtime_error("ctable: lhs and rhs must have only one column");
        if (lhsNumRows != rhsNumRows)
            throw std::runtime_error("ctable: lhs and rhs must have the same number of rows");

        const bool isResNumRowsFromLhs = resNumRows < 0;
        const bool isResNumColsFromRhs = resNumCols < 0;
        if (res == nullptr) {
            if (isResNumRowsFromLhs)
                resNumRows = *std::max_element(lhsVals, &lhsVals[lhsNumRows]) + 1;
            if (isResNumColsFromRhs)
                resNumCols = *std::max_element(rhsVals, &rhsVals[rhsNumRows]) + 1;
            res = DataObjectFactory::create<CSRMatrix<VTWeight>>(
                resNumRows, resNumCols, std::min(static_cast<int64_t>(lhsNumRows), resNumRows * resNumCols), true);
        }

        if (isResNumRowsFromLhs && isResNumColsFromRhs) {
            // The number of rows and columns of the result were derived from
            // the left-hand-side and right-hand-side arguments. Thus, all
            // positions are in-bounds.
            for (size_t i = 0; i < lhsNumRows; i++) {
                const ssize_t r = lhsVals[i];
                const ssize_t c = rhsVals[i];
                res->set(r, c, res->get(r, c) + weight);
            }
        } else {
            // The number of rows and/or columns of the result were given by the
            // caller. Thus, positions might be out-of-bounds. If that is the
            // case, they shall be silently ignored.
            for (size_t i = 0; i < lhsNumRows; i++) {
                const ssize_t r = lhsVals[i];
                const ssize_t c = rhsVals[i];
                if (r < resNumRows && c < resNumCols)
                    res->set(r, c, res->get(r, c) + weight);
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Matrix <- Matrix, Matrix
// ----------------------------------------------------------------------------

template <typename VTCoord, class VTWeight>
struct CTable<Matrix<VTWeight>, Matrix<VTCoord>, Matrix<VTCoord>, VTWeight> {
    static void apply(Matrix<VTWeight> *&res, const Matrix<VTCoord> *lhs, const Matrix<VTCoord> *rhs, VTWeight weight,
                      int64_t resNumRows, int64_t resNumCols, DCTX(ctx)) {
        const size_t lhsNumRows = lhs->getNumRows();

        if ((lhs->getNumCols() != 1) || (rhs->getNumCols() != 1))
            throw std::runtime_error("ctable: lhs and rhs must have only one column");
        if (lhsNumRows != rhs->getNumRows())
            throw std::runtime_error("ctable: lhs and rhs must have the same number of rows");

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

        // res[i, j] = |{ k | lhs[k] = i and rhs[k] = j, 0 ≤ k ≤ n-1 }|.
        if (isResNumRowsFromLhs && isResNumColsFromRhs) {
            // The number of rows and columns of the result were derived from
            // the left-hand-side and right-hand-side arguments. Thus, all
            // positions are in-bounds.
            for (size_t i = 0; i < lhsNumRows; ++i) {
                const ssize_t r = lhs->get(i, 0);
                const ssize_t c = rhs->get(i, 0);
                res->set(r, c, res->get(r, c) + weight);
            }
        } else {
            // The number of rows and/or columns of the result were given by the
            // caller. Thus, positions might be out-of-bounds. If that is the
            // case, they shall be silently ignored.
            for (size_t i = 0; i < lhsNumRows; ++i) {
                const ssize_t r = lhs->get(i, 0);
                const ssize_t c = rhs->get(i, 0);
                if (r < resNumRows && c < resNumCols)
                    res->set(r, c, res->get(r, c) + weight);
            }
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_CTABLE_H
