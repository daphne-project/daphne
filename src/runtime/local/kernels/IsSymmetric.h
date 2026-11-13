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

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>

#include <cstddef>
#include <cstdio>

template <class DTArg> struct IsSymmetric {
    static bool apply(const DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

/**
 * @brief Checks the input matrix for symmetry. Returns false for non-square matrices.
 */
template <class DTArg> bool isSymmetric(const DTArg *arg, DCTX(ctx)) { return IsSymmetric<DTArg>::apply(arg, ctx); }

// ****************************************************************************
// (Partial) template specializations for different DataTypes
// ****************************************************************************

// ----------------------------------------------------------------------------
// bool <- DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct IsSymmetric<DenseMatrix<VT>> {
    static bool apply(const DenseMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (numRows != numCols)
            return false;

        // TODO add cache-conscious operations
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++)
            for (size_t colIdx = rowIdx + 1; colIdx < numCols; colIdx++) {
                const VT val1 = arg->get(rowIdx, colIdx);
                const VT val2 = arg->get(colIdx, rowIdx);
                if (val1 != val2)
                    return false;
            }
        return true;
    }
};

// ----------------------------------------------------------------------------
// bool <- CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct IsSymmetric<CSRMatrix<VT>> {
    static bool apply(const CSRMatrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (numRows != numCols)
            return false;

        // 0x0 and 1x1 matrices are always symmetric.
        if (numRows <= 1)
            return true;

        std::vector<size_t> positions(numRows, -1); // indexes of the column index array

        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            const VT *rowA = arg->getValues(rowIdx);
            const size_t *colIdxsA = arg->getColIdxs(rowIdx);
            const size_t numNonZerosA = arg->getNumNonZeros(rowIdx);

            for (size_t idx = 0; idx < numNonZerosA; idx++) {
                const size_t colIdxA = colIdxsA[idx];

                if (colIdxA <= rowIdx) // exit early if diagonal element or before
                    continue;

                positions[rowIdx] = idx;
                VT valA = rowA[idx];

                // B references the transposed element to compare for symmetry.
                const VT *rowB = arg->getValues(colIdxA);
                const size_t *colIdxsB = arg->getColIdxs(colIdxA);
                const size_t numNonZerosB = arg->getNumNonZeros(colIdxA);

                positions[colIdxA]++; // colIdxA is rowIdxB
                const size_t posB = positions[colIdxA];

                if (numNonZerosB <= posB) // Does the next expected element exist?
                    return false;

                const size_t colIdxB = colIdxsB[posB];
                VT valB = rowB[posB];

                if (colIdxB != rowIdx || valA != valB) // indexes or values differ, not sym
                    return false;
            }

            const size_t rowLastPos = positions[rowIdx];

            if (rowLastPos == static_cast<size_t>(-1) && numNonZerosA != 0)
                // Not all elements of this row were iterated over, not symmetric!
                return false;
        }
        return true;
    }
};

// ----------------------------------------------------------------------------
// bool <- Matrix
// ----------------------------------------------------------------------------

template <typename VT> struct IsSymmetric<Matrix<VT>> {
    static bool apply(const Matrix<VT> *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();

        if (numRows != numCols)
            return false;

        for (size_t rowIdx = 0; rowIdx < numRows; ++rowIdx)
            for (size_t colIdx = rowIdx + 1; colIdx < numCols; ++colIdx)
                if (arg->get(rowIdx, colIdx) != arg->get(colIdx, rowIdx))
                    return false;

        return true;
    }
};