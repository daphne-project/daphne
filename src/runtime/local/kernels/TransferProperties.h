/*
 * Copyright 2025 The DAPHNE Consortium
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

#include <ir/daphneir/DataPropertyTypes.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DT> struct TransferProperties {
    static void apply(const DT *arg, double sparsity, int64_t symmetric, int64_t mncSketchId, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DT> void transferProperties(const DT *arg, double sparsity, int64_t symmetric, int64_t mncSketchId, DCTX(ctx)) {
    TransferProperties<DT>::apply(arg, sparsity, symmetric, mncSketchId, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct TransferProperties<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, double sparsity, int64_t symmetric, int64_t mncSketchId, DCTX(ctx)) {
        const_cast<DenseMatrix<VT> *>(arg)->sparsity = sparsity;
        const_cast<DenseMatrix<VT> *>(arg)->symmetric = static_cast<BoolOrUnknown>(symmetric);
        const_cast<DenseMatrix<VT> *>(arg)->mncSketchId = mncSketchId;
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct TransferProperties<CSRMatrix<VT>> {
    static void apply(const CSRMatrix<VT> *arg, double sparsity, int64_t symmetric, int64_t mncSketchId, DCTX(ctx)) {
        const_cast<CSRMatrix<VT> *>(arg)->sparsity = sparsity;
        const_cast<CSRMatrix<VT> *>(arg)->symmetric = static_cast<BoolOrUnknown>(symmetric);
        const_cast<CSRMatrix<VT> *>(arg)->mncSketchId = mncSketchId;
    }
};