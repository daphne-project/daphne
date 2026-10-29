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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_NaiveAnalysis_H
#define SRC_RUNTIME_LOCAL_KERNELS_NaiveAnalysis_H

#include "ir/daphneir/DataPropertyTypes.h"
#include "runtime/local/kernels/analysis-fusion/AnalysisFlags.h"
#include <limits>
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

template <class DTArg> struct NaiveAnalysis {
    static void apply(DTArg *arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> void naiveAnalysis(DTArg *arg, DCTX(ctx)) { NaiveAnalysis<DTArg>::apply(arg, ctx); }

// One function to call on every possible Matrix type
template <typename DT>
    requires std::derived_from<DT, Matrix<typename DT::VT>> // everything that is subtype or itself of Matrix<VT>
struct NaiveAnalysis<DT> {
    static void apply(DT *arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        using VT = typename DT::VT;

        VT minAcc = std::numeric_limits<VT>::max();
        VT maxAcc = std::numeric_limits<VT>::lowest();
        VT sumAcc = static_cast<VT>(0);
        size_t nnz = 0;
        std::unordered_set<VT> distinct;
        bool isSymmetric = numRows == numCols;

        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                VT val = arg->get(r, c);
                minAcc = std::min(minAcc, val);
                maxAcc = std::max(maxAcc, val);
                nnz += (val != static_cast<VT>(0));
                sumAcc += val;
                distinct.insert(val);
                if (isSymmetric)
                    isSymmetric = val == arg->get(c, r);
            }
        }
        if (numRows * numCols == 0)
            return;
        arg->min = minAcc;
        arg->max = maxAcc;
        arg->sparsity = (double)(nnz) / (numRows * numCols);
        arg->mean = (double)(sumAcc) / (numRows * numCols);
        arg->symmetric = isSymmetric ? mlir::daphne::BoolOrUnknown::True : mlir::daphne::BoolOrUnknown::False;
        arg->numDistinct = distinct.size();
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_NaiveAnalysis_H
