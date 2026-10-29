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

#include "runtime/local/kernels/analysis-fusion/AnalysisFlags.h"
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/kernels/analysis-fusion/CTableAnalAcc.h>
#include <runtime/local/kernels/analysis-fusion/NaiveAnalysis.h>
#include <runtime/local/kernels/analysis-fusion/Util.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>
#include <type_traits>

#define TEST_NAME(opName) "CTableAnalAcc (" opName ")"
#define VALUE_TYPES int32_t, double

// Map the result data-structure types to the matching argument types and run the fused kernel against naiveAnalysis
// over a separately-built reference. We just take DenseMatrix's as arguments and cast them to a Matrix if we are
// testing for Matrix args.
template <template <typename> class ResMT, template <typename> class ArgMT, typename VTWeight, typename VTCoord,
          AnalysisFlag... Fs>
void checkCTableAnalAcc(const DenseMatrix<VTCoord> *lhsSrc, const DenseMatrix<VTCoord> *rhsSrc, VTWeight weight,
                        int64_t resNumRows, int64_t resNumCols) {
    using DTRes = ResMT<VTWeight>;
    using DTArg = ArgMT<VTCoord>;

    const DTArg *lhs = nullptr;
    const DTArg *rhs = nullptr;
    if constexpr (std::is_same_v<DTArg, DenseMatrix<VTCoord>>) {
        lhs = lhsSrc;
        rhs = rhsSrc;
    } else {
        lhs = static_cast<const DTArg *>(lhsSrc);
        rhs = static_cast<const DTArg *>(rhsSrc);
    }

    DTRes *res = nullptr;
    ctableAnalAcc<DTRes, DTArg, DTArg, VTWeight, AnalysisFlags<Fs...>>(res, lhs, rhs, weight, resNumRows, resNumCols,
                                                                       nullptr);

    DTRes *expRes = nullptr;
    ctableAnalAcc<DTRes, DTArg, DTArg, VTWeight, AnalysisFlags<>>(expRes, lhs, rhs, weight, resNumRows, resNumCols,
                                                                  nullptr);
    naiveAnalysis<DTRes>(expRes, nullptr);

    CHECK(checkEqApprox(res, expRes, 1e-2, nullptr));
    checkAnalysisResult<DTRes, Fs...>(res, expRes);

    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(expRes);
}

// Dense result
TEMPLATE_TEST_CASE(TEST_NAME("dense, no analysis"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 1});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 2});
    checkCTableAnalAcc<DenseMatrix, DenseMatrix, VT, int64_t>(lhs, rhs, static_cast<VT>(1), -1, -1);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("dense: mean, max, sparsity"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 1});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 2});
    checkCTableAnalAcc<DenseMatrix, DenseMatrix, VT, int64_t, AnalysisFlag::mean, AnalysisFlag::max,
                       AnalysisFlag::sparsity>(lhs, rhs, static_cast<VT>(1), -1, -1);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("dense: min, numDistinct"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 1});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 2});
    checkCTableAnalAcc<DenseMatrix, DenseMatrix, VT, int64_t, AnalysisFlag::min, AnalysisFlag::numDistinct>(
        lhs, rhs, static_cast<VT>(1), -1, -1);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("dense: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(6, {0, 1, 0, 2, 1, 0});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(6, {0, 1, 0, 2, 2, 0});
    checkCTableAnalAcc<DenseMatrix, DenseMatrix, VT, int64_t, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                       AnalysisFlag::numDistinct, AnalysisFlag::sparsity>(lhs, rhs, static_cast<VT>(2), -1, -1);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("dense, symmetric table: symmetry, min, numDistinct"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(4, {0, 1, 2, 2});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(4, {1, 0, 2, 2});
    checkCTableAnalAcc<DenseMatrix, DenseMatrix, VT, int64_t, AnalysisFlag::symmetry, AnalysisFlag::min,
                       AnalysisFlag::numDistinct>(lhs, rhs, static_cast<VT>(1), 3, 3);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("dense, explicit dims with zeros: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                   VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(4, {0, 1, 0, 2});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(4, {0, 1, 0, 1});
    // 4x4 result but coords only fill a corner -> many zero cells.
    checkCTableAnalAcc<DenseMatrix, DenseMatrix, VT, int64_t, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                       AnalysisFlag::numDistinct, AnalysisFlag::sparsity>(lhs, rhs, static_cast<VT>(1), 4, 4);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

// CSR result
TEMPLATE_TEST_CASE(TEST_NAME("csr: mean, max, sparsity"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 1});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 2});
    checkCTableAnalAcc<CSRMatrix, DenseMatrix, VT, int64_t, AnalysisFlag::mean, AnalysisFlag::max,
                       AnalysisFlag::sparsity>(lhs, rhs, static_cast<VT>(1), -1, -1);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("csr: min, numDistinct"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 1});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 2});
    checkCTableAnalAcc<CSRMatrix, DenseMatrix, VT, int64_t, AnalysisFlag::min, AnalysisFlag::numDistinct>(
        lhs, rhs, static_cast<VT>(1), -1, -1);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("csr, symmetric table: symmetry, min, numDistinct"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(4, {0, 1, 2, 2});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(4, {1, 0, 2, 2});
    checkCTableAnalAcc<CSRMatrix, DenseMatrix, VT, int64_t, AnalysisFlag::symmetry, AnalysisFlag::min,
                       AnalysisFlag::numDistinct>(lhs, rhs, static_cast<VT>(1), 3, 3);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("matrix: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 1});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(5, {0, 1, 0, 2, 2});
    checkCTableAnalAcc<Matrix, Matrix, VT, int64_t, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                       AnalysisFlag::numDistinct, AnalysisFlag::sparsity>(lhs, rhs, static_cast<VT>(1), -1, -1);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("matrix, symmetric table: symmetry, min, numDistinct"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(4, {0, 1, 2, 2});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(4, {1, 0, 2, 2});
    checkCTableAnalAcc<Matrix, Matrix, VT, int64_t, AnalysisFlag::symmetry, AnalysisFlag::min,
                       AnalysisFlag::numDistinct>(lhs, rhs, static_cast<VT>(1), 3, 3);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_TEST_CASE(TEST_NAME("numDistinctApprox sanity"), TAG_KERNELS, VALUE_TYPES) {
    using VT = TestType;
    auto lhs = genGivenVals<DenseMatrix<int64_t>>(4, {0, 1, 2, 2});
    auto rhs = genGivenVals<DenseMatrix<int64_t>>(4, {1, 0, 2, 2});
    checkCTableAnalAcc<Matrix, Matrix, VT, int64_t, AnalysisFlag::symmetry, AnalysisFlag::min,
                       AnalysisFlag::numDistinctApprox>(lhs, rhs, static_cast<VT>(1), 3, 3);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}