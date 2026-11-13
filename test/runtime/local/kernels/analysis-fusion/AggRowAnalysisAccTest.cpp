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
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/AggRow.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/kernels/analysis-fusion/AggRowAnalAcc.h>

#include <runtime/local/kernels/analysis-fusion/NaiveAnalysis.h>
#include <runtime/local/kernels/analysis-fusion/Util.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

#define TEST_NAME(opName) "AggRowAnalysisAcc (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix

// Separate list of types that support IDXMIN/MAX
#define DATA_TYPES_IDX DenseMatrix, Matrix
#define VALUE_TYPES int32_t, double
#define VALUE_TYPES_FP double

// helper mappings for types for simplicity
template <class DTArg> struct AggRowResType;
template <typename VT> struct AggRowResType<DenseMatrix<VT>> {
    using type = DenseMatrix<VT>;
};
template <typename VT> struct AggRowResType<CSRMatrix<VT>> {
    using type = DenseMatrix<VT>;
};
template <typename VT> struct AggRowResType<Matrix<VT>> {
    using type = Matrix<VT>;
};

template <class DTArg, AnalysisFlag... Fs> void checkAggRowAnal(AggOpCode opCode, const DTArg *arg) {
    using DTRes = typename AggRowResType<DTArg>::type;

    DTRes *res = nullptr;
    DTRes *expRes = nullptr;

    aggRowAnalysisAcc<DTRes, DTArg, AnalysisFlags<Fs...>>(opCode, res, arg, nullptr);

    aggRow<DTRes, DTArg>(opCode, expRes, arg, nullptr);
    naiveAnalysis<DTRes>(expRes, nullptr);

    CHECK(checkEqApprox(res, expRes, 1e-2, nullptr));
    checkAnalysisResult<DTRes, Fs...>(res, expRes);

    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(expRes);
}

// SUM
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, no analysis"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {1, 0, 3, 0, 5, 0, 7, 0, 9});
    checkAggRowAnal<DT>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, dense data: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, rows summing to zero: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    // first two rows sum to 0
    auto arg = genGivenVals<DT>(3, {5, -3, -2, 0, 0, 0, 1, 2, 3});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, all zeros: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, single column: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(4, {1, 0, -2, 3});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

// PROD (not sparse-safe -> CSR path has to check for extra zeros)
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("prod: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    // row 0 -> 6, row 1 -> 0 (contains a zero), row 2 -> 0 (empty row)
    auto arg = genGivenVals<DT>(3, {1, 2, 3, 4, 0, 6, 0, 0, 0});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::PROD, arg);
    DataObjectFactory::destroy(arg);
}

// MIN / MAX
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {3, 1, 2, -1, 0, 4, 7, 8, 9});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::MIN, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {3, 1, 2, -1, 0, -4, 0, 0, 0});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::MAX, arg);
    DataObjectFactory::destroy(arg);
}

// MEAN / STDDEV / VAR

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mean: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES_FP)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {1, 2, 3, 0, 0, 0, 4, 0, 8});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::MEAN, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("stddev: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES_FP)) {
    using DT = TestType;
    // row 1 is constant -> stddev 0, row 2 is empty -> stddev 0
    auto arg = genGivenVals<DT>(3, {1, 2, 3, 5, 5, 5, 0, 0, 0});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::STDDEV, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("var: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES_FP)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {1, 2, 3, 5, 5, 5, 0, 4, 0});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::VAR, arg);
    DataObjectFactory::destroy(arg);
}

// IDXMIN / IDXMAX

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("idxmin: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES_IDX),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {3, 1, 2, -1, 0, 4, 7, 7, 5});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::IDXMIN, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("idxmax: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES_IDX),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {3, 1, 2, 9, 0, 4, 0, 0, 5});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity>(AggOpCode::IDXMAX, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("idxmin, no analysis"), TAG_KERNELS, (DATA_TYPES_IDX), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(2, {4, 2, 6, 1, 5, 3});
    checkAggRowAnal<DT>(AggOpCode::IDXMIN, arg);
    DataObjectFactory::destroy(arg);
}

// 1 x 1 square and symmetric matrix

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, 1x4 arg (1x1 result): mean, min, max, numDistinct, sparsity, symmetry"),
                           TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(1, {1, 0, 3, 4});
    checkAggRowAnal<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                    AnalysisFlag::sparsity, AnalysisFlag::symmetry>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, non-square result (3x1): symmetry"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {1, 2, 0, 4, 0, 6});
    checkAggRowAnal<DT, AnalysisFlag::symmetry>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("numDistinctApprox sanity"), TAG_KERNELS, (DenseMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    const size_t n = 100;
    auto arg = DataObjectFactory::create<DT>(n, n, false);
    VT *v = arg->getValues();
    for (size_t r = 0; r < n; r++)
        for (size_t c = 0; c < n; c++)
            v[r * n + c] = static_cast<VT>(c);

    DenseMatrix<VT> *res = nullptr;
    aggRowAnalysisAcc<DenseMatrix<VT>, DT, AnalysisFlags<AnalysisFlag::numDistinctApprox>>(AggOpCode::MEAN, res, arg,
                                                                                           nullptr);
    std::unordered_set<VT> distinct;
    for (size_t r = 0; r < res->getNumRows(); r++)
        for (size_t c = 0; c < res->getNumCols(); c++)
            distinct.insert(res->get(r, c));
    double truth = static_cast<double>(distinct.size());

    CHECK(res->numDistinct.has_value());
    double est = static_cast<double>(res->numDistinct.value());
    CHECK(std::abs(est - truth) / truth < 0.25);

    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(arg);
}