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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/AggCol.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/kernels/analysis-fusion/AggColAnalAcc.h>
#include <runtime/local/kernels/analysis-fusion/NaiveAnalysis.h>
#include <runtime/local/kernels/analysis-fusion/Util.h>

#include <tags.h>

#include <catch.hpp>

#include <limits>
#include <type_traits>

#include <cstdint>

#define TEST_NAME(opName) "AggColAnalysisAcc (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES int32_t, double

// Type level operator to map the output matrix type depending on the input matrix type
template <typename DTArg> struct AggColResType {
    using type = DenseMatrix<typename DTArg::VT>;
};
template <typename VT> struct AggColResType<Matrix<VT>> {
    using type = Matrix<VT>;
};

template <typename DTArg, AnalysisFlag... Fs> void checkAggColAnalAcc(AggOpCode opCode, const DTArg *arg) {
    using DTRes = typename AggColResType<DTArg>::type;

    DTRes *res = nullptr;
    DTRes *expRes = nullptr;

    aggColAnalysisAcc<DTRes, DTArg, AnalysisFlags<Fs...>>(opCode, res, arg, nullptr);
    aggCol<DTRes, DTArg>(opCode, expRes, arg, nullptr);
    naiveAnalysis<DTRes>(expRes, nullptr);

    CHECK(checkEqApprox(res, expRes, 1e-2, nullptr));
    checkAnalysisResult<DTRes, Fs...>(res, expRes);

    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(expRes);
}

// SUM: pure binary reduction, sparse-safe

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, no analysis"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {
                                       1,
                                       0,
                                       2,
                                       0,
                                       3,
                                       0,
                                       4,
                                       0,
                                       5,
                                   });
    checkAggColAnalAcc<DT>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {
                                       1,
                                       0,
                                       2,
                                       0,
                                       3,
                                       0,
                                       4,
                                       0,
                                       5,
                                   });
    checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                       AnalysisFlag::sparsity>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

// Summing up a column of zeroes should sum to zero
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, one zero column: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {
                                       1,
                                       0,
                                       2,
                                       3,
                                       0,
                                       4,
                                       5,
                                       0,
                                       6,
                                   });
    checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                       AnalysisFlag::sparsity>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

// Single-row input - test CSR numRows<=1
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, single row with zeros: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(1, {5, 0, 0, 3});
    checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                       AnalysisFlag::sparsity>(AggOpCode::SUM, arg);
    DataObjectFactory::destroy(arg);
}

// MIN / MAX: pure binary reduction, sparse-safe
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("min: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {
                                       3,
                                       0,
                                       9,
                                       1,
                                       7,
                                       0,
                                       0,
                                       2,
                                       5,
                                   });
    checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                       AnalysisFlag::sparsity>(AggOpCode::MIN, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("max: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {
                                       3,
                                       0,
                                       9,
                                       1,
                                       7,
                                       0,
                                       0,
                                       2,
                                       5,
                                   });
    checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                       AnalysisFlag::sparsity>(AggOpCode::MAX, arg);
    DataObjectFactory::destroy(arg);
}

// MEAN: not sparse-safe and manual post computation (division)
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mean: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(4, {
                                       2,
                                       0,
                                       4,
                                       0,
                                       6,
                                       0,
                                       8,
                                       0,
                                       10,
                                       0,
                                       12,
                                       0,
                                   });
    checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                       AnalysisFlag::sparsity>(AggOpCode::MEAN, arg);
    DataObjectFactory::destroy(arg);
}

// STDDEV / VAR: not sparse-safe, manual post computation
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("stddev: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    if constexpr (std::is_floating_point_v<VT>) {
        auto arg = genGivenVals<DT>(3, {
                                           1,
                                           0,
                                           4,
                                           2,
                                           5,
                                           0,
                                           3,
                                           0,
                                           6,
                                       });
        checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                           AnalysisFlag::sparsity>(AggOpCode::STDDEV, arg);
        DataObjectFactory::destroy(arg);
    }
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("var: mean, min, max, numDistinct, sparsity"), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    if constexpr (std::is_floating_point_v<VT>) {
        auto arg = genGivenVals<DT>(3, {
                                           1,
                                           0,
                                           4,
                                           2,
                                           5,
                                           0,
                                           3,
                                           0,
                                           6,
                                       });
        checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                           AnalysisFlag::sparsity>(AggOpCode::VAR, arg);
        DataObjectFactory::destroy(arg);
    }
}

// Single column input: check square+symmetric output
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("sum, single column input: mean, min, max, numDistinct, sparsity, symmetry"),
                           TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(3, {4, 0, 6});
    checkAggColAnalAcc<DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                       AnalysisFlag::sparsity, AnalysisFlag::symmetry>(AggOpCode::SUM, arg);
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
    aggColAnalysisAcc<DenseMatrix<VT>, DT, AnalysisFlags<AnalysisFlag::numDistinctApprox>>(AggOpCode::MEAN, res, arg,
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