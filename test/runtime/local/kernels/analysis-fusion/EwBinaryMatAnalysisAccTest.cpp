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
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/kernels/EwBinaryMat.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/analysis-fusion/EwBinaryMatAnalAcc.h>

#include <runtime/local/kernels/analysis-fusion/NaiveAnalysis.h>
#include <runtime/local/kernels/analysis-fusion/Util.h>

#include <tags.h>

#include <catch.hpp>

#include <limits>

#include <cstdint>

#define TEST_NAME(opName) "EwBinaryMatAnalysisAcc (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES int32_t, double

template <typename DTRes, typename DTLhs, typename DTRhs, AnalysisFlag... Fs>
void checkEwBinaryMatAnalAcc(BinaryOpCode opCode, const DTLhs *lhs, const DTLhs *rhs) {
    DTRes *res = nullptr;
    DTRes *expRes = nullptr;
    ewBinaryMatAnalAcc<DTRes, DTLhs, DTRhs, AnalysisFlags<Fs...>>(opCode, res, lhs, rhs, nullptr);
    ewBinaryMat<DTRes, DTLhs, DTRhs>(opCode, expRes, lhs, rhs, nullptr);
    naiveAnalysis<DTRes>(expRes, nullptr);

    CHECK(checkEqApprox(res, expRes, 1e-2, nullptr));
    checkAnalysisResult<DTRes, Fs...>(res, expRes);
    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(expRes);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add, no analysis"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto lhs = genGivenVals<DT>(2, {1, 0, 3, 0, 5, 0});
    auto rhs = genGivenVals<DT>(2, {0, 2, 0, 4, 0, 6});
    checkEwBinaryMatAnalAcc<DT, DT, DT>(BinaryOpCode::ADD, lhs, rhs);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add, disjoint sparsity: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    // lhs and rhs have non-zeros in different positions -> merge, no overlap.
    auto lhs = genGivenVals<DT>(3, {1, 0, 0, 0, 2, 0, 0, 0, 3});
    auto rhs = genGivenVals<DT>(3, {0, 4, 0, 5, 0, 0, 0, 6, 0});
    checkEwBinaryMatAnalAcc<DT, DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                            AnalysisFlag::numDistinct, AnalysisFlag::sparsity>(BinaryOpCode::ADD, lhs, rhs);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add, overlapping sparsity: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    // Some positions overlap
    auto lhs = genGivenVals<DT>(3, {1, 2, 0, 0, 3, 0, 4, 0, 5});
    auto rhs = genGivenVals<DT>(3, {1, 0, 6, 0, 3, 7, 0, 0, 5});
    checkEwBinaryMatAnalAcc<DT, DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                            AnalysisFlag::numDistinct, AnalysisFlag::sparsity>(BinaryOpCode::ADD, lhs, rhs);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add, sum to zero: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    // Overlapping positions summing to zero.
    auto lhs = genGivenVals<DT>(2, {5, -3, 2, 0});
    auto rhs = genGivenVals<DT>(2, {-5, 3, 1, 0});
    checkEwBinaryMatAnalAcc<DT, DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                            AnalysisFlag::numDistinct, AnalysisFlag::sparsity>(BinaryOpCode::ADD, lhs, rhs);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("mul, intersecting non-zeros: mean, min, max, numDistinct, sparsity"), TAG_KERNELS,
                           (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    // MUL keeps only positions where BOTH are non-zero.
    auto lhs = genGivenVals<DT>(3, {2, 3, 0, 0, 4, 5, 6, 0, 7});
    auto rhs = genGivenVals<DT>(3, {1, 0, 8, 0, 2, 3, 1, 0, 0});
    checkEwBinaryMatAnalAcc<DT, DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                            AnalysisFlag::numDistinct, AnalysisFlag::sparsity>(BinaryOpCode::MUL, lhs, rhs);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add, symmetric result: mean, min, max, numDistinct, sparsity, symmetry"),
                           TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto lhs = genGivenVals<DT>(3, {1, 2, 0, 2, 3, 4, 0, 4, 5});
    auto rhs = genGivenVals<DT>(3, {0, 1, 6, 1, 0, 2, 6, 2, 0});
    checkEwBinaryMatAnalAcc<DT, DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                            AnalysisFlag::numDistinct, AnalysisFlag::sparsity, AnalysisFlag::symmetry>(
        BinaryOpCode::ADD, lhs, rhs);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add, asymmetric result: symmetry"), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto lhs = genGivenVals<DT>(3, {1, 2, 3, 0, 4, 5, 0, 0, 6});
    auto rhs = genGivenVals<DT>(3, {1, 0, 0, 7, 4, 0, 8, 9, 6});
    checkEwBinaryMatAnalAcc<DT, DT, DT, AnalysisFlag::symmetry>(BinaryOpCode::ADD, lhs, rhs);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("add, non-square(2x3): mean, min, max, numDistinct, sparsity, symmetry"),
                           TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto lhs = genGivenVals<DT>(2, {1, 0, 3, 0, 5, 0});
    auto rhs = genGivenVals<DT>(2, {0, 2, 0, 4, 0, 6});
    checkEwBinaryMatAnalAcc<DT, DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max,
                            AnalysisFlag::numDistinct, AnalysisFlag::sparsity, AnalysisFlag::symmetry>(
        BinaryOpCode::ADD, lhs, rhs);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("numDistinctApprox sanity"), TAG_KERNELS, (DenseMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;
    const size_t n = 100;
    auto lhs = DataObjectFactory::create<DT>(n, n, false);
    VT *v = lhs->getValues();
    for (size_t i = 0; i < n * n; i++)
        v[i] = static_cast<VT>(i);
    auto rhs = DataObjectFactory::create<DT>(n, n, false);
    VT *v2 = lhs->getValues();
    for (size_t i = 0; i < n * n; i++)
        v2[i] = static_cast<VT>(i);

    DT *res = nullptr;
    ewBinaryMatAnalAcc<DT, DT, DT, AnalysisFlags<AnalysisFlag::numDistinctApprox>>(BinaryOpCode::ADD, res, lhs, rhs,
                                                                                   nullptr);

    CHECK(res->numDistinct.has_value());
    double est = static_cast<double>(res->numDistinct.value());
    double truth = static_cast<double>(n * n);

    CHECK(std::abs(est - truth) / truth < 0.25); // within 25%

    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(lhs);
    DataObjectFactory::destroy(rhs);
}
