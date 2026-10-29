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
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/analysis-fusion/EwUnaryMatAnalAcc.h>
#include <runtime/local/kernels/analysis-fusion/NaiveAnalysis.h>
#include <runtime/local/kernels/analysis-fusion/Util.h>

#include <tags.h>

#include <catch.hpp>

#define TEST_NAME(opName) "EwUnaryMatAnalysisAcc (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES int32_t, double

template <typename DTRes, typename DTArg, AnalysisFlag... Fs>
void checkEwUnaryMatAnalAcc(UnaryOpCode opCode, const DTArg *arg) {
    DTRes *res = nullptr;
    DTRes *expRes = nullptr;
    ewUnaryMatAnalysisAcc<DTRes, DTArg, AnalysisFlags<Fs...>>(opCode, res, arg, nullptr);
    ewUnaryMat<DTRes, DTArg>(opCode, expRes, arg, nullptr);
    naiveAnalysis<DTRes>(expRes, nullptr);

    CHECK(checkEqApprox(res, expRes, 1e-2, nullptr));
    checkAnalysisResult<DTRes, Fs...>(res, expRes);
    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(expRes);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("abs with no result analysis: "), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>(3, {
                                       0,
                                       1,
                                       5,
                                   });

    checkEwUnaryMatAnalAcc<DT, DT>(UnaryOpCode::ABS, arg);

    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("abs, non-symmetric square matrix: mean, min, numDistinct, sparsity, symmetry"),
                           TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(2, {
                                       1,
                                       2,
                                       3,
                                       4,
                                   });
    checkEwUnaryMatAnalAcc<DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::numDistinct,
                           AnalysisFlag::sparsity, AnalysisFlag::symmetry>(UnaryOpCode::ABS, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("abs, symmetric square matrix: mean, min, numDistinct, sparsity, symmetry"),
                           TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    // 3x3 matrix
    auto arg = genGivenVals<DT>(3, {
                                       1,
                                       -2,
                                       0,
                                       -2,
                                       3,
                                       4,
                                       0,
                                       4,
                                       -5,
                                   });
    checkEwUnaryMatAnalAcc<DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::numDistinct,
                           AnalysisFlag::sparsity, AnalysisFlag::symmetry>(UnaryOpCode::ABS, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("abs, symmetric 4x4 matrix: mean, min, numDistinct, sparsity, symmetry"),
                           TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    auto arg = genGivenVals<DT>(4, {
                                       1,
                                       -2,
                                       3,
                                       0,
                                       -2,
                                       4,
                                       0,
                                       5,
                                       3,
                                       0,
                                       6,
                                       -1,
                                       0,
                                       5,
                                       -1,
                                       7,
                                   });
    checkEwUnaryMatAnalAcc<DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                           AnalysisFlag::sparsity, AnalysisFlag::symmetry>(UnaryOpCode::ABS, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(
    TEST_NAME("exp, zero maps to non-zero, non-square matrix: mean, min, max, numDistinct, sparsity, symmetry"),
    TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    // Useful to check for ops that map 0 to 1.
    auto arg = genGivenVals<DT>(2, {
                                       0,
                                       1,
                                       2,
                                       -1,
                                       0,
                                       3,
                                   });
    checkEwUnaryMatAnalAcc<DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                           AnalysisFlag::sparsity, AnalysisFlag::symmetry>(UnaryOpCode::EXP, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("round, non-zero maps to zero: mean, min, max, numDistinct, sparsity, symmetry"),
                           TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;
    // check if mapping non-zero to zero is correct.
    auto arg = genGivenVals<DT>(3, {
                                       0,
                                       1,
                                       0,
                                       2,
                                       0,
                                       3,
                                       0,
                                       4,
                                       0,
                                   });
    checkEwUnaryMatAnalAcc<DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::numDistinct,
                           AnalysisFlag::sparsity, AnalysisFlag::symmetry>(UnaryOpCode::ROUND, arg);
    DataObjectFactory::destroy(arg);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("numDistinctApprox sanity"), TAG_KERNELS, (DenseMatrix), (double)) {
    using DT = TestType;
    using VT = typename DT::VT;

    const size_t n = 100;
    auto arg = DataObjectFactory::create<DT>(n, n, false);
    VT *v = arg->getValues();
    for (size_t i = 0; i < n * n; i++)
        v[i] = static_cast<VT>(i);

    DT *res = nullptr;
    ewUnaryMatAnalysisAcc<DT, DT, AnalysisFlags<AnalysisFlag::numDistinctApprox>>(UnaryOpCode::ABS, res, arg, nullptr);

    CHECK(res->numDistinct.has_value());
    double est = static_cast<double>(res->numDistinct.value());
    double truth = static_cast<double>(n * n);

    CHECK(std::abs(est - truth) / truth < 0.25); // within 25%

    DataObjectFactory::destroy(res);
    DataObjectFactory::destroy(arg);
}
