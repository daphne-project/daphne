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

#include "ir/daphneir/DataPropertyTypes.h"
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/CheckEq.h>
#include <runtime/local/kernels/CheckEqApprox.h>
#include <runtime/local/kernels/analysis-fusion/EwUnaryMatAnalysis.h>

#include <tags.h>

#include <catch.hpp>

#include <limits>

#include <cstdint>

#define TEST_NAME(opName) "EwUnaryMatAnalysis (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES int32_t, double

template <typename DTRes, typename DTArg, AnalysisFlag... Fs>
void checkEwUnaryMatAnal(UnaryOpCode opCode, const DTArg *arg, const DTRes *exp) {
    DTRes *res = nullptr;
    ewUnaryMatAnalysis<DTRes, DTArg, AnalysisFlags<Fs...>>(opCode, res, arg, nullptr);
    CHECK(checkEqApprox(res, exp, 1e-2, nullptr));
    checkEwUnaryMatAnalOnly<DTRes, Fs...>(opCode, res, exp);
    DataObjectFactory::destroy(res);
}

template <typename DT, AnalysisFlag... Fs>
void checkEwUnaryMatAnalOnly(UnaryOpCode opCode, const DT *arg, const DT *exp) {
    using anal_t = AnalysisFlags<Fs...>;
    if constexpr (anal_t::template contains<AnalysisFlag::mean>) {
        CHECK(arg->mean.has_value() == exp->mean.has_value());
        if (arg->mean.has_value())
            CHECK(std::abs(arg->mean.value() - exp->mean.value()) < 1e-2);
    }
    if constexpr (anal_t::template contains<AnalysisFlag::min>) {
        CHECK(arg->min.has_value() == exp->min.has_value());
        if (arg->min.has_value())
            CHECK(arg->min.value() == exp->min.value());
    }
    if constexpr (anal_t::template contains<AnalysisFlag::sparsity>)
        CHECK(std::abs(arg->sparsity - exp->sparsity) < 1e-2);

    if constexpr (anal_t::template contains<AnalysisFlag::symmetry>)
        CHECK(arg->symmetric == exp->symmetric);

    if constexpr (anal_t::template contains<AnalysisFlag::numDistinct>) {
        CHECK(arg->numDistinct.has_value() == exp->numDistinct.has_value());
        CHECK(arg->numDistinct.value() == exp->numDistinct.value());
    }
}

// ****************************************************************************
// Arithmetic/general math
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("abs with no result analysis: "), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>(3, {
                                       0,
                                       1,
                                       5,
                                   });

    auto exp = genGivenVals<DT>(3, {
                                       0,
                                       1,
                                       5,
                                   });

    checkEwUnaryMatAnal(UnaryOpCode::ABS, arg, exp);

    DataObjectFactory::destroy(arg, exp);
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
    auto exp = genGivenVals<DT>(2, {
                                       1,
                                       2,
                                       3,
                                       4,
                                   });
    exp->mean = 2.5; // (1 + 2 + 3 + 4) / 4
    exp->min = 1;
    exp->numDistinct = 4;
    exp->sparsity = 1.0;
    exp->symmetric = BoolOrUnknown::False;
    checkEwUnaryMatAnal<DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::numDistinct,
                        AnalysisFlag::sparsity, AnalysisFlag::symmetry>(UnaryOpCode::ABS, arg, exp);
    DataObjectFactory::destroy(arg, exp);
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
    auto exp = genGivenVals<DT>(3, {
                                       1,
                                       2,
                                       0,
                                       2,
                                       3,
                                       4,
                                       0,
                                       4,
                                       5,
                                   });
    exp->mean = 21.0 / 9.0; // (1+2+0+2+3+4+0+4+5)/9
    exp->min = 0;
    exp->numDistinct = 6;
    exp->sparsity = 7.0 / 9.0; // 7 non-zeros
    exp->symmetric = BoolOrUnknown::True;
    checkEwUnaryMatAnal<DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::numDistinct,
                        AnalysisFlag::sparsity, AnalysisFlag::symmetry>(UnaryOpCode::ABS, arg, exp);
    DataObjectFactory::destroy(arg, exp);
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
    auto exp = genGivenVals<DT>(4, {
                                       1,
                                       2,
                                       3,
                                       0,
                                       2,
                                       4,
                                       0,
                                       5,
                                       3,
                                       0,
                                       6,
                                       1,
                                       0,
                                       5,
                                       1,
                                       7,
                                   });
    exp->mean = 40.0 / 16.0;
    exp->min = 0;
    exp->numDistinct = 8;        // 0,1,2,3,4,5,6,7
    exp->sparsity = 12.0 / 16.0; // 4 zeros
    exp->symmetric = BoolOrUnknown::True;
    checkEwUnaryMatAnal<DT, DT, AnalysisFlag::mean, AnalysisFlag::min, AnalysisFlag::numDistinct,
                        AnalysisFlag::sparsity, AnalysisFlag::symmetry>(UnaryOpCode::ABS, arg, exp);
    DataObjectFactory::destroy(arg, exp);
}