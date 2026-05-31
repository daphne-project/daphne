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
#define DATA_TYPES DenseMatrix //, CSRMatrix, Matrix
#define VALUE_TYPES int32_t, double

// compare two analysis results for equality with a given epsilon value if there is floating point in there.
template <AnalysisFlag... Fs, typename... TArgs>
bool checkAnalEqApprox(const AnalysisResult<FlagArg<Fs, TArgs>...> &resA,
                       const AnalysisResult<FlagArg<Fs, TArgs>...> &resB, float eps = 1e-2) {
    // if constexpr is a statement and not an expression, so we wrap it in a lambda function body.
    // for each flag, we compare the matching value.
    return ([&]() {
        auto &va = resA.template get<Fs>();
        auto &vb = resB.template get<Fs>();
        // static dispatch to floating point comparison if floating point.
        if constexpr (std::is_floating_point_v<TArgs>) {
            return std::abs(va - vb) < eps;
        } else {
            return va == vb;
        }
    }() && ...);
}

template <typename DTRes, typename DTArg, typename DTAnalRes>
void checkEwUnaryMatAnal(UnaryOpCode opCode, const DTArg *arg, const DTRes *exp, const DTAnalRes &analExp) {
    DTRes *res = nullptr;
    DTAnalRes anal;
    ewUnaryMatAnalysis<DTRes, DTArg, DTAnalRes>(opCode, res, arg, anal, nullptr);
    CHECK(*res == *exp);
    // compare by value
    CHECK(checkAnalEqApprox(anal, analExp));
    DataObjectFactory::destroy(res);
}

// ****************************************************************************
// Arithmetic/general math
// ****************************************************************************

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("abs with no result analysis: "), TAG_KERNELS, (DATA_TYPES), (VALUE_TYPES)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>(3, {
                                       0,
                                       1,
                                       -1,
                                   });

    auto exp = genGivenVals<DT>(3, {
                                       0,
                                       1,
                                       1,
                                   });

    AnalysisResult<> anal;
    checkEwUnaryMatAnal(UnaryOpCode::ABS, arg, exp, anal);

    DataObjectFactory::destroy(arg, exp);
}

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("abs with all possible result analysis: "), TAG_KERNELS, (DATA_TYPES),
                           (VALUE_TYPES)) {
    using DT = TestType;

    auto arg = genGivenVals<DT>(3, {
                                       0,
                                       1,
                                       -5,
                                   });

    auto exp = genGivenVals<DT>(3, {
                                       0,
                                       1,
                                       5,
                                   });
    using VT = DT::VT;
    AnalysisResult<FlagArg<AnalysisFlag::min, VT>, FlagArg<AnalysisFlag::max, VT>, FlagArg<AnalysisFlag::mean, VT>>
        anal{{0, 5, 2}};
    checkEwUnaryMatAnal(UnaryOpCode::ABS, arg, exp, anal);
    DataObjectFactory::destroy(arg, exp);
}
