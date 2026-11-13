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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/kernels/GroupJoin.h>

#include <tags.h>

#include <catch.hpp>

#include <string>
#include <vector>

#include <cstdint>

static size_t rowOf(const Frame *res, int64_t key) {
    auto ids = res->getColumn<int64_t>(0);
    for (size_t r = 0; r < res->getNumRows(); r++)
        if (ids->get(r, 0) == key)
            return r;
    FAIL("expected groupJoin result key not found: " << key);
    return 0;
}

TEST_CASE("GroupJoin", TAG_KERNELS) {
    auto lhsC0 = genGivenVals<DenseMatrix<int64_t>>(3, {1, 2, 3});
    auto lhsC1 = genGivenVals<DenseMatrix<int64_t>>(3, {11, 22, 33});
    std::vector<Structure *> lhsCols = {lhsC0, lhsC1};
    std::string lhsLabels[] = {"d.id", "d.foo"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhsC0 = genGivenVals<DenseMatrix<int64_t>>(10, {1, 1, 1, 3, 1, 3, 3, 1, 3, 3});
    auto rhsC1 = genGivenVals<DenseMatrix<int64_t>>(10, {42, 42, 42, 42, 42, 42, 42, 42, 42, 42});
    auto rhsC2 = genGivenVals<DenseMatrix<double>>(10, {10, 20, 30, 10, 20, 30, 10, 20, 30, 10});
    std::vector<Structure *> rhsCols = {rhsC0, rhsC1, rhsC2};
    std::string rhsLabels[] = {"f.id", "f.bar", "f.agg"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);

    Frame *res = nullptr;
    DenseMatrix<size_t> *lhsTid = nullptr;
    const char *rhsAggCols[] = {"f.agg"};
    mlir::daphne::GroupEnum rhsAggFuncs[] = {mlir::daphne::GroupEnum::SUM};
    groupJoin<size_t>(res, lhsTid, lhs, rhs, "d.id", "f.id", rhsAggCols, 1, rhsAggFuncs, 1, nullptr);

    CHECK(res->getNumRows() == 2);
    CHECK(res->getNumCols() == 2);
    CHECK(res->getColumnType(0) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(1) == ValueTypeCode::F64);
    CHECK(res->getLabels()[0] == "d.id");
    CHECK(res->getLabels()[1] == "SUM(f.agg)");
    CHECK(lhsTid->getNumRows() == 2);
    CHECK(lhsTid->getNumCols() == 1);

    auto sums = res->getColumn<double>(1);
    const size_t r1 = rowOf(res, 1);
    CHECK(sums->get(r1, 0) == 100);
    CHECK(lhsTid->get(r1, 0) == 0);
    const size_t r3 = rowOf(res, 3);
    CHECK(sums->get(r3, 0) == 90);
    CHECK(lhsTid->get(r3, 0) == 2);
}

TEST_CASE("GroupJoin multi aggregate", TAG_KERNELS) {
    auto lhsC0 = genGivenVals<DenseMatrix<int64_t>>(3, {1, 2, 3});
    std::vector<Structure *> lhsCols = {lhsC0};
    std::string lhsLabels[] = {"d.id"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto rhsC0 = genGivenVals<DenseMatrix<int64_t>>(6, {1, 1, 3, 1, 3, 3});
    auto rhsC1 = genGivenVals<DenseMatrix<int64_t>>(6, {10, 20, 30, 40, 50, 10});
    auto rhsC2 = genGivenVals<DenseMatrix<double>>(6, {1.5, 2.5, 4.0, 6.0, 8.0, 2.0});
    auto rhsC3 = genGivenVals<DenseMatrix<int64_t>>(6, {100, 200, 300, 400, 500, 600});
    std::vector<Structure *> rhsCols = {rhsC0, rhsC1, rhsC2, rhsC3};
    std::string rhsLabels[] = {"f.id", "f.val", "f.dbl", "f.alt"};
    auto rhs = DataObjectFactory::create<Frame>(rhsCols, rhsLabels);

    Frame *res = nullptr;
    DenseMatrix<size_t> *lhsTid = nullptr;
    const char *rhsAggCols[] = {"f.val", "f.val", "f.val", "f.alt", "f.dbl"};
    mlir::daphne::GroupEnum rhsAggFuncs[] = {mlir::daphne::GroupEnum::COUNT, mlir::daphne::GroupEnum::SUM,
                                             mlir::daphne::GroupEnum::MIN, mlir::daphne::GroupEnum::MAX,
                                             mlir::daphne::GroupEnum::AVG};
    groupJoin<size_t>(res, lhsTid, lhs, rhs, "d.id", "f.id", rhsAggCols, 5, rhsAggFuncs, 5, nullptr);

    // Meta data.
    CHECK(res->getNumRows() == 2);
    CHECK(res->getNumCols() == 6);
    CHECK(res->getColumnType(0) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(1) == ValueTypeCode::UI64);
    CHECK(res->getColumnType(2) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(3) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(4) == ValueTypeCode::SI64);
    CHECK(res->getColumnType(5) == ValueTypeCode::F64);
    CHECK(res->getLabels()[1] == "COUNT(f.val)");
    CHECK(res->getLabels()[2] == "SUM(f.val)");
    CHECK(res->getLabels()[3] == "MIN(f.val)");
    CHECK(res->getLabels()[4] == "MAX(f.alt)");
    CHECK(res->getLabels()[5] == "AVG(f.dbl)");

    auto counts = res->getColumn<uint64_t>(1);
    auto sums = res->getColumn<int64_t>(2);
    auto mins = res->getColumn<int64_t>(3);
    auto maxs = res->getColumn<int64_t>(4);
    auto avgs = res->getColumn<double>(5);

    const size_t r1 = rowOf(res, 1);
    CHECK(counts->get(r1, 0) == 3);
    CHECK(sums->get(r1, 0) == 70);
    CHECK(mins->get(r1, 0) == 10);
    CHECK(maxs->get(r1, 0) == 400);
    CHECK(avgs->get(r1, 0) == 10.0 / 3.0);
    CHECK(lhsTid->get(r1, 0) == 0);

    const size_t r3 = rowOf(res, 3);
    CHECK(counts->get(r3, 0) == 3);
    CHECK(sums->get(r3, 0) == 90);
    CHECK(mins->get(r3, 0) == 10);
    CHECK(maxs->get(r3, 0) == 600);
    CHECK(avgs->get(r3, 0) == 14.0 / 3.0);
    CHECK(lhsTid->get(r3, 0) == 2);
}
