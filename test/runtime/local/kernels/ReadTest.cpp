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
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/kernels/Read.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

static Frame *emptyFrame = DataObjectFactory::create<Frame>(0, 0, nullptr, nullptr, false);

TEMPLATE_PRODUCT_TEST_CASE("Read CSV", TAG_KERNELS, (DenseMatrix), (double)) {
    using DT = TestType;

    DT *m = nullptr;

    size_t numRows = 2;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/ReadCsv1.csv";

    read(m, filename, emptyFrame, nullptr);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->get(0, 0) == -0.1);
    CHECK(m->get(0, 1) == -0.2);
    CHECK(m->get(0, 2) == 0.1);
    CHECK(m->get(0, 3) == 0.2);

    CHECK(m->get(1, 0) == 3.14);
    CHECK(m->get(1, 1) == 5.41);
    CHECK(m->get(1, 2) == 6.22216);
    CHECK(m->get(1, 3) == 5);

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("Read MM", TAG_KERNELS, (DenseMatrix), (uint32_t)) {
    using DT = TestType;

    DT *m = nullptr;
    size_t numRows = 9;
    size_t numCols = 9;

    char filename[] = "./test/runtime/local/io/cig.mtx";
    read(m, filename, emptyFrame, nullptr);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->get(0, 0) == 1);
    CHECK(m->get(2, 0) == 0);
    CHECK(m->get(3, 4) == 9);
    CHECK(m->get(7, 4) == 4);

    DataObjectFactory::destroy(m);
}

TEST_CASE("Read - Frame", TAG_KERNELS) {
    Frame *f = nullptr;
    read(f, "./test/runtime/local/io/ReadCsv4.csv", emptyFrame, nullptr);

    CHECK(f->getNumRows() == 2);
    CHECK(f->getNumCols() == 2);
    CHECK(f->getColumnType(0) == ValueTypeCode::SI64);
    CHECK(f->getColumnType(1) == ValueTypeCode::F64);
    CHECK(f->getLabels()[0] == "foo");
    CHECK(f->getLabels()[1] == "bar");

    auto c0 = f->getColumn<int64_t>(0);
    CHECK(c0->get(0, 0) == 1);
    CHECK(c0->get(1, 0) == 2);
    auto c1 = f->getColumn<double>(1);
    CHECK(c1->get(0, 0) == 0.5);
    CHECK(c1->get(1, 0) == 1.0);

    DataObjectFactory::destroy(f);
    DataObjectFactory::destroy(c0);
    DataObjectFactory::destroy(c1);
}
