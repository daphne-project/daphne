/*
 * Copyright 2026 The DAPHNE Consortium
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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/ReadOrc.h>
#include <runtime/local/kernels/Read.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// Happy paths
// ----------------------------------------------------------------------------

TEST_CASE("ReadOrc, DenseMatrix<double>, basic", TAG_IO) {
    DenseMatrix<double> *m = DataObjectFactory::create<DenseMatrix<double>>(2, 4, false);

    FileMetaData fmd(2, 4, true, ValueTypeCode::F64);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseDouble.orc";

    readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr);

    REQUIRE(m->getNumRows() == 2);
    REQUIRE(m->getNumCols() == 4);
    CHECK(m->get(0, 0) == -0.1);
    CHECK(m->get(0, 1) == -0.2);
    CHECK(m->get(0, 2) == 0.1);
    CHECK(m->get(0, 3) == 0.2);
    CHECK(m->get(1, 0) == 3.14);
    CHECK(m->get(1, 1) == 5.41);
    CHECK(m->get(1, 2) == 6.22216);
    CHECK(m->get(1, 3) == 5.0);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, DenseMatrix<int64_t>, basic", TAG_IO) {
    DenseMatrix<int64_t> *m = DataObjectFactory::create<DenseMatrix<int64_t>>(2, 4, false);

    FileMetaData fmd(2, 4, true, ValueTypeCode::SI64);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseInt64.orc";

    readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr);

    REQUIRE(m->getNumRows() == 2);
    REQUIRE(m->getNumCols() == 4);
    CHECK(m->get(0, 0) == 1);
    CHECK(m->get(0, 1) == 2);
    CHECK(m->get(0, 2) == 3);
    CHECK(m->get(0, 3) == 4);
    CHECK(m->get(1, 0) == 5);
    CHECK(m->get(1, 1) == 6);
    CHECK(m->get(1, 2) == 7);
    CHECK(m->get(1, 3) == 8);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, Frame, mixed f64/si64/f64", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::F64, ValueTypeCode::SI64, ValueTypeCode::F64};
    std::string labelsArr[] = {"a", "b", "c"};
    Frame *f = DataObjectFactory::create<Frame>(3, 3, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::F64, ValueTypeCode::SI64, ValueTypeCode::F64};
    std::vector<std::string> labels{"a", "b", "c"};
    FileMetaData fmd(3, 3, false, schema, labels);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_Frame.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 3);
    REQUIRE(f->getNumCols() == 3);

    CHECK(f->getColumn<double>(0)->get(0, 0) == 1.1);
    CHECK(f->getColumn<double>(0)->get(1, 0) == 2.2);
    CHECK(f->getColumn<double>(0)->get(2, 0) == 3.3);

    CHECK(f->getColumn<int64_t>(1)->get(0, 0) == 10);
    CHECK(f->getColumn<int64_t>(1)->get(1, 0) == 20);
    CHECK(f->getColumn<int64_t>(1)->get(2, 0) == 30);

    CHECK(f->getColumn<double>(2)->get(0, 0) == 0.5);
    CHECK(f->getColumn<double>(2)->get(1, 0) == 0.6);
    CHECK(f->getColumn<double>(2)->get(2, 0) == 0.7);

    DataObjectFactory::destroy(f);
}

// ----------------------------------------------------------------------------
// End-to-end through Read.h's dispatcher
// ----------------------------------------------------------------------------

TEST_CASE("ReadOrc, end-to-end through Read.h, DenseMatrix<double>", TAG_IO) {
    DenseMatrix<double> *m = nullptr;
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseDouble.orc";
    read(m, filename, /*ctx=*/nullptr);

    REQUIRE(m != nullptr);
    REQUIRE(m->getNumRows() == 2);
    REQUIRE(m->getNumCols() == 4);
    CHECK(m->get(0, 0) == -0.1);
    CHECK(m->get(1, 3) == 5.0);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, end-to-end through Read.h, Frame", TAG_IO) {
    Frame *f = nullptr;
    const char filename[] = "./test/runtime/local/io/ReadOrc_Frame.orc";
    read(f, filename, /*ctx=*/nullptr);

    REQUIRE(f != nullptr);
    REQUIRE(f->getNumRows() == 3);
    REQUIRE(f->getNumCols() == 3);
    CHECK(f->getColumn<double>(0)->get(0, 0) == 1.1);
    CHECK(f->getColumn<int64_t>(1)->get(2, 0) == 30);
    CHECK(f->getColumn<double>(2)->get(1, 0) == 0.6);

    DataObjectFactory::destroy(f);
}

// ----------------------------------------------------------------------------
// Error paths
// ----------------------------------------------------------------------------

TEST_CASE("ReadOrc, missing file", TAG_IO) {
    DenseMatrix<double> *m = DataObjectFactory::create<DenseMatrix<double>>(1, 1, false);
    FileMetaData fmd(1, 1, true, ValueTypeCode::F64);
    std::map<std::string, std::string> opts;
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&m), fmd, "/no/such/file.orc", opts, nullptr),
                      std::runtime_error);
    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, row count mismatch", TAG_IO) {
    DenseMatrix<double> *m = DataObjectFactory::create<DenseMatrix<double>>(99, 4, false);
    FileMetaData fmd(99, 4, true, ValueTypeCode::F64); // file actually has 2 rows
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseDouble.orc";
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, column count mismatch", TAG_IO) {
    DenseMatrix<double> *m = DataObjectFactory::create<DenseMatrix<double>>(2, 99, false);
    FileMetaData fmd(2, 99, true, ValueTypeCode::F64); // file actually has 4 cols
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseDouble.orc";
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, rejects string columns", TAG_IO) {
    DenseMatrix<double> *m = nullptr;
    FileMetaData fmd(2, 1, true, ValueTypeCode::STR);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringCol.orc";
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr), std::runtime_error);
}

TEST_CASE("ReadOrc, rejects nulls", TAG_IO) {
    DenseMatrix<double> *m = DataObjectFactory::create<DenseMatrix<double>>(2, 1, false);
    FileMetaData fmd(2, 1, true, ValueTypeCode::F64);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_HasNulls.orc";
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, type mismatch (file double, meta SI64)", TAG_IO) {
    DenseMatrix<int64_t> *m = DataObjectFactory::create<DenseMatrix<int64_t>>(2, 4, false);
    FileMetaData fmd(2, 4, true, ValueTypeCode::SI64);
    std::map<std::string, std::string> opts;
    // File ReadOrc_DenseDouble.orc has DOUBLE columns; meta requests SI64 → throw
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseDouble.orc";
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, unsupported value type F32", TAG_IO) {
    DenseMatrix<float> *m = DataObjectFactory::create<DenseMatrix<float>>(2, 4, false);
    FileMetaData fmd(2, 4, true, ValueTypeCode::F32);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseDouble.orc";
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(m);
}
