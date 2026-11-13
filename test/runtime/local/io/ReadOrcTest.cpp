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

TEST_CASE("ReadOrc, Frame with strings, mixed types", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR, ValueTypeCode::F64};
    std::string labelsArr[] = {"name", "age", "dept", "salary"};
    Frame *f = DataObjectFactory::create<Frame>(3, 4, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR, ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 3);
    REQUIRE(f->getNumCols() == 4);

    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "alice");
    CHECK(f->getColumn<std::string>(0)->get(1, 0) == "bob");
    CHECK(f->getColumn<std::string>(0)->get(2, 0) == "carol");

    CHECK(f->getColumn<int64_t>(1)->get(0, 0) == 30);
    CHECK(f->getColumn<int64_t>(1)->get(1, 0) == 41);
    CHECK(f->getColumn<int64_t>(1)->get(2, 0) == 29);

    CHECK(f->getColumn<std::string>(2)->get(0, 0) == "eng");
    CHECK(f->getColumn<std::string>(2)->get(1, 0) == "ops");
    CHECK(f->getColumn<std::string>(2)->get(2, 0) == "eng");

    CHECK(f->getColumn<double>(3)->get(0, 0) == 80.5);
    CHECK(f->getColumn<double>(3)->get(1, 0) == 95.0);
    CHECK(f->getColumn<double>(3)->get(2, 0) == 72.25);

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

TEST_CASE("ReadOrc, end-to-end through Read.h, Frame with strings", TAG_IO) {
    Frame *f = nullptr;
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";
    read(f, filename, /*ctx=*/nullptr);

    REQUIRE(f != nullptr);
    REQUIRE(f->getNumRows() == 3);
    REQUIRE(f->getNumCols() == 4);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "alice");
    CHECK(f->getColumn<int64_t>(1)->get(1, 0) == 41);
    CHECK(f->getColumn<std::string>(2)->get(2, 0) == "eng");
    CHECK(f->getColumn<double>(3)->get(0, 0) == 80.5);

    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, Frame single string column", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::STR};
    std::string labelsArr[] = {"s"};
    Frame *f = DataObjectFactory::create<Frame>(2, 1, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::STR};
    std::vector<std::string> labels{"s"};
    FileMetaData fmd(2, 1, false, schema, labels);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringCol.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 2);
    REQUIRE(f->getNumCols() == 1);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "alpha");
    CHECK(f->getColumn<std::string>(0)->get(1, 0) == "beta");

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

TEST_CASE("ReadOrc, type mismatch - file STRING, meta SI64", TAG_IO) {
    // ReadOrc_StringCol.orc has one STRING column; meta requests SI64 → throw.
    ValueTypeCode schemaArr[] = {ValueTypeCode::SI64};
    std::string labelsArr[] = {"s"};
    Frame *f = DataObjectFactory::create<Frame>(2, 1, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::SI64};
    std::vector<std::string> labels{"s"};
    FileMetaData fmd(2, 1, false, schema, labels);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringCol.orc";

    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, type mismatch - file DOUBLE, meta STR", TAG_IO) {
    // ReadOrc_DenseDouble.orc has DOUBLE columns; meta requests STR → throw.
    ValueTypeCode schemaArr[] = {ValueTypeCode::STR, ValueTypeCode::STR, ValueTypeCode::STR, ValueTypeCode::STR};
    std::string labelsArr[] = {"c0", "c1", "c2", "c3"};
    Frame *f = DataObjectFactory::create<Frame>(2, 4, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::STR, ValueTypeCode::STR, ValueTypeCode::STR};
    std::vector<std::string> labels{"c0", "c1", "c2", "c3"};
    FileMetaData fmd(2, 4, false, schema, labels);
    std::map<std::string, std::string> opts;
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseDouble.orc";

    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(f);
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

// ----------------------------------------------------------------------------
// Column projection
// ----------------------------------------------------------------------------

TEST_CASE("ReadOrc, project subset of Frame", TAG_IO) {
    // Project [age, salary] out of the 4-col fixture. Output shape is 3 x 2.
    ValueTypeCode schemaArr[] = {ValueTypeCode::SI64, ValueTypeCode::F64};
    std::string labelsArr[] = {"age", "salary"};
    Frame *f = DataObjectFactory::create<Frame>(3, 2, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::SI64, ValueTypeCode::F64};
    std::vector<std::string> labels{"age", "salary"};
    FileMetaData fmd(3, 2, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "age,salary"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 3);
    REQUIRE(f->getNumCols() == 2);
    CHECK(f->getColumn<int64_t>(0)->get(0, 0) == 30);
    CHECK(f->getColumn<int64_t>(0)->get(2, 0) == 29);
    CHECK(f->getColumn<double>(1)->get(0, 0) == 80.5);
    CHECK(f->getColumn<double>(1)->get(2, 0) == 72.25);

    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, project single column", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::STR};
    std::string labelsArr[] = {"name"};
    Frame *f = DataObjectFactory::create<Frame>(3, 1, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::STR};
    std::vector<std::string> labels{"name"};
    FileMetaData fmd(3, 1, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "name"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumCols() == 1);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "alice");
    CHECK(f->getColumn<std::string>(0)->get(1, 0) == "bob");
    CHECK(f->getColumn<std::string>(0)->get(2, 0) == "carol");

    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, project with reordering", TAG_IO) {
    // Request salary,name - output order matches request order.
    ValueTypeCode schemaArr[] = {ValueTypeCode::F64, ValueTypeCode::STR};
    std::string labelsArr[] = {"salary", "name"};
    Frame *f = DataObjectFactory::create<Frame>(3, 2, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::F64, ValueTypeCode::STR};
    std::vector<std::string> labels{"salary", "name"};
    FileMetaData fmd(3, 2, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "salary,name"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    CHECK(f->getColumn<double>(0)->get(0, 0) == 80.5);
    CHECK(f->getColumn<double>(0)->get(2, 0) == 72.25);
    CHECK(f->getColumn<std::string>(1)->get(0, 0) == "alice");
    CHECK(f->getColumn<std::string>(1)->get(2, 0) == "carol");

    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, project mixed string and numeric", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::STR, ValueTypeCode::SI64};
    std::string labelsArr[] = {"name", "age"};
    Frame *f = DataObjectFactory::create<Frame>(3, 2, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64};
    std::vector<std::string> labels{"name", "age"};
    FileMetaData fmd(3, 2, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "name,age"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    CHECK(f->getColumn<std::string>(0)->get(1, 0) == "bob");
    CHECK(f->getColumn<int64_t>(1)->get(1, 0) == 41);

    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, project whitespace tolerance", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::F64};
    std::string labelsArr[] = {"name", "age", "salary"};
    Frame *f = DataObjectFactory::create<Frame>(3, 3, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "salary"};
    FileMetaData fmd(3, 3, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "name, age , salary"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "alice");
    CHECK(f->getColumn<int64_t>(1)->get(2, 0) == 29);
    CHECK(f->getColumn<double>(2)->get(1, 0) == 95.0);

    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, project on DenseMatrix<double>", TAG_IO) {
    DenseMatrix<double> *m = DataObjectFactory::create<DenseMatrix<double>>(2, 2, false);

    FileMetaData fmd(2, 2, true, ValueTypeCode::F64);
    std::map<std::string, std::string> opts{{"columns", "c0,c2"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_DenseDouble.orc";

    readOrc(reinterpret_cast<void *>(&m), fmd, filename, opts, nullptr);

    REQUIRE(m->getNumRows() == 2);
    REQUIRE(m->getNumCols() == 2);
    CHECK(m->get(0, 0) == -0.1);
    CHECK(m->get(0, 1) == 0.1);
    CHECK(m->get(1, 0) == 3.14);
    CHECK(m->get(1, 1) == 6.22216);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadOrc, projection listing all columns identity-like", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::F64, ValueTypeCode::SI64, ValueTypeCode::F64};
    std::string labelsArr[] = {"a", "b", "c"};
    Frame *f = DataObjectFactory::create<Frame>(3, 3, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::F64, ValueTypeCode::SI64, ValueTypeCode::F64};
    std::vector<std::string> labels{"a", "b", "c"};
    FileMetaData fmd(3, 3, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "a,b,c"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_Frame.orc";

    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    CHECK(f->getColumn<double>(0)->get(0, 0) == 1.1);
    CHECK(f->getColumn<int64_t>(1)->get(2, 0) == 30);
    CHECK(f->getColumn<double>(2)->get(1, 0) == 0.6);

    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, projection references missing column", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::SI64};
    std::string labelsArr[] = {"nonesuch"};
    Frame *f = DataObjectFactory::create<Frame>(3, 1, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::SI64};
    std::vector<std::string> labels{"nonesuch"};
    FileMetaData fmd(3, 1, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "nonesuch"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, projection has duplicate names", TAG_IO) {
    // Frame labels must be unique, so use placeholders; the reader rejects
    // the duplicate in `options["columns"]` before ever inspecting the Frame.
    ValueTypeCode schemaArr[] = {ValueTypeCode::SI64, ValueTypeCode::SI64};
    std::string labelsArr[] = {"a", "b"};
    Frame *f = DataObjectFactory::create<Frame>(3, 2, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::SI64, ValueTypeCode::SI64};
    std::vector<std::string> labels{"a", "b"};
    FileMetaData fmd(3, 2, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "age,age"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, projection type mismatch", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::F64};
    std::string labelsArr[] = {"name"};
    Frame *f = DataObjectFactory::create<Frame>(3, 1, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::F64};
    std::vector<std::string> labels{"name"};
    FileMetaData fmd(3, 1, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "name"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, projection empty entry", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::SI64, ValueTypeCode::F64};
    std::string labelsArr[] = {"age", "salary"};
    Frame *f = DataObjectFactory::create<Frame>(3, 2, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::SI64, ValueTypeCode::F64};
    std::vector<std::string> labels{"age", "salary"};
    FileMetaData fmd(3, 2, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "age,,salary"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(f);
}

// ----------------------------------------------------------------------------
// Predicate push-down
// ----------------------------------------------------------------------------

TEST_CASE("ReadOrc, predicate age > 40 on Frame", TAG_IO) {
    // Fixture: name/age/dept/salary with rows alice/30, bob/41, carol/29.
    // Predicate age > 40 matches only bob.
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age > 40"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr; // reader allocates
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f != nullptr);
    REQUIRE(f->getNumRows() == 1);
    REQUIRE(f->getNumCols() == 4);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "bob");
    CHECK(f->getColumn<int64_t>(1)->get(0, 0) == 41);
    CHECK(f->getColumn<std::string>(2)->get(0, 0) == "ops");
    CHECK(f->getColumn<double>(3)->get(0, 0) == 95.0);

    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate age = 30", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age = 30"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 1);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "alice");
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate age != 30", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age != 30"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 2);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "bob");
    CHECK(f->getColumn<std::string>(0)->get(1, 0) == "carol");
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate age < 30", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age < 30"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 1);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "carol");
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate age <= 30", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age <= 30"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 2);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate age >= 41", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age >= 41"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 1);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "bob");
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate salary > 80.0 (float literal)", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "salary > 80.0"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 2);
    CHECK(f->getColumn<double>(3)->get(0, 0) == 80.5);
    CHECK(f->getColumn<double>(3)->get(1, 0) == 95.0);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate dept = 'eng' (string literal)", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "dept = 'eng'"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 2);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "alice");
    CHECK(f->getColumn<std::string>(0)->get(1, 0) == "carol");
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate dept != 'eng'", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "dept != 'eng'"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 1);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "bob");
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate matches zero rows", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age > 100"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f != nullptr);
    CHECK(f->getNumRows() == 0);
    CHECK(f->getNumCols() == 4);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate matches all rows", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age >= 0"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 3);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate combined with column projection", TAG_IO) {
    // Project [name, age], filter age > 40. Expect one row: bob, 41.
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64};
    std::vector<std::string> labels{"name", "age"};
    FileMetaData fmd(3, 2, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "name,age"}, {"predicate", "age > 40"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr);

    REQUIRE(f->getNumRows() == 1);
    REQUIRE(f->getNumCols() == 2);
    CHECK(f->getColumn<std::string>(0)->get(0, 0) == "bob");
    CHECK(f->getColumn<int64_t>(1)->get(0, 0) == 41);
    DataObjectFactory::destroy(f);
}

TEST_CASE("ReadOrc, predicate references column not in projection", TAG_IO) {
    // Project [name], filter on age (not projected) → throw.
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR};
    std::vector<std::string> labels{"name"};
    FileMetaData fmd(3, 1, false, schema, labels);
    std::map<std::string, std::string> opts{{"columns", "name"}, {"predicate", "age > 40"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
}

TEST_CASE("ReadOrc, predicate literal type mismatch (int on STR col)", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::STR};
    std::vector<std::string> labels{"name"};
    FileMetaData fmd(3, 1, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "name > 5"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
}

TEST_CASE("ReadOrc, predicate bad syntax (missing literal)", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::SI64};
    std::vector<std::string> labels{"age"};
    FileMetaData fmd(3, 1, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age >"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
}

TEST_CASE("ReadOrc, predicate bad syntax (unknown operator)", TAG_IO) {
    std::vector<ValueTypeCode> schema{ValueTypeCode::SI64};
    std::vector<std::string> labels{"age"};
    FileMetaData fmd(3, 1, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age ~ 40"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    Frame *f = nullptr;
    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
}

TEST_CASE("ReadOrc, predicate requires *res == nullptr", TAG_IO) {
    ValueTypeCode schemaArr[] = {ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR, ValueTypeCode::F64};
    std::string labelsArr[] = {"name", "age", "dept", "salary"};
    Frame *f = DataObjectFactory::create<Frame>(3, 4, schemaArr, labelsArr, false);

    std::vector<ValueTypeCode> schema{ValueTypeCode::STR, ValueTypeCode::SI64, ValueTypeCode::STR,
                                      ValueTypeCode::F64};
    std::vector<std::string> labels{"name", "age", "dept", "salary"};
    FileMetaData fmd(3, 4, false, schema, labels);
    std::map<std::string, std::string> opts{{"predicate", "age > 40"}};
    const char filename[] = "./test/runtime/local/io/ReadOrc_StringFrame.orc";

    REQUIRE_THROWS_AS(readOrc(reinterpret_cast<void *>(&f), fmd, filename, opts, nullptr), std::runtime_error);
    DataObjectFactory::destroy(f);
}
