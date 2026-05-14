/*
 * Copyright 2022 The DAPHNE Consortium
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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/io/ReadMM.h>

#include <tags.h>

#include <catch.hpp>

#include <cmath>
#include <cstdint>

#define DATA_TYPES_MAT DenseMatrix, CSRMatrix
#define VALUE_TYPES_INT int64_t, int32_t
#define VALUE_TYPES_FLT double

// TODO Currently, we test only a subset of the format/field/symmetry combinations supported by the MatrixMarket file
// format. We should support all (except for complex values as long as DAPHNE does not support a complex value type).
// - Current order of test cases: CIG, AIG, CRG, CRS, CRK, CPS, AIK, AIS
// - Desired order of test cases: CIG, +CIS, +CIK, CRG, CRS, CRK, AIG, AIK, AIS, +ARG, +ARK, +ARS, CPS, +CPG, +CPK
//   (combinations with + currently lack a test case)

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CIG", TAG_IO, (DATA_TYPES_MAT), (VALUE_TYPES_INT)) {
    // coordinate integer general

    using DT = TestType;

    DT *res = nullptr;

    SECTION("small file") {
        DT *exp = genGivenVals<DT>(9, {
                                          1, 0, 0, 0, 0, 0, 3, 0, 1, //
                                          2, 9, 4, 0, 0, 0, 4, 0, 2, //
                                          0, 1, 5, 0, 0, 0, 5, 0, 3, //
                                          3, 0, 6, 3, 9, 6, 0, 0, 0, //
                                          4, 0, 7, 4, 1, 7, 0, 0, 0, //
                                          5, 0, 8, 5, 2, 8, 0, 0, 0, //
                                          6, 0, 9, 6, 3, 9, 0, 0, 0, //
                                          7, 2, 1, 7, 4, 1, 6, 8, 4, //
                                          8, 3, 2, 8, 5, 2, 7, 9, 5, //
                                      });

        readMM(res, "./test/runtime/local/io/cig.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 9);
        CHECK(res->getNumCols() == 9);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 50);

        // Check the data.
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp);
    }
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIG", TAG_IO, (DATA_TYPES_MAT), (VALUE_TYPES_INT)) {
    // array integer general

    using DT = TestType;

    DT *res = nullptr;

    SECTION("small file") {
        DT *exp = genGivenVals<DT>(4, {
                                          1, 5, 9,  //
                                          2, 6, 10, //
                                          3, 7, 11, //
                                          4, 8, 12, //
                                      });

        readMM(res, "./test/runtime/local/io/aig.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 4);
        CHECK(res->getNumCols() == 3);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 12);

        // Check the data.
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp);
    }

    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRG", TAG_IO, (DATA_TYPES_MAT), (VALUE_TYPES_FLT)) {
    // coordinate real general

    using DT = TestType;

    DT *res = nullptr;

    SECTION("small file") {
        DT *exp = genGivenVals<DT>(4, {
                                          1.1, 0, 1.3, 0, //
                                          2.1, 0, 2.3, 0, //
                                          0, 0, 0, 0,     //
                                          0, 0, 0, 4.4,   //
                                      });

        readMM(res, "./test/runtime/local/io/crg_small.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 4);
        CHECK(res->getNumCols() == 4);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 5);

        // Check the data.
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp);
    }
    SECTION("large file") {
        readMM(res, "./test/runtime/local/io/crg.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 497);
        CHECK(res->getNumCols() == 507);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 53403);

        // Check a sample of the data.
        CHECK(res->get(6 - 1, 1 - 1) == 2.5599762000000e-01);     // 1st entry (line 3)
        CHECK(res->get(296 - 1, 49 - 1) == 4.7000942000000e-01);  // (line  5000)
        CHECK(res->get(10 - 1, 98 - 1) == 2.3929999000000e-01);   // (line 10000)
        CHECK(res->get(479 - 1, 159 - 1) == 9.0002853000000e-01); // (line 15000)
        CHECK(res->get(293 - 1, 215 - 1) == 7.0073879000000e-01); // (line 20000)
        CHECK(res->get(151 - 1, 266 - 1) == 8.0002236000000e-01); // (line 25000)
        CHECK(res->get(490 - 1, 313 - 1) == 4.0002668000000e-01); // (line 30000)
        CHECK(res->get(284 - 1, 354 - 1) == 1.0000938000000e-01); // (line 35000)
        CHECK(res->get(478 - 1, 395 - 1) == 7.0001853000000e-01); // (line 40000)
        CHECK(res->get(429 - 1, 437 - 1) == 9.9998951000000e-02); // (line 45000)
        CHECK(res->get(195 - 1, 482 - 1) == 1.9997281000000e-01); // (line 50000)
        CHECK(res->get(493 - 1, 507 - 1) == 1.3226395000000e-01); // last entry (line 53405)
    }

    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRS", TAG_IO, (DATA_TYPES_MAT), (VALUE_TYPES_FLT)) {
    // coordinate real symmetric

    using DT = TestType;

    DT *res = nullptr;

    SECTION("small file") {
        DT *exp = genGivenVals<DT>(4, {
                                          1.1, 0, 3.1, 0, //
                                          0, 2.2, 0, 0,   //
                                          3.1, 0, 0, 4.3, //
                                          0, 0, 4.3, 0,   //
                                      });

        readMM(res, "./test/runtime/local/io/crs_small.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 4);
        CHECK(res->getNumCols() == 4);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 6);

        // Check the data.
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp);
    }
    SECTION("large file") {
        size_t numRows = 66;
        size_t numCols = 66;

        readMM(res, "./test/runtime/local/io/crs.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 66);
        CHECK(res->getNumCols() == 66);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 2211 * 2 - numRows);

        // Check a sample of the data.
        CHECK(res->get(1 - 1, 1 - 1) == 1.9903332861200e+03);    // 1st entry (line 3)
        CHECK(res->get(6 - 1, 4 - 1) == -1.5330130341600e+03);   // (line 200)
        CHECK(res->get(23 - 1, 7 - 1) == 7.1417983692700e+01);   // (line 400)
        CHECK(res->get(49 - 1, 10 - 1) == 1.6295453843900e-02);  // (line 600)
        CHECK(res->get(31 - 1, 14 - 1) == -8.2139131759200e+01); // (line 800)
        CHECK(res->get(29 - 1, 18 - 1) == -3.8898712339300e+02); // (line 1000)
        CHECK(res->get(43 - 1, 22 - 1) == 4.6121380640700e-01);  // (line 1200)
        CHECK(res->get(33 - 1, 27 - 1) == -8.7815773845000e-01); // (line 1400)
        CHECK(res->get(48 - 1, 32 - 1) == 8.3887975060100e+01);  // (line 1600)
        CHECK(res->get(59 - 1, 38 - 1) == -9.1999258137900e+02); // (line 1800)
        CHECK(res->get(63 - 1, 46 - 1) == 3.8538772376900e+02);  // (line 2000)
        CHECK(res->get(63 - 1, 62 - 1) == -3.8624022300600e+02); // (line 2200)
        CHECK(res->get(66 - 1, 66 - 1) == 1.3630769148600e+03);  // last entry (line 2213)

        // Check if the data is symmetric.
        for (size_t r = 0; r < numRows; r++)
            for (size_t c = r + 1; c < numCols; c++)
                // Use REQUIRE here to prevent a huge output in case something is generally wrong.
                REQUIRE(res->get(r, c) == res->get(c, r));
    }

    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CRK", TAG_IO, (DATA_TYPES_MAT), (VALUE_TYPES_FLT)) {
    // coordinate real skew-symmetric

    using DT = TestType;

    DT *res = nullptr;

    SECTION("small file") {
        DT *exp = genGivenVals<DT>(4, {
                                          0, 0, -3.1, 0,  //
                                          0, 0, 0, 0,     //
                                          3.1, 0, 0, 4.3, //
                                          0, 0, -4.3, 0,  //
                                      });

        readMM(res, "./test/runtime/local/io/crk_small.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 4);
        CHECK(res->getNumCols() == 4);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 4);

        // Check the data.
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp);
    }
    SECTION("large file") {
        size_t numRows = 66;
        size_t numCols = 66;

        readMM(res, "./test/runtime/local/io/crk.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == numRows);
        CHECK(res->getNumCols() == numCols);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 2145 * 2);

        // Check a sample of the data.
        CHECK(res->get(2 - 1, 1 - 1) == 5.6791217991800e+02);    // 1st entry (line 3)
        CHECK(res->get(10 - 1, 4 - 1) == -1.1304774307100e+00);  // (line 200)
        CHECK(res->get(30 - 1, 7 - 1) == -1.1455084216200e-01);  // (line 400)
        CHECK(res->get(59 - 1, 10 - 1) == -4.6895762181000e-03); // (line 600)
        CHECK(res->get(45 - 1, 14 - 1) == 1.6289601701600e-01);  // (line 800)
        CHECK(res->get(47 - 1, 18 - 1) == 1.3346243602800e-01);  // (line 1000)
        CHECK(res->get(65 - 1, 22 - 1) == -8.6519467288300e-02); // (line 1200)
        CHECK(res->get(60 - 1, 27 - 1) == 6.9743808101800e-02);  // (line 1400)
        CHECK(res->get(47 - 1, 33 - 1) == 8.4517672432100e+01);  // (line 1600)
        CHECK(res->get(44 - 1, 40 - 1) == 6.1351305728000e-01);  // (line 1800)
        CHECK(res->get(55 - 1, 49 - 1) == -1.3314176264900e-02); // (line 2000)
        CHECK(res->get(66 - 1, 65 - 1) == -3.1481901065800e-15); // last entry (line 2147)

        // Check if the data is skew-symmetric.
        for (size_t r = 0; r < numRows; r++) {
            // Use REQUIRE here to prevent a huge output in case something is generally wrong.
            REQUIRE(res->get(r, r) == 0);
            for (size_t c = r + 1; c < numCols; c++)
                // Use REQUIRE here to prevent a huge output in case something is generally wrong.
                REQUIRE(res->get(r, c) == -res->get(c, r));
        }
    }

    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM CPS", TAG_IO, (DATA_TYPES_MAT), (VALUE_TYPES_INT)) {
    // coordinate pattern symmetric

    using DT = TestType;

    DT *res = nullptr;

    SECTION("small file") {
        DT *exp = genGivenVals<DT>(5, {
                                          0, 0, 1, 1, 0, //
                                          0, 1, 0, 0, 0, //
                                          1, 0, 0, 0, 0, //
                                          1, 0, 0, 1, 1, //
                                          0, 0, 0, 1, 0, //
                                      });

        readMM(res, "./test/runtime/local/io/cps_small.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 5);
        CHECK(res->getNumCols() == 5);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 8);

        // Check the data.
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp);
    }
    SECTION("large file") {
        size_t numRows = 24;
        size_t numCols = 24;

        readMM(res, "./test/runtime/local/io/cps.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == numRows);
        CHECK(res->getNumCols() == numCols);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 92 * 2 - numRows);

        // Check a sample of the data.
        CHECK(res->get(1 - 1, 1 - 1) == 1);   // 1st entry (line 3)
        CHECK(res->get(20 - 1, 1 - 1) == 1);  // (line 10)
        CHECK(res->get(12 - 1, 3 - 1) == 1);  // (line 20)
        CHECK(res->get(5 - 1, 5 - 1) == 1);   // (line 30)
        CHECK(res->get(18 - 1, 6 - 1) == 1);  // (line 40)
        CHECK(res->get(16 - 1, 8 - 1) == 1);  // (line 50)
        CHECK(res->get(18 - 1, 10 - 1) == 1); // (line 60)
        CHECK(res->get(24 - 1, 12 - 1) == 1); // (line 70)
        CHECK(res->get(18 - 1, 18 - 1) == 1); // (line 80)
        CHECK(res->get(23 - 1, 21 - 1) == 1); // (line 90)
        CHECK(res->get(24 - 1, 24 - 1) == 1); // last entry (line 94)

        // Check if the data is symmetric and if all values are either 0 or 1.
        for (size_t r = 0; r < numRows; r++)
            for (size_t c = r + 1; c < numCols; c++) {
                typename DT::VT v = res->get(r, c);
                // Use REQUIRE here to prevent a huge output in case something is generally wrong.
                REQUIRE((v == 0 || v == 1));
                REQUIRE(res->get(c, r) == v);
            }
    }

    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIK", TAG_IO, (DATA_TYPES_MAT), (VALUE_TYPES_INT)) {
    // array integer skew-symmetric

    using DT = TestType;

    DT *res = nullptr;

    SECTION("small file") {
        DT *exp = genGivenVals<DT>(4, {
                                          0, -1, -2, -3, //
                                          1, 0, -4, -5,  //
                                          2, 4, 0, -6,   //
                                          3, 5, 6, 0,    //
                                      });

        readMM(res, "./test/runtime/local/io/aik.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 4);
        CHECK(res->getNumCols() == 4);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 12);

        // Check the data.
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp);
    }
    DataObjectFactory::destroy(res);
}

TEMPLATE_PRODUCT_TEST_CASE("ReadMM AIS", TAG_IO, (DATA_TYPES_MAT), (VALUE_TYPES_INT)) {
    // array integer symmetric

    using DT = TestType;

    DT *res = nullptr;

    SECTION("small file") {
        DT *exp = genGivenVals<DT>(3, {
                                          1, 2, 3, //
                                          2, 4, 5, //
                                          3, 5, 6, //
                                      });

        readMM(res, "./test/runtime/local/io/ais.mtx");

        // Check the meta data.
        CHECK(res->getNumRows() == 3);
        CHECK(res->getNumCols() == 3);
        if constexpr (std::is_same_v<DT, CSRMatrix<typename DT::VT>>)
            CHECK(res->getNumNonZeros() == 9);

        // Check the data.
        CHECK(*res == *exp);

        DataObjectFactory::destroy(exp);
    }
    DataObjectFactory::destroy(res);
}

TEST_CASE("ReadMM CIG (Frame)", TAG_IO) {
    using DT = Frame;
    DT *m = nullptr;

    size_t numRows = 9;
    size_t numCols = 9;

    char filename[] = "./test/runtime/local/io/cig.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<int64_t>(0)->get(0, 0) == 1);
    CHECK(m->getColumn<int64_t>(0)->get(2, 0) == 0);
    CHECK(m->getColumn<int64_t>(4)->get(3, 0) == 9);
    CHECK(m->getColumn<int64_t>(4)->get(7, 0) == 4);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM AIG (Frame)", TAG_IO) {
    using DT = Frame;
    DT *m = nullptr;

    size_t numRows = 4;
    size_t numCols = 3;

    char filename[] = "./test/runtime/local/io/aig.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<int64_t>(0)->get(0, 0) == 1);
    CHECK(m->getColumn<int64_t>(0)->get(1, 0) == 2);
    CHECK(m->getColumn<int64_t>(1)->get(0, 0) == 5);
    CHECK(m->getColumn<int64_t>(2)->get(3, 0) == 12);
    CHECK(m->getColumn<int64_t>(1)->get(2, 0) == 7);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CRG (Frame)", TAG_IO) {
    using DT = Frame;
    DT *m = nullptr;

    size_t numRows = 497;
    size_t numCols = 507;

    char filename[] = "./test/runtime/local/io/crg.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<double>(0)->get(5, 0) == 0.25599762);
    CHECK(m->getColumn<double>(0)->get(6, 0) == 0.13827993);
    CHECK(m->getColumn<double>(4)->get(200, 0) == 0.20001954);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CRS (Frame)", TAG_IO) {
    using DT = Frame;
    DT *m = nullptr;

    size_t numRows = 66;
    size_t numCols = 66;

    char filename[] = "./test/runtime/local/io/crs.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<double>(29)->get(36, 0) == 926.188986068);

    for (size_t r = 0; r < numRows; r++)
        for (size_t c = r + 1; c < numCols; c++)
            CHECK(m->getColumn<double>(c)->get(r, 0) == m->getColumn<double>(r)->get(c, 0));

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CRK (Frame)", TAG_IO) {
    using DT = Frame;
    DT *m = nullptr;

    size_t numRows = 66;
    size_t numCols = 66;

    char filename[] = "./test/runtime/local/io/crk.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<double>(36)->get(29, 0) == -926.188986068);

    for (size_t r = 0; r < numRows; r++) {
        CHECK(m->getColumn<double>(r)->get(r, 0) == 0);
        for (size_t c = r + 1; c < numCols; c++)
            CHECK(m->getColumn<double>(c)->get(r, 0) == -m->getColumn<double>(r)->get(c, 0));
    }

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM CPS (Frame)", TAG_IO) {
    using DT = Frame;
    DT *m = nullptr;

    size_t numRows = 24;
    size_t numCols = 24;

    char filename[] = "./test/runtime/local/io/cps.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<double>(0)->get(0, 0) != 0);
    CHECK(m->getColumn<double>(0)->get(1, 0) == 0);
    CHECK(m->getColumn<double>(15)->get(3, 0) != 0);

    for (size_t r = 0; r < numRows; r++)
        for (size_t c = r + 1; c < numCols; c++)
            if (m->getColumn<double>(c)->get(r, 0) == 0)
                CHECK(m->getColumn<double>(r)->get(c, 0) == 0);
            else
                CHECK(m->getColumn<double>(r)->get(c, 0) != 0);

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM AIK (Frame)", TAG_IO) {
    using DT = Frame;
    DT *m = nullptr;

    size_t numRows = 4;
    size_t numCols = 4;

    char filename[] = "./test/runtime/local/io/aik.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<int64_t>(0)->get(1, 0) == 1);

    for (size_t r = 0; r < numRows; r++) {
        CHECK(m->getColumn<int64_t>(r)->get(r, 0) == 0);
        for (size_t c = r + 1; c < numCols; c++)
            CHECK(m->getColumn<int64_t>(c)->get(r, 0) == -m->getColumn<int64_t>(r)->get(c, 0));
    }

    DataObjectFactory::destroy(m);
}

TEST_CASE("ReadMM AIS (Frame)", TAG_IO) {
    using DT = Frame;
    DT *m = nullptr;

    size_t numRows = 3;
    size_t numCols = 3;

    char filename[] = "./test/runtime/local/io/ais.mtx";
    readMM(m, filename);

    REQUIRE(m->getNumRows() == numRows);
    REQUIRE(m->getNumCols() == numCols);

    CHECK(m->getColumn<int64_t>(1)->get(1, 0) == 4);

    for (size_t r = 0; r < numRows; r++)
        for (size_t c = r + 1; c < numCols; c++)
            CHECK(m->getColumn<int64_t>(c)->get(r, 0) == m->getColumn<int64_t>(r)->get(c, 0));

    DataObjectFactory::destroy(m);
}
