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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE("CSRMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES) {
    // No assertions in this test case. We just want to see if it runs without
    // crashing.

    using ValueType = TestType;

    const size_t numRows = 10000;
    const size_t numCols = 2000;
    const size_t numNonZeros = 500;

    CSRMatrix<ValueType> *m = DataObjectFactory::create<CSRMatrix<ValueType>>(numRows, numCols, numNonZeros, false);

    ValueType *values = m->getValues();
    size_t *colIdxs = m->getColIdxs();
    size_t *rowOffsets = m->getRowOffsets();

    // Fill all arrays with ones of the respective type. Note that this does
    // not result in a valid CSR representation, but we only want to check if
    // there is enough space.
    for (size_t i = 0; i < numNonZeros; i++) {
        values[i] = ValueType(1);
        colIdxs[i] = size_t(1);
    }
    for (size_t i = 0; i <= numRows; i++)
        rowOffsets[i] = size_t(1);

    DataObjectFactory::destroy(m);
}

TEST_CASE("CSRMatrix sub-matrix works properly", TAG_DATASTRUCTURES) {
    using ValueType = uint64_t;

    const size_t numRowsOrig = 10;
    const size_t numColsOrig = 7;
    const size_t numNonZeros = 3;

    CSRMatrix<ValueType> *mOrig =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsOrig, numColsOrig, numNonZeros, true);
    CSRMatrix<ValueType> *mSub = DataObjectFactory::create<CSRMatrix<ValueType>>(mOrig, 3, 5);

    // Sub-matrix dimensions are as expected.
    CHECK(mSub->getNumRows() == 2);
    CHECK(mSub->getNumCols() == numColsOrig);

    // Sub-matrix shares data array with original.
    CHECK(mSub->getValues() == mOrig->getValues());
    CHECK(mSub->getColIdxs() == mOrig->getColIdxs());

    ValueType *rowOffsetsOrig = mOrig->getRowOffsets();
    ValueType *rowOffsetsSub = mSub->getRowOffsets();
    CHECK((rowOffsetsSub >= rowOffsetsOrig && rowOffsetsSub <= rowOffsetsOrig + numRowsOrig));
    rowOffsetsSub[0] = 123;
    CHECK(rowOffsetsOrig[3] == 123);

    // Freeing both matrices does not result in double-free errors.
    SECTION("Freeing the original matrix first is fine") {
        DataObjectFactory::destroy(mOrig);
        DataObjectFactory::destroy(mSub);
    }
    SECTION("Freeing the sub-matrix first is fine") {
        DataObjectFactory::destroy(mSub);
        DataObjectFactory::destroy(mOrig);
    }
}

TEMPLATE_TEST_CASE("CSRMatrix validity check, valid matrices", TAG_DATASTRUCTURES, double, int32_t) {
    using VT = TestType;
    using DT = CSRMatrix<VT>;

    DT *o = genGivenVals<DT>(5, {0, 1, 0, 0, 2, //
                                 3, 0, 0, 0, 0, //
                                 0, 0, 0, 4, 0, //
                                 0, 0, 0, 0, 0, //
                                 0, 5, 0, 0, 0});
    DT *m = nullptr;

    SECTION("0x0, non-view") { m = DataObjectFactory::create<DT>(0, 0, 0, true); }
    SECTION("5x0, non-view") { m = DataObjectFactory::create<DT>(5, 0, 0, true); }
    SECTION("0x5, non-view") { m = DataObjectFactory::create<DT>(0, 5, 0, true); }
    SECTION("5x5, non-view") { m = o; }
    SECTION("2x5, view (row-segment, top)") {
        m = o->sliceRow(0, 2);
        DataObjectFactory::destroy(o);
    }
    SECTION("1x5, view (row-segment, middle)") {
        m = o->sliceRow(2, 3);
        DataObjectFactory::destroy(o);
    }
    SECTION("2x5, view (row-segment, bottom)") {
        m = o->sliceRow(3, 5);
        DataObjectFactory::destroy(o);
    }

    m->checkValidity();

    DataObjectFactory::destroy(o, m);
}

TEMPLATE_TEST_CASE("CSRMatrix validity check, invalid matrices", TAG_DATASTRUCTURES, double, int32_t) {
    using VT = TestType;
    using DT = CSRMatrix<VT>;

    // Start with a valid 3x5 CSRMatrix...

    DT *m = genGivenVals<DT>(3, {1, 0, 2, 0, 3, //
                                 0, 0, 4, 5, 6, //
                                 0, 0, 0, 0, 0});
    // rowOffsets: [0, 3, 6, 6]
    // values: [1, 2, 3, 4, 5, 6]
    // colIdxs: [0, 2, 4, 2, 3, 4]

    // ...and make it invalid in some way.

    SECTION("the first row offset is not zero (and the CSRMatrix is not a view with rows allocated before it)") {
        m->getRowOffsets()[0] = 42;
        // rowOffsets: [42, 3, 6, 6]
    }
    SECTION("the row offsets are not an increasing sequence") {
        m->getRowOffsets()[2] = 0;
        // rowOffsets: [0, 3, 0, 6]
    }
    SECTION("the row offsets indicate that some row has more non-zeros than columns") {
        m->getRowOffsets()[1] = 6;
        // rowOffsets: [0, 6, 6, 6]
    }
    SECTION("the row offsets indicate more non-zeros than allocated") {
        m->getRowOffsets()[3] = 100;
        // rowOffsets: [0, 3, 6, 100]
    }
    SECTION("explicitly stored zero values") {
        m->getValues(1)[1] = 0;
        // values: [1, 2, 3, 4, 0, 6]
    }
    SECTION("out-of-bounds column index") {
        m->getColIdxs(0)[2] = 100;
        // colIdxs: [0, 2, 100, 2, 3, 4]
    }
    SECTION("unsorted column indexes within a row") {
        m->getColIdxs(0)[1] = 4;
        m->getColIdxs(0)[2] = 2;
        // colIdxs: [0, 4, 2, 2, 3, 4]
    }
    SECTION("non-unique column indexes within a row") {
        m->getColIdxs(1)[1] = 4;
        // colIdxs: [0, 2, 4, 2, 4, 4]
    }

    CHECK_THROWS(m->checkValidity());

    DataObjectFactory::destroy(m);
}

TEMPLATE_TEST_CASE("CSRMatrix::getPhysicalSizeByte()", TAG_DATASTRUCTURES, double, int32_t) {
    using VT = TestType;
    using DT = CSRMatrix<VT>;

    DT *m = nullptr;
    size_t exp = -1;

    SECTION("0x0") {
        m = DataObjectFactory::create<DT>(0, 0, 0, true);
        exp = sizeof(size_t);
    }
    SECTION("3x0") {
        m = DataObjectFactory::create<DT>(3, 0, 0, true);
        exp = 4 * sizeof(size_t);
    }
    SECTION("0x3") {
        m = DataObjectFactory::create<DT>(0, 3, 0, true);
        exp = sizeof(size_t);
    }
    SECTION("3x3, all zeros (no space allocated)") {
        m = DataObjectFactory::create<DT>(3, 3, 0, true);
        exp = 4 * sizeof(size_t);
    }
    SECTION("3x3, all zeros (much space allocated)") {
        // The additional allocated space should not have an impact on the physical size.
        m = DataObjectFactory::create<DT>(3, 3, 100, true);
        exp = 4 * sizeof(size_t);
    }
    SECTION("3x3, some non-zeros") {
        m = genGivenVals<DT>(3, {1, 2, 0, 3, 4, 0, 0, 5, 0});
        exp = 4 * sizeof(size_t) + 5 * (sizeof(VT) + sizeof(size_t));
    }

    CHECK(m->getPhysicalSizeByte() == exp);

    DataObjectFactory::destroy(m);
}