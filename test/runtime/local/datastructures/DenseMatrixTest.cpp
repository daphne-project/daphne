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
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>

TEMPLATE_TEST_CASE("DenseMatrix allocates enough space", TAG_DATASTRUCTURES, ALL_VALUE_TYPES) {
    // No assertions in this test case. We just want to see if it runs without
    // crashing.

    using ValueType = TestType;

    const size_t numRows = 10000;
    const size_t numCols = 2000;

    DenseMatrix<ValueType> *m = DataObjectFactory::create<DenseMatrix<ValueType>>(numRows, numCols, false);

    ValueType *values = m->getValues();
    const size_t numCells = numRows * numCols;

    // Fill the matrix with ones of the respective value type.
    for (size_t i = 0; i < numCells; i++)
        values[i] = ValueType(1);

    DataObjectFactory::destroy(m);
}

TEST_CASE("DenseMatrix for strings", TAG_DATASTRUCTURES) {
    const size_t numRows = 3;
    const size_t numCols = 4;
    const size_t bytesPerCell = 1;

    using expectedStrings = const std::vector<std::string>;

    // We do not use operator== to compare to a matrix created by genGivenVals()
    // here, since this would rely on the functionality we want to test.
    auto compareMatToArr = [](const DenseMatrix<const char *> *mat, const expectedStrings &exp) {
        for (size_t r = 0; r < mat->getNumRows(); r++)
            for (size_t c = 0; c < mat->getNumCols(); c++)
                if (strcmp(mat->get(r, c), exp[r * mat->getNumCols() + c].c_str()))
                    return false;
        return true;
    };

    SECTION("Append") {
        auto m = DataObjectFactory::create<DenseMatrix<const char *>>(numRows, numCols, false);
        m->prepareAppend();
        expectedStrings exp = {"0", "", "", "3", "10", "", "", "13", "20", "", "", "23"};
        m->prepareAppend();
        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++)
                if (c % 3 == 0)
                    m->append(r, c, std::string(std::to_string(r * 10 + c)).c_str());

        m->finishAppend();
        CHECK(compareMatToArr(m, exp));
        DataObjectFactory::destroy(m);
    }

    SECTION("Set") {
        expectedStrings exp1 = {"", "1", "", "3", "", "11", "", "13", "", "21", "", "23"};

        expectedStrings exp2 = {"0", "1", "2", "3", "10", "11", "12", "13", "20", "21", "22", "23"};
        DenseMatrix<const char *> *m = DataObjectFactory::create<DenseMatrix<const char *>>(
            numRows, numCols, false, numRows * numCols * bytesPerCell);

        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++) {
                size_t num = r * 10 + c;
                if (num % 2) {
                    m->set(r, c, std::string(std::to_string(num)).c_str());
                }
            }
        CHECK(compareMatToArr(m, exp1));
        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++) {
                size_t num = r * 10 + c;
                if (!(num % 2)) {
                    m->set(r, c, std::string(std::to_string(num)).c_str());
                }
            }
        CHECK(compareMatToArr(m, exp2));
        DataObjectFactory::destroy(m);
    }

    SECTION("Append + Set") {

        expectedStrings exp1 = {"0", "", "", "3", "10", "", "", "13", "20", "", "", "23"};

        expectedStrings exp2 = {"0", "", "", "3", "10", std::string(100, 'O').c_str(), "", "13", "20", "", "", "23"};

        expectedStrings exp3 = {
            "0", "",  "", "3", "10", std::string(100, 'O').c_str(), "", "13", "20", std::string(5000, 'X').c_str(),
            "",  "23"};

        expectedStrings exp4 = {
            "0", "",  "", "3", "10", std::string(100, 'O').c_str(), "", "13", "20", std::string(5, 'X').c_str(),
            "",  "23"};

        auto m = DataObjectFactory::create<DenseMatrix<const char *>>(numRows, numCols, false,
                                                                      numRows * numCols * bytesPerCell);
        m->set(1, 1, std::string(20, 'C').c_str()); // will be overwritten by append
        m->prepareAppend();
        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++)
                if (c % 3 == 0)
                    m->append(r, c, std::string(std::to_string(r * 10 + c)).c_str());
        m->finishAppend();

        CHECK(compareMatToArr(m, exp1));
        m->set(1, 1, std::string(100, 'O').c_str());
        CHECK(compareMatToArr(m, exp2));
        m->set(2, 1, std::string(5000, 'X').c_str());
        CHECK(compareMatToArr(m, exp3));
        m->set(2, 1, std::string(5, 'X').c_str());
        CHECK(compareMatToArr(m, exp4));
        DataObjectFactory::destroy(m);
    }

    SECTION("View") {
        expectedStrings exp1 = {"1", "2", "11", "12"};
        expectedStrings exp2 = {"1", "2", "11", std::string(5, 'X').c_str()};
        expectedStrings exp3 = {"0",  "1",  "2",  "3",  "10", "11", std::string(5, 'X').c_str(),
                                "13", "20", "21", "22", "23"};
        DenseMatrix<const char *> *m = DataObjectFactory::create<DenseMatrix<const char *>>(
            numRows, numCols, false, numRows * numCols * bytesPerCell);
        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++)
                m->set(r, c, std::string(std::to_string(r * 10 + c)).c_str());
        auto mView = DataObjectFactory::create<DenseMatrix<const char *>>(m, 0, 2, 1, 3);
        CHECK(compareMatToArr(mView, exp1));

        mView->set(1, 1, std::string(5, 'X').c_str());
        CHECK(compareMatToArr(mView, exp2));
        CHECK(compareMatToArr(m, exp3));

        DataObjectFactory::destroy(m);
        DataObjectFactory::destroy(mView);
    }
}

TEST_CASE("DenseMatrix sub-matrix works properly", TAG_DATASTRUCTURES) {
    using ValueType = uint64_t;

    const size_t numRowsOrig = 10;
    const size_t numColsOrig = 7;
    const size_t numCellsOrig = numRowsOrig * numColsOrig;

    DenseMatrix<ValueType> *mOrig = DataObjectFactory::create<DenseMatrix<ValueType>>(numRowsOrig, numColsOrig, true);
    DenseMatrix<ValueType> *mSub = DataObjectFactory::create<DenseMatrix<ValueType>>(mOrig, 3, 5, 1, 4);

    // Sub-matrix dimensions are as expected.
    CHECK(mSub->getNumRows() == 2);
    CHECK(mSub->getNumCols() == 3);
    CHECK(mSub->getRowSkip() == numColsOrig);

    // Sub-matrix shares data array with original.
    ValueType *valuesOrig = mOrig->getValues();
    ValueType *valuesSub = mSub->getValues();
    CHECK((valuesSub >= valuesOrig && valuesSub < valuesOrig + numCellsOrig));
    valuesSub[0] = 123;
    CHECK(valuesOrig[3 * numColsOrig + 1] == 123);

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

TEMPLATE_TEST_CASE("DenseMatrix with string value type", TAG_DATASTRUCTURES, ALL_STRING_VALUE_TYPES) {
    using ValueType = TestType;

    using expectedStrings = const std::vector<ValueType>;

    // We do not use operator == to compare to a matrix created by genGivenVals()
    // here, since this would rely on the functionality we want to test.
    auto compareMatToArr = [](const DenseMatrix<ValueType> *mat, const expectedStrings &exp) {
        for (size_t r = 0; r < mat->getNumRows(); r++)
            for (size_t c = 0; c < mat->getNumCols(); c++)
                if ((mat->get(r, c) != exp[r * mat->getNumCols() + c]))
                    return false;
        return true;
    };

    SECTION("Create") {
        const size_t numRows = 10000;
        const size_t numCols = 2000;

        DenseMatrix<ValueType> *m = DataObjectFactory::create<DenseMatrix<ValueType>>(numRows, numCols, false);

        ValueType *values = m->getValues();
        const size_t numCells = numRows * numCols;

        for (size_t i = 0; i < numCells; i++)
            values[i] = ValueType();

        DataObjectFactory::destroy(m);
    }

    SECTION("Append") {
        const size_t numRows = 3;
        const size_t numCols = 4;

        DenseMatrix<ValueType> *m = DataObjectFactory::create<DenseMatrix<ValueType>>(numRows, numCols, false);
        expectedStrings exp = {ValueType("0"),  ValueType(""), ValueType(""), ValueType("3"),
                               ValueType("10"), ValueType(""), ValueType(""), ValueType("13"),
                               ValueType("20"), ValueType(""), ValueType(""), ValueType("23")};
        m->prepareAppend();

        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++)
                if (c % 3 == 0)
                    m->append(r, c, ValueType(std::to_string(r * 10 + c).c_str()));

        m->finishAppend();

        CHECK(compareMatToArr(m, exp));

        DataObjectFactory::destroy<DenseMatrix<ValueType>>(m);
    }

    SECTION("Set") {
        const size_t numRows = 3;
        const size_t numCols = 4;
        expectedStrings exp1 = {ValueType(""), ValueType("1"),  ValueType(""), ValueType("3"),
                                ValueType(""), ValueType("11"), ValueType(""), ValueType("13"),
                                ValueType(""), ValueType("21"), ValueType(""), ValueType("23")};

        expectedStrings exp2 = {ValueType("0"),  ValueType("1"),  ValueType("2"),  ValueType("3"),
                                ValueType("10"), ValueType("11"), ValueType("12"), ValueType("13"),
                                ValueType("20"), ValueType("21"), ValueType("22"), ValueType("23")};
        DenseMatrix<ValueType> *m = DataObjectFactory::create<DenseMatrix<ValueType>>(numRows, numCols, false);

        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++) {
                size_t num = r * 10 + c;
                if (num % 2) {
                    m->set(r, c, ValueType(std::to_string(num).c_str()));
                }
            }
        CHECK(compareMatToArr(m, exp1));

        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++) {
                size_t num = r * 10 + c;
                if (!(num % 2)) {
                    m->set(r, c, ValueType(std::to_string(num).c_str()));
                }
            }
        CHECK(compareMatToArr(m, exp2));

        DataObjectFactory::destroy(m);
    }

    SECTION("View") {
        const size_t numRows = 3;
        const size_t numCols = 4;
        expectedStrings exp1 = {ValueType("1"), ValueType("2"), ValueType("11"), "12"};
        expectedStrings exp2 = {ValueType("1"), ValueType("2"), ValueType("11"),
                                ValueType(std::string(5, 'X').c_str())};
        expectedStrings exp3 = {ValueType("0"),
                                ValueType("1"),
                                ValueType("2"),
                                ValueType("3"),
                                ValueType("10"),
                                ValueType("11"),
                                ValueType(std::string(5, 'X').c_str()),
                                ValueType("13"),
                                ValueType("20"),
                                ValueType("21"),
                                ValueType("22"),
                                ValueType("23")};
        DenseMatrix<ValueType> *m = DataObjectFactory::create<DenseMatrix<ValueType>>(numRows, numCols, false);
        for (size_t r = 0; r < numRows; r++)
            for (size_t c = 0; c < numCols; c++)
                m->set(r, c, ValueType(std::to_string(r * 10 + c).c_str()));
        auto mView = DataObjectFactory::create<DenseMatrix<ValueType>>(m, 0, 2, 1, 3);
        CHECK(compareMatToArr(mView, exp1));

        mView->set(1, 1, ValueType(std::string(5, 'X').c_str()));
        CHECK(compareMatToArr(mView, exp2));

        CHECK(compareMatToArr(m, exp3));

        DataObjectFactory::destroy(m);
        DataObjectFactory::destroy(mView);
    }
}

TEMPLATE_TEST_CASE("DenseMatrix::getPhysicalSizeByte()", TAG_DATASTRUCTURES, double, int32_t) {
    using VT = TestType;
    using DT = DenseMatrix<VT>;

    DT *m = nullptr;
    size_t exp = -1;

    SECTION("0x0") {
        m = DataObjectFactory::create<DT>(0, 0, true);
        exp = 0;
    }
    SECTION("3x0") {
        m = DataObjectFactory::create<DT>(3, 0, true);
        exp = 0;
    }
    SECTION("0x3") {
        m = DataObjectFactory::create<DT>(0, 3, true);
        exp = 0;
    }
    SECTION("3x3, all zeros") {
        m = DataObjectFactory::create<DT>(3, 3, true);
        exp = 3 * 3 * sizeof(VT);
    }
    SECTION("3x3, all non-zeros") {
        m = genGivenVals<DT>(3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
        exp = 3 * 3 * sizeof(VT);
    }

    CHECK(m->getPhysicalSizeByte() == exp);

    DataObjectFactory::destroy(m);
}