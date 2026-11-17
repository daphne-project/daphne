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
#include <runtime/local/kernels/CheckEq.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

#include <cstdint>

TEMPLATE_PRODUCT_TEST_CASE("Matrix.get()", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix), (double, uint32_t)) {
    using DT = TestType;

    SECTION("empty matrix") {
        auto m = genGivenVals<DT>(3, {
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                     });

        for (size_t r = 0; r < 3; r++)
            for (size_t c = 0; c < 3; c++)
                CHECK(m->get(r, c) == 0);

        DataObjectFactory::destroy(m);
    }
    SECTION("sparse matrix") {
        auto m = genGivenVals<DT>(3, {
                                         0,
                                         0,
                                         1,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         2,
                                         0,
                                         0,
                                     });

        CHECK(m->get(0, 0) == 0);
        CHECK(m->get(0, 1) == 0);
        CHECK(m->get(0, 2) == 1);
        CHECK(m->get(0, 3) == 0);
        CHECK(m->get(1, 0) == 0);
        CHECK(m->get(1, 1) == 0);
        CHECK(m->get(1, 2) == 0);
        CHECK(m->get(1, 3) == 0);
        CHECK(m->get(2, 0) == 0);
        CHECK(m->get(2, 1) == 2);
        CHECK(m->get(2, 2) == 0);
        CHECK(m->get(2, 3) == 0);

        DataObjectFactory::destroy(m);
    }
    SECTION("full matrix") {
        auto m = genGivenVals<DT>(3, {
                                         6,
                                         7,
                                         3,
                                         8,
                                         9,
                                         2,
                                         10,
                                         4,
                                         1,
                                         11,
                                         5,
                                         12,
                                     });

        CHECK(m->get(0, 0) == 6);
        CHECK(m->get(0, 1) == 7);
        CHECK(m->get(0, 2) == 3);
        CHECK(m->get(0, 3) == 8);
        CHECK(m->get(1, 0) == 9);
        CHECK(m->get(1, 1) == 2);
        CHECK(m->get(1, 2) == 10);
        CHECK(m->get(1, 3) == 4);
        CHECK(m->get(2, 0) == 1);
        CHECK(m->get(2, 1) == 11);
        CHECK(m->get(2, 2) == 5);
        CHECK(m->get(2, 3) == 12);

        DataObjectFactory::destroy(m);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.get() on view", TAG_DATASTRUCTURES, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;

    auto m = genGivenVals<DT>(4, {
                                     1,
                                     2,
                                     3,
                                     4,
                                     5,
                                     6,
                                     7,
                                     8,
                                     9,
                                     10,
                                     11,
                                     12,
                                     13,
                                     14,
                                     15,
                                     16,
                                 });

    SECTION("row range") {
        auto view = DataObjectFactory::create<DT>(m, 1, 3, 0, 4);
        CHECK(view->get(0, 0) == 5);
        CHECK(view->get(0, 1) == 6);
        CHECK(view->get(0, 2) == 7);
        CHECK(view->get(0, 3) == 8);
        CHECK(view->get(1, 0) == 9);
        CHECK(view->get(1, 1) == 10);
        CHECK(view->get(1, 2) == 11);
        CHECK(view->get(1, 3) == 12);
        DataObjectFactory::destroy(view);
    }
    SECTION("col range") {
        auto view = DataObjectFactory::create<DT>(m, 0, 4, 1, 3);
        CHECK(view->get(0, 0) == 2);
        CHECK(view->get(0, 1) == 3);
        CHECK(view->get(1, 0) == 6);
        CHECK(view->get(1, 1) == 7);
        CHECK(view->get(2, 0) == 10);
        CHECK(view->get(2, 1) == 11);
        CHECK(view->get(3, 0) == 14);
        CHECK(view->get(3, 1) == 15);
        DataObjectFactory::destroy(view);
    }
    SECTION("row/col range") {
        auto view = DataObjectFactory::create<DT>(m, 1, 3, 1, 3);
        CHECK(view->get(0, 0) == 6);
        CHECK(view->get(0, 1) == 7);
        CHECK(view->get(1, 0) == 10);
        CHECK(view->get(1, 1) == 11);
        DataObjectFactory::destroy(view);
    }

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.get() on view", TAG_DATASTRUCTURES, (CSRMatrix), (double, uint32_t)) {
    using DT = TestType;

    auto m = genGivenVals<DT>(4, {
                                     0,
                                     0,
                                     1,
                                     0,
                                     0,
                                     2,
                                     3,
                                     0,
                                     4,
                                     0,
                                     5,
                                     0,
                                     0,
                                     0,
                                     0,
                                     6,
                                 });

    SECTION("row range") {
        auto view = DataObjectFactory::create<DT>(m, 1, 3);
        CHECK(view->get(0, 0) == 0);
        CHECK(view->get(0, 1) == 2);
        CHECK(view->get(0, 2) == 3);
        CHECK(view->get(0, 3) == 0);
        CHECK(view->get(1, 0) == 4);
        CHECK(view->get(1, 1) == 0);
        CHECK(view->get(1, 2) == 5);
        CHECK(view->get(1, 3) == 0);
        DataObjectFactory::destroy(view);
    }
    // CSRMatrix does not support views on column ranges.

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.set() for filling a matrix", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix),
                           (double, uint32_t)) {
    using DT = TestType;

    auto m = genGivenVals<DT>(3,
                              {
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                              },
                              12);

    auto mExpEnd = genGivenVals<DT>(3, {
                                           6,
                                           7,
                                           3,
                                           8,
                                           9,
                                           2,
                                           10,
                                           4,
                                           1,
                                           11,
                                           5,
                                           12,
                                       });

    SECTION("sorted coordinates") {
        m->set(0, 0, 6);
        m->set(0, 1, 7);
        m->set(0, 2, 3);

        auto mExpMid = genGivenVals<DT>(3, {
                                               6,
                                               7,
                                               3,
                                               0,
                                               0,
                                               0,
                                               0,
                                               0,
                                               0,
                                               0,
                                               0,
                                               0,
                                           });
        CHECK(*m == *mExpMid);
        DataObjectFactory::destroy(mExpMid);

        m->set(0, 3, 8);
        m->set(1, 0, 9);
        m->set(1, 1, 2);
        m->set(1, 2, 10);
        m->set(1, 3, 4);
        m->set(2, 0, 1);
        m->set(2, 1, 11);
        m->set(2, 2, 5);
        m->set(2, 3, 12);

        CHECK(*m == *mExpEnd);
    }
    SECTION("unsorted coordinates") {
        m->set(0, 3, 8);
        m->set(2, 0, 1);
        m->set(1, 1, 2);

        auto mExpMid = genGivenVals<DT>(3, {
                                               0,
                                               0,
                                               0,
                                               8,
                                               0,
                                               2,
                                               0,
                                               0,
                                               1,
                                               0,
                                               0,
                                               0,
                                           });
        CHECK(*m == *mExpMid);
        DataObjectFactory::destroy(mExpMid);

        m->set(2, 1, 11);
        m->set(0, 1, 7);
        m->set(0, 2, 3);
        m->set(1, 2, 10);
        m->set(1, 0, 9);
        m->set(2, 2, 5);
        m->set(1, 3, 4);
        m->set(0, 0, 6);
        m->set(2, 3, 12);

        CHECK(*m == *mExpEnd);
    }

    DataObjectFactory::destroy(mExpEnd);
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.set() for overwriting elements", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix),
                           (double, uint32_t)) {
    using DT = TestType;

    std::vector<typename DT::VT> vals = {
        0, 0, 1, 0, 0, 2, 0, 3, 0, 0, 4, 0,
    };
    auto m = genGivenVals<DT>(3, vals, 12);

    SECTION("zero to zero") {
        m->set(1, 2, 0);
        auto mExp = genGivenVals<DT>(3, vals, 12);
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }
    SECTION("zero to non-zero") {
        m->set(1, 2, 5);
        auto mExp = genGivenVals<DT>(3, {
                                            0,
                                            0,
                                            1,
                                            0,
                                            0,
                                            2,
                                            5,
                                            3,
                                            0,
                                            0,
                                            4,
                                            0,
                                        });
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }
    SECTION("non-zero to zero") {
        m->set(1, 1, 0);
        auto mExp = genGivenVals<DT>(3, {
                                            0,
                                            0,
                                            1,
                                            0,
                                            0,
                                            0,
                                            0,
                                            3,
                                            0,
                                            0,
                                            4,
                                            0,
                                        });
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }
    SECTION("non-zero to non-zero") {
        m->set(1, 1, 5);
        auto mExp = genGivenVals<DT>(3, {
                                            0,
                                            0,
                                            1,
                                            0,
                                            0,
                                            5,
                                            0,
                                            3,
                                            0,
                                            0,
                                            4,
                                            0,
                                        });
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.set() on view", TAG_DATASTRUCTURES, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;

    auto m = genGivenVals<DT>(4, {
                                     1,
                                     2,
                                     3,
                                     4,
                                     5,
                                     6,
                                     7,
                                     8,
                                     9,
                                     10,
                                     11,
                                     12,
                                     13,
                                     14,
                                     15,
                                     16,
                                 });

    SECTION("row range") {
        auto view = DataObjectFactory::create<DT>(m, 1, 3, 0, 4);
        view->set(0, 0, 111);
        view->set(0, 2, 222);
        view->set(1, 1, 333);
        view->set(1, 3, 444);
        auto mExp = genGivenVals<DT>(4, {
                                            1,
                                            2,
                                            3,
                                            4,
                                            111,
                                            6,
                                            222,
                                            8,
                                            9,
                                            333,
                                            11,
                                            444,
                                            13,
                                            14,
                                            15,
                                            16,
                                        });
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
        DataObjectFactory::destroy(view);
    }
    SECTION("col range") {
        auto view = DataObjectFactory::create<DT>(m, 0, 4, 1, 3);
        view->set(0, 0, 111);
        view->set(1, 1, 222);
        view->set(2, 0, 333);
        view->set(3, 1, 444);
        auto mExp = genGivenVals<DT>(4, {
                                            1,
                                            111,
                                            3,
                                            4,
                                            5,
                                            6,
                                            222,
                                            8,
                                            9,
                                            333,
                                            11,
                                            12,
                                            13,
                                            14,
                                            444,
                                            16,
                                        });
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
        DataObjectFactory::destroy(view);
    }
    SECTION("row/col range") {
        auto view = DataObjectFactory::create<DT>(m, 1, 3, 1, 3);
        view->set(0, 1, 111);
        view->set(1, 0, 222);
        auto mExp = genGivenVals<DT>(4, {
                                            1,
                                            2,
                                            3,
                                            4,
                                            5,
                                            6,
                                            111,
                                            8,
                                            9,
                                            222,
                                            11,
                                            12,
                                            13,
                                            14,
                                            15,
                                            16,
                                        });
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
        DataObjectFactory::destroy(view);
    }

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.set() on view", TAG_DATASTRUCTURES, (CSRMatrix), (double, uint32_t)) {
    using DT = TestType;

    auto m = genGivenVals<DT>(4,
                              {
                                  0,
                                  0,
                                  1,
                                  0,
                                  0,
                                  2,
                                  3,
                                  0,
                                  4,
                                  0,
                                  5,
                                  0,
                                  0,
                                  0,
                                  6,
                                  7,
                              },
                              12);

    SECTION("row range") {
        auto view = DataObjectFactory::create<DT>(m, 1, 3);
        view->set(0, 0, 111); // zero to non-zero
        view->set(0, 1, 0);   // non-zero to zero
        view->set(1, 0, 222); // non-zero to non-zero
        view->set(1, 1, 0);   // zero to zero
        auto mExp = genGivenVals<DT>(4, {
                                            0,
                                            0,
                                            1,
                                            0,
                                            111,
                                            0,
                                            3,
                                            0,
                                            222,
                                            0,
                                            5,
                                            0,
                                            0,
                                            0,
                                            6,
                                            7,
                                        });
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
        DataObjectFactory::destroy(view);
    }
    // CSRMatrix does not support views on column ranges.

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.append()", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix), (double, uint32_t)) {
    using DT = TestType;

    auto m = genGivenVals<DT>(3,
                              {
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                              },
                              12);

    SECTION("appending nothing") {
        auto mExp = genGivenVals<DT>(3, {
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                        });
        m->prepareAppend();
        m->finishAppend();
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }
    SECTION("appending one non-last element") {
        auto mExp = genGivenVals<DT>(3, {
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            3,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                        });
        m->prepareAppend();
        m->append(1, 2, 3);
        m->finishAppend();
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }
    SECTION("appending only last element") {
        auto mExp = genGivenVals<DT>(3, {
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            3,
                                        });
        m->prepareAppend();
        m->append(2, 3, 3);
        m->finishAppend();
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }
    SECTION("appending multiple elements") {
        auto mExp = genGivenVals<DT>(3, {
                                            0,
                                            1,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0,
                                            2,
                                            3,
                                            0,
                                        });
        m->prepareAppend();
        m->append(0, 1, 1);
        m->append(2, 1, 2);
        m->append(2, 2, 3);
        m->finishAppend();
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }
    SECTION("appending all elements") {
        auto mExp = genGivenVals<DT>(3, {
                                            6,
                                            7,
                                            3,
                                            8,
                                            9,
                                            2,
                                            10,
                                            4,
                                            1,
                                            11,
                                            5,
                                            12,
                                        });
        m->prepareAppend();
        m->append(0, 0, 6);
        m->append(0, 1, 7);
        m->append(0, 2, 3);
        m->append(0, 3, 8);
        m->append(1, 0, 9);
        m->append(1, 1, 2);
        m->append(1, 2, 10);
        m->append(1, 3, 4);
        m->append(2, 0, 1);
        m->append(2, 1, 11);
        m->append(2, 2, 5);
        m->append(2, 3, 12);
        m->finishAppend();
        CHECK(*m == *mExp);
        DataObjectFactory::destroy(mExp);
    }

    DataObjectFactory::destroy(m);
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.append() on view", TAG_DATASTRUCTURES, (DenseMatrix), (double, uint32_t)) {
    using DT = TestType;

    auto m = DataObjectFactory::create<DT>(4, 5, false);

    auto view1 = DataObjectFactory::create<DT>(m, 0, 2, 0, 2);
    auto view2 = DataObjectFactory::create<DT>(m, 0, 2, 2, 5);
    auto view3 = DataObjectFactory::create<DT>(m, 2, 4, 0, 5);

    // Views can be populated using append in any order for DenseMatrix.

    view2->prepareAppend();
    view2->append(0, 0, 11);
    view2->append(0, 1, 22);
    view2->append(0, 2, 33);
    view2->append(1, 0, 44);
    view2->append(1, 1, 55);
    view2->append(1, 2, 66);
    view2->finishAppend();

    view1->prepareAppend();
    view1->append(0, 0, 1);
    view1->append(0, 1, 2);
    view1->append(1, 0, 3);
    view1->append(1, 1, 4);
    view1->finishAppend();

    view3->prepareAppend();
    view3->append(0, 0, 111);
    view3->append(0, 1, 222);
    view3->append(0, 2, 333);
    view3->append(0, 3, 444);
    view3->append(0, 4, 555);
    view3->append(1, 0, 666);
    view3->append(1, 1, 777);
    view3->append(1, 2, 888);
    view3->append(1, 3, 999);
    view3->append(1, 4, 123);
    view3->finishAppend();

    CHECK(*m ==
          *genGivenVals<DT>(4, {
                                   1, 2, 11, 22, 33, 3, 4, 44, 55, 66, 111, 222, 333, 444, 555, 666, 777, 888, 999, 123,
                               }));

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(view1);
    DataObjectFactory::destroy(view2);
    DataObjectFactory::destroy(view3);
}

TEMPLATE_PRODUCT_TEST_CASE("Matrix.append() on view", TAG_DATASTRUCTURES, (CSRMatrix), (double, uint32_t)) {
    using DT = TestType;

    auto m = DataObjectFactory::create<DT>(5, 3, 15, false);

    auto view1 = DataObjectFactory::create<DT>(m, 0, 2);
    auto view2 = DataObjectFactory::create<DT>(m, 2, 3);
    auto view3 = DataObjectFactory::create<DT>(m, 3, 5);

    // When using append, views must be populated in order for CSRMatrix.

    view1->prepareAppend();
    view1->append(0, 1, 1);
    view1->append(1, 2, 2);
    view1->finishAppend();

    view2->prepareAppend();
    view2->append(0, 0, 11);
    view2->append(0, 1, 22);
    view2->append(0, 2, 0);
    view2->finishAppend();

    view3->prepareAppend();
    view3->append(0, 1, 111);
    view3->append(0, 2, 0);
    view3->append(1, 0, 222);
    view3->append(1, 1, 333);
    view3->finishAppend();

    CHECK(*m == *genGivenVals<DT>(5, {
                                         0,
                                         1,
                                         0,
                                         0,
                                         0,
                                         2,
                                         11,
                                         22,
                                         0,
                                         0,
                                         111,
                                         0,
                                         222,
                                         333,
                                         0,
                                     }));

    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(view1);
    DataObjectFactory::destroy(view2);
    DataObjectFactory::destroy(view3);
}