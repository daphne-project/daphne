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

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/TransferProperties.h>

#include <tags.h>

#include <catch.hpp>

TEMPLATE_PRODUCT_TEST_CASE("Matrix.numDistinct transfer", TAG_DATASTRUCTURES, (DenseMatrix, CSRMatrix),
                           (double, int64_t)) {
    using DT = TestType;
    auto matrix = genGivenVals<DT>(2, {0, 1, 1, 2});
    CHECK(matrix->numDistinct == -1);
    transferProperties(matrix, 0.75, static_cast<int64_t>(BoolOrUnknown::Unknown), 3, nullptr);
    CHECK(matrix->numDistinct == 3);
    CHECK(matrix->sparsity == 0.75);
    transferProperties(matrix, -1, static_cast<int64_t>(BoolOrUnknown::Unknown), -1, nullptr);
    CHECK(matrix->numDistinct == -1);
    DataObjectFactory::destroy(matrix);
}