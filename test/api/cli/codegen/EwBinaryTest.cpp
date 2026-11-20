/*
 * Copyright 2023 The DAPHNE Consortium
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

#include <api/cli/Utils.h>
#include <tags.h>

#include <catch.hpp>
#include <sstream>
#include <string>

#include "api/cli/StatusCode.h"

const std::string dirPath = "test/api/cli/codegen/";

void test_binary_lowering(const std::string op, const std::string kernel_call, const std::string result) {
    std::stringstream out;
    std::stringstream err;

    int status = runDaphne(out, err, "--explain", "llvm", (dirPath + "ewbinary_" + op + ".daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    CHECK_THAT(err.str(), Catch::Contains(kernel_call));
    CHECK(out.str() == result);

    out.str(std::string());
    err.str(std::string());

    status = runDaphne(out, err, "--explain", "llvm", "--codegen", (dirPath + "ewbinary_" + op + ".daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    CHECK_THAT(err.str(), !Catch::Contains(kernel_call));
    CHECK(out.str() == result);
}

TEST_CASE("ewBinaryAdd", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x3, double)\n"
                            "2 4 6\n"
                            "8 10 12\n"
                        "DenseMatrix(2x3, double)\n"
                            "6 6 6\n"
                            "9 9 9\n"
                        "DenseMatrix(2x3, double)\n"
                            "5 6 7\n"
                            "7 8 9\n"
                        "DenseMatrix(2x3, double)\n"
                            "3 4 5\n"
                            "6 7 8\n"
                        "6\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "2 4 6\n"
                            "8 10 12\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "6 6 6\n"
                            "9 9 9\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "5 6 7\n"
                            "7 8 9\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "3 4 5\n"
                            "6 7 8\n"
                        "6\n";
    // clang-format on
    test_binary_lowering("add", "llvm.call @_ewAdd__", result);
}

TEST_CASE("ewBinarySub", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x3, double)\n"
                            "-5 -3 -1\n"
                            "1 3 5\n"
                        "DenseMatrix(2x3, double)\n"
                            "-4 -2 0\n"
                            "-1 1 3\n"
                        "DenseMatrix(2x3, double)\n"
                            "-3 -2 -1\n"
                            "1 2 3\n"
                        "DenseMatrix(2x3, double)\n"
                            "-1 0 1\n"
                            "2 3 4\n"
                        "-2\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "-5 -3 -1\n"
                            "1 3 5\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "-4 -2 0\n"
                            "-1 1 3\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "-3 -2 -1\n"
                            "1 2 3\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "-1 0 1\n"
                            "2 3 4\n"
                        "-2\n";
    // clang-format on
    test_binary_lowering("sub", "llvm.call @_ewSub__", result);
}

TEST_CASE("ewBinaryMul", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x3, double)\n"
                            "1 4 9\n"
                            "16 25 36\n"
                        "DenseMatrix(2x3, double)\n"
                            "5 8 9\n"
                            "20 20 18\n"
                        "DenseMatrix(2x3, double)\n"
                            "4 8 12\n"
                            "12 15 18\n"
                        "DenseMatrix(2x3, double)\n"
                            "2 4 6\n"
                            "8 10 12\n"
                        "8\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "1 4 9\n"
                            "16 25 36\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "5 8 9\n"
                            "20 20 18\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "4 8 12\n"
                            "12 15 18\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "2 4 6\n"
                            "8 10 12\n"
                        "8\n";
    // clang-format on
    test_binary_lowering("mul", "llvm.call @_ewMul__", result);
}

TEST_CASE("ewBinaryDiv", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x3, double)\n"
                            "1 1 1\n"
                            "1 1 1\n"
                        "DenseMatrix(2x3, double)\n"
                            "0.2 0.5 1\n"
                            "0.8 1.25 2\n"
                        "DenseMatrix(2x3, double)\n"
                            "0.25 0.5 0.75\n"
                            "1.33333 1.66667 2\n"
                        "DenseMatrix(2x3, double)\n"
                            "0.5 1 1.5\n"
                            "2 2.5 3\n"
                        "0.5\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "1 1 1\n"
                            "1 1 1\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "0 0 1\n"
                            "0 1 2\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "0 0 0\n"
                            "1 1 2\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "0 0 0\n"
                            "1 1 1\n"
                        "2\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "1 1 1\n"
                            "1 1 1\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "0 0 1\n"
                            "0 1 2\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "0 0 0\n"
                            "1 1 2\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "0 0 0\n"
                            "1 1 1\n"
                        "2\n";
    // clang-format on
    test_binary_lowering("div", "llvm.call @_ewDiv__", result);
}

// TEST_CASE("ewBinaryPow", TAG_CODEGEN) {
//     // clang-format off
//     std::string result = "DenseMatrix(2x3, double)\n"
//                             "1 4 27\n"
//                             "256 3125 46656\n"
//                         "DenseMatrix(2x3, double)\n"
//                             "1 16 81\n"
//                             "256 625 1296\n"
//                         "16\n"
//                         "DenseMatrix(2x3, int64_t)\n"
//                             "1 4 27\n"
//                             "256 3125 46656\n"
//                         "DenseMatrix(2x3, int64_t)\n"
//                             "1 16 81\n"
//                             "256 625 1296\n"
//                         "16\n";
//     // clang-format on
//     test_binary_lowering("pow", "llvm.call @_ewPow__", "llvm.intr.pow", result);
// }

TEST_CASE("ewBinaryMin", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x3, double)\n"
                            "0 1 3\n"
                            "4 3 2\n"
                        "DenseMatrix(2x3, double)\n"
                            "1 2 1\n"
                            "2 4 1\n"
                        "DenseMatrix(2x3, double)\n"
                            "1 2 3\n"
                            "3 3 3\n"
                        "DenseMatrix(2x3, double)\n"
                            "1 2 3\n"
                            "4 4 4\n"
                        "2\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "0 1 3\n"
                            "4 3 2\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "1 2 1\n"
                            "2 4 1\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "1 2 3\n"
                            "3 3 3\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "1 2 3\n"
                            "4 4 4\n"
                        "2\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "0 1 3\n"
                            "4 3 2\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "1 2 1\n"
                            "2 4 1\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "1 2 3\n"
                            "3 3 3\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "1 2 3\n"
                            "4 4 4\n"
                        "2\n";
    // clang-format on
    test_binary_lowering("min", "llvm.call @_ewMin__", result);
}

TEST_CASE("ewBinaryMax", TAG_CODEGEN) {
    // clang-format off
    std::string result = "DenseMatrix(2x3, double)\n"
                            "1 2 4\n"
                            "11 5 6\n"
                        "DenseMatrix(2x3, double)\n"
                            "5 2 3\n"
                            "5 5 6\n"
                        "DenseMatrix(2x3, double)\n"
                            "3 3 3\n"
                            "4 5 6\n"
                        "DenseMatrix(2x3, double)\n"
                            "4 4 4\n"
                            "4 5 6\n"
                        "4\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "1 2 4\n"
                            "11 5 6\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "5 2 3\n"
                            "5 5 6\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "3 3 3\n"
                            "4 5 6\n"
                        "DenseMatrix(2x3, int64_t)\n"
                            "4 4 4\n"
                            "4 5 6\n"
                        "4\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "1 2 4\n"
                            "11 5 6\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "5 2 3\n"
                            "5 5 6\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "3 3 3\n"
                            "4 5 6\n"
                        "DenseMatrix(2x3, uint64_t)\n"
                            "4 4 4\n"
                            "4 5 6\n"
                        "4\n";
    // clang-format on
    test_binary_lowering("max", "llvm.call @_ewMax__", result);
}
