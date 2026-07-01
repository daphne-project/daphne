/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * Modifications 2024 The DAPHNE Consortium.
 */

// This code has been manually translated from Apache SystemDS (and
// significantly adapted).

#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>

const std::string DATASET_DIR = "test/data/";

// const std::string TITANIC_DATA = DATASET_DIR + "titanic.csv";
const std::string WINE_DATA = DATASET_DIR + "wine/winequality-red-white.csv";
// const std::string EEG_DATA = DATASET_DIR + "EEG.csv";

void runDecisionTree(int testNr, const std::string &dataFilePath, double minAcc, int dt, double maxV) {
    const std::string scriptFileName =
        ("test/api/cli/algorithms/decisionTreeRealData" + std::to_string(testNr)) + ".daphne";
    const std::string argData = "data=\"" + dataFilePath + "\"";
    const std::string argDt = "dt=" + std::to_string(dt);
    const std::string argMaxV = "maxV=" + std::to_string(maxV);
    std::stringstream out;
    std::stringstream err;

    // Run DAPHNE.
    int status = runDaphne(out, err, scriptFileName.c_str(), argData.c_str(), argDt.c_str(), argMaxV.c_str());

    // DAPHNE must parse/compile/execute successfully.
    CHECK(status == StatusCode::SUCCESS);

    // DAPHNE's stdout must represent a floating-point value (the accuracy printed by the DaphneDSL script) and...
    double acc;
    auto myIsFloatRepr = [&](std::string const &str) -> bool { return isFloatRepr(str, acc); };
    // Note: REQUIRE_THAT and Predicate ensure that out.str() is printed if and only if the predicate fails (important
    // for interpreting the output).
    REQUIRE_THAT(out.str(),
                 Catch::Matchers::Predicate<std::string>(
                     myIsFloatRepr, "the string must represent a floating-point number acceptable by std::stod() and "
                                    "nothing else (no leading whitespace, no trailing characters)"));
    // ...this floating-point value (the accuracy) must be greater than or equal to a certain threshold.
    CHECK(acc >= minAcc);

    // DAPHNE's stderr must be empty.
    CHECK(err.str() == "");
}

// TEST_CASE("decisionTreeTitanic_MaxV1", TAG_ALGORITHMS) {
//     runDecisionTree(1, TITANIC_DATA, 0.875, 1, 1.0);
// }

// TEST_CASE("randomForestTitanic1_MaxV1", TAG_ALGORITHMS) {
//     //one tree with sample_frac=1 should be equivalent to decision tree
//     runDecisionTree(1, TITANIC_DATA, 0.875, 2, 1.0);
// }

// TEST_CASE("randomForestTitanic8_MaxV1", TAG_ALGORITHMS) {
//     //8 trees with sample fraction 0.125 each, accuracy 0.785 due to
//     randomness runDecisionTree(1, TITANIC_DATA, 0.793, 9, 1.0);
// }

// TEST_CASE("decisionTreeTitanic_MaxV06", TAG_ALGORITHMS) {
//     runDecisionTree(1, TITANIC_DATA, 0.871, 1, 0.6);
// }

// TEST_CASE("randomForestTitanic1_MaxV06", TAG_ALGORITHMS) {
//     //one tree with sample_frac=1 should be equivalent to decision tree
//     runDecisionTree(1, TITANIC_DATA, 0.871, 2, 0.6);
// }

// TEST_CASE("randomForestTitanic8_MaxV06", TAG_ALGORITHMS) {
//     //8 trees with sample fraction 0.125 each, accuracy 0.785 due to
//     randomness runDecisionTree(1, TITANIC_DATA, 0.793, 9, 0.6);
// }

TEST_CASE("decisionTree_Wine_MaxV1", TAG_ALGORITHMS) { runDecisionTree(2, WINE_DATA, 0.989, 1, 1.0); }

// Added in DAPHNE (wasn't present in SystemDS).
TEST_CASE("decisionTree_Wine_MaxV05", TAG_ALGORITHMS) { runDecisionTree(2, WINE_DATA, 0.988, 1, 0.5); }

TEST_CASE("randomForestWine_MaxV1", TAG_ALGORITHMS) { runDecisionTree(2, WINE_DATA, 0.989, 2, 1.0); }

// Added in DAPHNE (wasn't present in SystemDS).
TEST_CASE("randomForestWine_MaxV05", TAG_ALGORITHMS) { runDecisionTree(2, WINE_DATA, 0.988, 2, 0.5); }

TEST_CASE("decisionTree_WineReg_MaxV1", TAG_ALGORITHMS) {
    // for regression we compare R2 and use rss to optimize
    runDecisionTree(3, WINE_DATA, 0.364, 1, 1.0);
}

// Added in DAPHNE (wasn't present in SystemDS).
TEST_CASE("decisionTree_WineReg_MaxV05", TAG_ALGORITHMS) {
    // for regression we compare R2 and use rss to optimize
    runDecisionTree(3, WINE_DATA, 0.291, 1, 0.5);
}

TEST_CASE("randomForestWineReg_MaxV1", TAG_ALGORITHMS) {
    // for regression we compare R2 and use rss to optimize
    runDecisionTree(3, WINE_DATA, 0.364, 2, 1.0);
}

// Added in DAPHNE (wasn't present in SystemDS).
TEST_CASE("randomForestWineReg_MaxV05", TAG_ALGORITHMS) {
    // for regression we compare R2 and use rss to optimize
    runDecisionTree(3, WINE_DATA, 0.284, 2, 0.5);
}

// TEST_CASE("decisionTreeEEG_MaxV1", TAG_ALGORITHMS) {
//     //for regression we compare R2 and use rss to optimize
//     runDecisionTree(4, EEG_DATA, 0.62, 1, 1.0);
// }

// TEST_CASE("randomForestEEG_MaxV1", TAG_ALGORITHMS) {
//     //for regression we compare R2 and use rss to optimize
//     runDecisionTree(4, EEG_DATA, 0.62, 2, 1.0);
// }