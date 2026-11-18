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

#include "run_tests.h"

#include "api/cli/StatusCode.h"
#include "api/cli/Utils.h"

#include <tags.h>

const std::string dirPath = "test/cse/";

// Place all test files with FileCheck directives in the dirPath.
// LIT will test all *.mlir files in the directory.
TEST_CASE("cse", TAG_CSE) {
    std::stringstream out;
    std::stringstream err;

    int status = runLIT(out, err, dirPath);

    if (status != StatusCode::SUCCESS) {
        std::cout << "runLIT return status: " << std::to_string(status) << "\n";
        std::cout << "runLIT out:\n" << out.str() << "\n";
        std::cout << "runLIT err:\n" << err.str() << "\n";
    }

    CHECK(status == StatusCode::SUCCESS);
}
