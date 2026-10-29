/*
 * Copyright 2025 The DAPHNE Consortium
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

#include <tags.h>

#include <cstdlib>
#include <string>

// Runs the shell test for build.sh's memory-detection helpers
// (get_memory_gb, calc_daphne_jobs) so the tested path stays the
// actual shell code, not a C++ port. Script self-contained; exits
// non-zero on any failed assertion.
TEST_CASE("build.sh memory-detection helpers", TAG_BUILD) {
    const std::string cmd = "bash test/build/test_build_helpers.sh";
    int rc = std::system(cmd.c_str());
    CHECK(rc == 0);
}
