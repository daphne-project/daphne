/*
 * Copyright 2024 The DAPHNE Consortium
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

const std::string dirPath = "test/api/cli/extensibility/";

TEST_CASE("extension_kernel", TAG_EXTENSIBILITY) {
    int status = -1;
    std::stringstream out;
    std::stringstream err;

    std::string extDir = std::string(dirPath + "kernel_extension_test");

    // *************************************************************************
    // Build the custom kernel extension.
    // *************************************************************************
    // The extension's build process is on purpose isolated from DAPHNE's build
    // process, since extensions are developed in stand-alone code bases and
    // exactly that shall be tested here.
    // We use ninja as the build system, since ninja is anyway required in the
    // DAPHNE development environment. It would also work with make, but make
    // would create an additional dependency.
    status = runProgram(out, err, "ninja", "ninja", "-C", extDir.c_str());
    if (status) {
        // We don't expect any specifc output from ninja, but only if ninja
        // failed, we want to see what it printed to stdout and stderr.
        CHECK(out.str() == "");
        CHECK(err.str() == "");
    }
    // Don't continue in case the build of the extension failed.
    REQUIRE(status == 0);

    // *************************************************************************
    // Use the custom kernel extension based on hints or priority.
    // *************************************************************************
    // All of the following invocations of DAPHNE register the extension with DAPHNE at run-time. DAPHNE itself is not
    // re-built or anything. This extension provides a custom kernel for a case (DaphneIR operation, input/output types,
    // backend) that is already covered by an existing built-in kernel.

    // If a kernel hint is given, this kernel must always be used, irrespective of its priority.
    // Run a DaphneDSL script which uses a kernel from the extension through a kernel hint.
    SECTION("hint, no priority") { // must use the custom kernel
        compareDaphneToStr("hello from mySumAll\n2\n",
                           std::string(dirPath + "extension_kernel_usage_hint.daphne").c_str(), "--kernel-ext",
                           std::string(dirPath + "kernel_extension_test/myKernels.json").c_str());
    }
    SECTION("hint, no priority, type mismatch") { // must use the custom kernel
        checkDaphneStatusCode(StatusCode::PASS_ERROR,
                              std::string(dirPath + "extension_kernel_usage_hint_type_mismatch.daphne").c_str(),
                              "--kernel-ext", std::string(dirPath + "kernel_extension_test/myKernels.json").c_str());
    }
    SECTION("hint, default priority") { // must use the custom kernel
        compareDaphneToStr("hello from mySumAll\n2\n",
                           std::string(dirPath + "extension_kernel_usage_hint.daphne").c_str(), "--kernel-ext",
                           std::string(dirPath + "kernel_extension_test/myKernels.json:0").c_str());
    }
    SECTION("hint, higher priority") { // must use the custom kernel
        compareDaphneToStr("hello from mySumAll\n2\n",
                           std::string(dirPath + "extension_kernel_usage_hint.daphne").c_str(), "--kernel-ext",
                           std::string(dirPath + "kernel_extension_test/myKernels.json:1").c_str());
    }
    SECTION("hint, lower priority") { // must use the custom kernel
        compareDaphneToStr("hello from mySumAll\n2\n",
                           std::string(dirPath + "extension_kernel_usage_hint.daphne").c_str(), "--kernel-ext",
                           std::string(dirPath + "kernel_extension_test/myKernels.json:-1").c_str());
    }
    // If no kernel hint is given, the kernel must be used when it has a higher-than-default priority and must not be
    // used if it has a lower-than-default priority. For "no hint, no priority" and "no hint, default priority",
    // DAPHNE's choice whether to use the custom or the built-in kernel is unspecified, so we don't test these cases
    // here.
    // Run a DaphneDSL script which does not use a kernel hint.
    SECTION("no hint, higher priority") { // must use the custom kernel
        compareDaphneToStr("hello from mySumAll\n2\n",
                           std::string(dirPath + "extension_kernel_usage_nohint.daphne").c_str(), "--kernel-ext",
                           std::string(dirPath + "kernel_extension_test/myKernels.json:1").c_str());
    }
    SECTION("no hint, lower priority") { // must NOT use the custom kernel
        compareDaphneToStr("2\n", std::string(dirPath + "extension_kernel_usage_nohint.daphne").c_str(), "--kernel-ext",
                           std::string(dirPath + "kernel_extension_test/myKernels.json:-1").c_str());
    }
    // If an invalid value is specified for the priority, DAPHNE must stop, even if the kernel extension itself exists.
    SECTION("no hint, invalid priority (empty)") {
        checkDaphneStatusCode(StatusCode::PARSER_ERROR,
                              std::string(dirPath + "extension_kernel_usage_nohint.daphne").c_str(), "--kernel-ext",
                              std::string(dirPath + "kernel_extension_test/myKernels.json:").c_str());
    }
    SECTION("no hint, invalid priority (float)") {
        checkDaphneStatusCode(StatusCode::PARSER_ERROR,
                              std::string(dirPath + "extension_kernel_usage_nohint.daphne").c_str(), "--kernel-ext",
                              std::string(dirPath + "kernel_extension_test/myKernels.json:0.1").c_str());
    }
    SECTION("no hint, invalid priority (string)") {
        checkDaphneStatusCode(StatusCode::PARSER_ERROR,
                              std::string(dirPath + "extension_kernel_usage_nohint.daphne").c_str(), "--kernel-ext",
                              std::string(dirPath + "kernel_extension_test/myKernels.json:abc").c_str());
    }
    SECTION("no hint, invalid priority (integer and string)") {
        checkDaphneStatusCode(StatusCode::PARSER_ERROR,
                              std::string(dirPath + "extension_kernel_usage_nohint.daphne").c_str(), "--kernel-ext",
                              std::string(dirPath + "kernel_extension_test/myKernels.json:123abc").c_str());
    }
    SECTION("no hint, invalid priority (too huge integer)") {
        checkDaphneStatusCode(
            StatusCode::PARSER_ERROR, std::string(dirPath + "extension_kernel_usage_nohint.daphne").c_str(),
            "--kernel-ext", std::string(dirPath + "kernel_extension_test/myKernels.json:99999999999999999999").c_str());
    }

    // Clear the streams.
    out.clear();
    err.clear();

    // *************************************************************************
    // Clean the build of the custom kernel extension.
    // *************************************************************************
    // Such that the next invocation of this test case needs to build the
    // extension again, thereby testing again if the extension can be built
    // successfully.
    status = runProgram(out, err, "ninja", "ninja", "-C", extDir.c_str(), "-t", "clean");
    if (status) {
        // We don't expect any specifc output from ninja, but only if ninja
        // failed, we want to see what it printed to stdout and stderr.
        CHECK(out.str() == "");
        CHECK(err.str() == "");
    }
    CHECK(status == 0);
}