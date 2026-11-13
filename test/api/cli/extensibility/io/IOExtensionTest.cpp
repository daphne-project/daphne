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

#include <api/cli/StatusCode.h>
#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>

const std::string dirPath = "test/api/cli/extensibility/io/";

TEST_CASE("extension_io", TAG_EXTENSIBILITY) {
    int status = -1;
    std::stringstream out;
    std::stringstream err;

    size_t numExt = 2;
    std::string extDirPrefix = std::string(dirPath + "io_extension_dummy");

    // *************************************************************************
    // Build the custom IO extensions.
    // *************************************************************************
    // The extensions' build processes are on purpose isolated from DAPHNE's build
    // process, since extensions are developed in stand-alone code bases and
    // exactly that shall be tested here.
    // We use ninja as the build system, since ninja is anyway required in the
    // DAPHNE development environment. It would also work with make, but make
    // would create an additional dependency.
    for (size_t i = 1; i <= numExt; i++) {
        status = runProgram(out, err, "ninja", "ninja", "-C", (extDirPrefix + std::to_string(i)).c_str());
        if (status) {
            // We don't expect any specifc output from ninja, but only if ninja
            // failed, we want to see what it printed to stdout and stderr.
            CHECK(out.str() == "");
            CHECK(err.str() == "");
        }
        // Don't continue in case the build of the extension failed.
        REQUIRE(status == 0);
    }

    // *************************************************************************
    // Use the custom IO extensions based on engine specification or priority.
    // *************************************************************************
    // All of the following invocations of DAPHNE register one or two IO extensions with DAPHNE at run-time. DAPHNE
    // itself is not re-built or anything. These extensions provide custom dummy readers/writers for the fictional
    // ".dummy" file format that is not normally covered by an existing built-in reader/writer.

    const std::string scriptReadMatrixEnginePathStr = dirPath + "extension_io_readMatrix_engine.daphne";
    const std::string scriptReadMatrixEngine1OptPathStr = dirPath + "extension_io_readMatrix_engine_1opt.daphne";
    const std::string scriptReadMatrixEngine2OptPathStr = dirPath + "extension_io_readMatrix_engine_2opt.daphne";
    const std::string scriptReadMatrixNoEnginePathStr = dirPath + "extension_io_readMatrix_noEngine.daphne";
    const std::string scriptReadFrameNoEnginePathStr = dirPath + "extension_io_readFrame_noEngine.daphne";
    const std::string scriptWriteMatrixNoEnginePathStr = dirPath + "extension_io_writeMatrix_noEngine.daphne";
    const std::string extDummy1Path = dirPath + "io_extension_dummy1/dummy1.json";
    const std::string extDummy2Path = dirPath + "io_extension_dummy2/dummy2.json";
    const char *scriptReadMatrixEnginePath = scriptReadMatrixEnginePathStr.c_str();
    const char *scriptReadMatrixEngine1OptPath = scriptReadMatrixEngine1OptPathStr.c_str();
    const char *scriptReadMatrixEngine2OptPath = scriptReadMatrixEngine2OptPathStr.c_str();
    const char *scriptReadMatrixNoEnginePath = scriptReadMatrixNoEnginePathStr.c_str();
    const char *scriptReadFrameNoEnginePath = scriptReadFrameNoEnginePathStr.c_str();
    const char *scriptWriteMatrixNoEnginePath = scriptWriteMatrixNoEnginePathStr.c_str();
    const std::string nothingInDummyArg = "file=\"" + dirPath + "nothing_in.dummy\"";
    const std::string nothingOutDummyArg = "file=\"" + dirPath + "nothing_out.dummy\"";

    // -------------------------------------------------------------------------
    // Invalid registration of extensions
    // -------------------------------------------------------------------------

    // If an invalid value is specified for the priority, DAPHNE must stop, even if the kernel extension itself exists.
    SECTION("invalid registration, invalid priority (empty)") { // must fail
        checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptReadMatrixNoEnginePath, "--fileio-ext",
                              (extDummy1Path + ":").c_str(), "--args", nothingInDummyArg.c_str());
    }
    SECTION("invalid registration, invalid priority (float)") { // must fail
        checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptReadMatrixNoEnginePath, "--fileio-ext",
                              (extDummy1Path + ":0.1").c_str(), "--args", nothingInDummyArg.c_str());
    }
    SECTION("invalid registration, invalid priority (string)") { // must fail
        checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptReadMatrixNoEnginePath, "--fileio-ext",
                              (extDummy1Path + ":abc").c_str(), "--args", nothingInDummyArg.c_str());
    }
    SECTION("invalid registration, invalid priority (integer and string)") { // must fail
        checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptReadMatrixNoEnginePath, "--fileio-ext",
                              (extDummy1Path + ":123abc").c_str(), "--args", nothingInDummyArg.c_str());
    }
    SECTION("invalid registration, invalid priority (too huge integer)") { // must fail
        checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptReadMatrixNoEnginePath, "--fileio-ext",
                              (extDummy1Path + ":99999999999999999999").c_str(), "--args", nothingInDummyArg.c_str());
    }
    // Duplicate registrations are not allowed.
    SECTION("invalid registration, duplicate, 2x the same extension") { // must fail
        checkDaphneStatusCode(StatusCode::PARSER_ERROR, scriptReadMatrixNoEnginePath, "--fileio-ext",
                              extDummy1Path.c_str(), "--fileio-ext", extDummy1Path.c_str(), "--args",
                              nothingInDummyArg.c_str());
    }
    // TODO "invalid registration, duplicate, different extensions with the same engine name"

    // -------------------------------------------------------------------------
    // Use of the right extension
    // -------------------------------------------------------------------------

    // If an engine is specified, this engine must always be used, irrespective of its priority.
    // Run a DaphneDSL script which uses a reader from an IO extension by specifying the engine.
    SECTION("engine dummy1, no priorities") { // must use dummy1
        compareDaphneToStr("hello from dummy1_read_DenseMatrix\nDenseMatrix(0x0, double)\n", scriptReadMatrixEnginePath,
                           "--fileio-ext", extDummy1Path.c_str(), "--fileio-ext", extDummy2Path.c_str(), "--args",
                           (nothingInDummyArg + ",engine=\"dummy1\"").c_str());
    }
    SECTION("engine dummy2, no priorities") { // must use dummy2
        compareDaphneToStr(
            "hello from dummy2_read_DenseMatrix, options: bar=\"123\",foo=\"abc\"\nDenseMatrix(0x0, double)\n",
            scriptReadMatrixEnginePath, "--fileio-ext", extDummy1Path.c_str(), "--fileio-ext", extDummy2Path.c_str(),
            "--args", (nothingInDummyArg + ",engine=\"dummy2\"").c_str());
    }
    SECTION("engine dummy1, higher priority") { // must use dummy1
        compareDaphneToStr("hello from dummy1_read_DenseMatrix\nDenseMatrix(0x0, double)\n", scriptReadMatrixEnginePath,
                           "--fileio-ext", (extDummy1Path + ":1").c_str(), "--fileio-ext",
                           (extDummy2Path + ":0").c_str(), "--args",
                           (nothingInDummyArg + ",engine=\"dummy1\"").c_str());
    }
    SECTION("engine dummy1, lower priority") { // must use dummy1
        compareDaphneToStr("hello from dummy1_read_DenseMatrix\nDenseMatrix(0x0, double)\n", scriptReadMatrixEnginePath,
                           "--fileio-ext", (extDummy1Path + ":0").c_str(), "--fileio-ext",
                           (extDummy2Path + ":1").c_str(), "--args",
                           (nothingInDummyArg + ",engine=\"dummy1\"").c_str());
    }
    SECTION("engine dummy2, higher priority") { // must use dummy2
        compareDaphneToStr(
            "hello from dummy2_read_DenseMatrix, options: bar=\"123\",foo=\"abc\"\nDenseMatrix(0x0, double)\n",
            scriptReadMatrixEnginePath, "--fileio-ext", (extDummy1Path + ":0").c_str(), "--fileio-ext",
            (extDummy2Path + ":1").c_str(), "--args", (nothingInDummyArg + ",engine=\"dummy2\"").c_str());
    }
    SECTION("engine dummy2, lower priority") { // must use dummy2
        compareDaphneToStr(
            "hello from dummy2_read_DenseMatrix, options: bar=\"123\",foo=\"abc\"\nDenseMatrix(0x0, double)\n",
            scriptReadMatrixEnginePath, "--fileio-ext", (extDummy1Path + ":1").c_str(), "--fileio-ext",
            (extDummy2Path + ":0").c_str(), "--args", (nothingInDummyArg + ",engine=\"dummy2\"").c_str());
    }
    SECTION("non-existing engine, no priorities") { // must fail
        checkDaphneStatusCode(StatusCode::EXECUTION_ERROR, scriptReadMatrixEnginePath, "--fileio-ext",
                              extDummy1Path.c_str(), "--fileio-ext", extDummy2Path.c_str(), "--args",
                              (nothingInDummyArg + ",engine=\"non-existing-engine\"").c_str());
    }
    // If no engine is specified, the reader with the higher priority must be used.
    // Run a DaphneDSL script which uses a reader from an IO extension without specifying the engine.
    SECTION("no engine, dummy1 higher priority") { // must use dummy1
        compareDaphneToStr("hello from dummy1_read_DenseMatrix\nDenseMatrix(0x0, double)\n",
                           scriptReadMatrixNoEnginePath, "--fileio-ext", (extDummy1Path + ":1").c_str(), "--fileio-ext",
                           (extDummy2Path + ":0").c_str(), "--args", nothingInDummyArg.c_str());
    }
    SECTION("no engine, dummy2 higher priority") { // must use dummy2
        compareDaphneToStr(
            "hello from dummy2_read_DenseMatrix, options: bar=\"123\",foo=\"abc\"\nDenseMatrix(0x0, double)\n",
            scriptReadMatrixNoEnginePath, "--fileio-ext", (extDummy1Path + ":0").c_str(), "--fileio-ext",
            (extDummy2Path + ":1").c_str(), "--args", nothingInDummyArg.c_str());
    }

    // -------------------------------------------------------------------------
    // Use of reader/writer options
    // -------------------------------------------------------------------------

    // Each option of a reader/writer can be specified independently. If an option is specified, the given values must
    // be passed to the reader/writer; otherwise, the default value must be used.
    SECTION("engine dummy2, no options") { // must use defaults for foo and bar
        compareDaphneToStr(
            "hello from dummy2_read_DenseMatrix, options: bar=\"123\",foo=\"abc\"\nDenseMatrix(0x0, double)\n",
            scriptReadMatrixEnginePath, "--fileio-ext", extDummy2Path.c_str(), "--args",
            (nothingInDummyArg + ",engine=\"dummy2\"").c_str());
    }
    SECTION("engine dummy2, option foo") { // must use given value for foo and default for bar
        compareDaphneToStr(
            "hello from dummy2_read_DenseMatrix, options: bar=\"123\",foo=\"def\"\nDenseMatrix(0x0, double)\n",
            scriptReadMatrixEngine1OptPath, "--fileio-ext", extDummy2Path.c_str(), "--args",
            (nothingInDummyArg + ",engine=\"dummy2\",opt1=\"foo\",val1=\"def\"").c_str());
    }
    SECTION("engine dummy2, option bar") { // must use default for foo and given value for bar
        compareDaphneToStr(
            "hello from dummy2_read_DenseMatrix, options: bar=\"456\",foo=\"abc\"\nDenseMatrix(0x0, double)\n",
            scriptReadMatrixEngine1OptPath, "--fileio-ext", extDummy2Path.c_str(), "--args",
            (nothingInDummyArg + ",engine=\"dummy2\",opt1=\"bar\",val1=\"456\"").c_str());
    }
    SECTION("engine dummy2, options foo and bar") { // must use given values for foo and bar
        compareDaphneToStr(
            "hello from dummy2_read_DenseMatrix, options: bar=\"456\",foo=\"def\"\nDenseMatrix(0x0, double)\n",
            scriptReadMatrixEngine2OptPath, "--fileio-ext", extDummy2Path.c_str(), "--args",
            (nothingInDummyArg + ",engine=\"dummy2\",opt1=\"foo\",val1=\"def\",opt2=\"bar\",val2=\"456\"").c_str());
    }
    SECTION("engine dummy2, non-existing option") { // must fail
        checkDaphneStatusCode(
            StatusCode::EXECUTION_ERROR, scriptReadMatrixEngine1OptPath, "--fileio-ext", extDummy2Path.c_str(),
            "--args", (nothingInDummyArg + ",engine=\"dummy2\",opt1=\"non-existing-option\",val1=\"000\"").c_str());
    }
    SECTION("engine dummy1, non-existing option that exists in another extension") { // must fail
        checkDaphneStatusCode(StatusCode::EXECUTION_ERROR, scriptReadMatrixEngine1OptPath, "--fileio-ext",
                              extDummy1Path.c_str(), "--fileio-ext", extDummy2Path.c_str(), "--args",
                              (nothingInDummyArg + ",engine=\"dummy1\",opt1=\"foo\",val1=\"def\"").c_str());
    }

    // -------------------------------------------------------------------------
    // Correct handling of data types
    // -------------------------------------------------------------------------
    // TODO More systematic tests for reading/writing Frame, CSRMatrix, and potentially other data types.

    SECTION("reading a Frame") {
        compareDaphneToStr("hello from dummy2_read_Frame\nFrame(0x0, [])\n", scriptReadFrameNoEnginePath,
                           "--fileio-ext", extDummy2Path.c_str(), "--args", nothingInDummyArg.c_str());
    }

    // -------------------------------------------------------------------------
    // Correct handling of writers
    // -------------------------------------------------------------------------
    // TODO More systematic tests for writing.

    SECTION("writing a DenseMatrix") {
        compareDaphneToStr("hello from dummy2_write_DenseMatrix\n", scriptWriteMatrixNoEnginePath, "--fileio-ext",
                           extDummy2Path.c_str(), "--args", nothingOutDummyArg.c_str());
        // We could check for the written file, but this test is mainly to check if the right writer is invoked.
    }

    // Clear the streams.
    out.clear();
    err.clear();

    // *************************************************************************
    // Clean the build of the custom IO extensions.
    // *************************************************************************
    // Such that the next invocation of this test case needs to build the
    // extension again, thereby testing again if the extension can be built
    // successfully.
    for (size_t extIdx = 1; extIdx <= numExt; extIdx++) {
        status = runProgram(out, err, "ninja", "ninja", "-C", (extDirPrefix + std::to_string(extIdx)).c_str(), "-t",
                            "clean");
        if (status) {
            // We don't expect any specifc output from ninja, but only if ninja
            // failed, we want to see what it printed to stdout and stderr.
            CHECK(out.str() == "");
            CHECK(err.str() == "");
        }
        CHECK(status == 0);
    }
}