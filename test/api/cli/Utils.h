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

#ifndef TEST_API_CLI_UTILS_H
#define TEST_API_CLI_UTILS_H

#include <api/cli/StatusCode.h>
#include <cerrno>
#include <complex>
#include <runtime/distributed/worker/WorkerImpl.h>

#include <catch.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <grpcpp/server.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/select.h>

/**
 * @brief Reads the entire contents of a plain text file into a string.
 *
 * Not intended to be used with large files.
 *
 * @param filePath The path to the file to be read.
 * @return A string containing the entire contents of the file.
 */
std::string readTextFile(const std::string &filePath);

#ifndef NDEBUG
template <class T> [[maybe_unused]] void LOG(T t) { std::cout << t << std::endl; }

template <class T, class... OtherArgs> void LOG(T &&var, OtherArgs &&...args) {
    std::cout << std::forward<T>(var) << " ";
    LOG(std::forward<OtherArgs>(args)...);
}
#endif

/**
 * @brief Executes the specified program with the given arguments and captures
 * `stdout`, `stderr`, and the status code.
 *
 * @param out The stream where to direct the program's standard output.
 * @param err The stream where to direct the program's standard error.
 * @param execPath The path to the executable.
 * @param args The arguments to pass. Despite the variadic template, each
 * element should be of type `char *`. The first one should be the name of the
 * program itself. The last one does *not* need to be a null pointer.
 * @return The status code returned by the process, or `-1` if it did not exit
 * normally.
 */
template <typename... Args>
int runProgram(std::stringstream &out, std::stringstream &err, const char *execPath, Args... args) {
    constexpr int READ = 0;
    constexpr int WRITE = 1;
    int linkOut[2]; // pipe ends for stdout
    int linkErr[2]; // pipe ends for stderr
    char buf[1024]; // internal buffer for reading from the pipes

    // Try to create the pipes.
    if (pipe(linkOut) == -1)
        throw std::runtime_error("could not create pipe" + std::to_string(errno));
    if (pipe(linkErr) == -1)
        throw std::runtime_error("could not create pipe" + std::to_string(errno));

    // Try to create the child process.
    pid_t p = fork();

    if (p == -1)
        throw std::runtime_error("could not create child process");
    else if (p) { // parent
        // Close write end of pipes.
        close(linkOut[WRITE]);
        close(linkErr[WRITE]);

        // Read data from stdout and stderr of the child from the pipes.
        ssize_t numBytes;
        bool eofOut = false;         // whether EOF has been reached on linkOut
        bool eofErr = false;         // whether EOF has been reached on linkErr
        while (!eofOut || !eofErr) { // as long as at least one out/err has not reached EOF
            // Prepare arguments to select().
            int nfds = 0;
            fd_set readfds;
            FD_ZERO(&readfds);
            if (!eofOut) {
                FD_SET(linkOut[READ], &readfds);
                nfds = std::max(nfds, linkOut[READ]);
            }
            if (!eofErr) {
                FD_SET(linkErr[READ], &readfds);
                nfds = std::max(nfds, linkErr[READ]);
            }

            int retval = select(nfds + 1, &readfds, NULL, NULL, NULL);

            if (retval == -1)
                throw std::runtime_error("select() failed");
            // retval == 0 should never happen, since we always provide at least one file descriptor to select() and
            // don't use a timeout.
            else {
                // Read from the child's stdout, if data is available.
                if (FD_ISSET(linkOut[READ], &readfds)) {
                    numBytes = read(linkOut[READ], buf, sizeof(buf));
                    if (numBytes == 0)
                        eofOut = true;
                    else if (numBytes == -1)
                        throw std::runtime_error("error reading from the child's stdout");
                    else
                        out.write(buf, numBytes);
                }
                // Read from the child's stderr, if data is available.
                if (FD_ISSET(linkErr[READ], &readfds)) {
                    numBytes = read(linkErr[READ], buf, sizeof(buf));
                    if (numBytes == 0)
                        eofErr = true;
                    else if (numBytes == -1)
                        throw std::runtime_error("error reading from the child's stderr");
                    else
                        err.write(buf, numBytes);
                }
            }
        }

        close(linkOut[READ]);
        close(linkErr[READ]);
        // Wait for child's termination.
        int status;
        waitpid(p, &status, 0);
        if (status != 0) {
#ifndef NDEBUG
            std::cout << "stdout: " << out.str() << std::endl;
            std::cout << "stderr: " << err.str() << std::endl;
            std::cout << "status: " << status << std::endl;
            LOG(args...);
#endif
        }
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    } else { // child
        // Redirect stdout and stderr to the pipe.
        dup2(linkOut[WRITE], STDOUT_FILENO);
        dup2(linkErr[WRITE], STDERR_FILENO);
        close(linkOut[READ]);
        close(linkOut[WRITE]);
        close(linkErr[READ]);
        close(linkErr[WRITE]);

        // Execute other program.
        // If execPath is a path (contains "/") use execl, otherwise use execlp.
        // We need this to support "mpirun" for the MPI test cases.
        if (std::string(execPath).find("/") != std::string::npos)
            execl(execPath, args..., static_cast<char *>(nullptr));
        else
            execlp(execPath, args..., static_cast<char *>(nullptr));

        // execl does not return, unless it failed.
        throw std::runtime_error("could not execute the program");
    }
}

/**
 * @brief Executes the specified program with the given arguments in the
 * background.
 *
 * @param out The file descriptor where to redirect standard output.
 * @param err The file descriptor where to redirect standard error.
 * @param execPath The path to the executable.
 * @param args The arguments to pass. Despite the variadic template, each
 * element should be of type `char *`. The first one should be the name of the
 * program itself. The last one does *not* need to be a null pointer.
 * @return The process id of the child process.
 *
 */
template <typename... Args> pid_t runProgramInBackground(int &out, int &err, const char *execPath, Args... args) {

    // Try to create the child process.
    pid_t p = fork();

    if (p == -1)
        throw std::runtime_error("could not create child process");
    else if (p) { // parent
        // Return pid
        return p;
    } else { // child
        // Redirect stdout and stderr to the pipe.
        dup2(out, STDOUT_FILENO);
        dup2(err, STDERR_FILENO);
        // Execute other program.
        execl(execPath, args..., static_cast<char *>(nullptr));

        // execl does not return, unless it failed.
        throw std::runtime_error("could not execute the program");
    }
}

/**
 * @brief Executes the "run-lit.py" Python script in a directory and
 * captures `stdout`, `stderr`, and the status code.
 *
 * "run-lit.py" is required to run the LLVM tool llvm-lit in order to
 * test "*.mlir" files in the directoy using the llvm-lit command RUN:
 * in each file.
 *
 * @param out The stream where to direct the program's standard output.
 * @param err The stream where to direct the program's standard error.
 * @param dirPath The path to the directory containing the "run-lit.py" script
 * and the "*.mlir" test cases.
 * @param args The arguments to pass in addition to the script's path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 * @return The status code returned by the process, or `-1` if it did not exit
 * normally.
 */
template <typename... Args>
int runLIT(std::stringstream &out, std::stringstream &err, std::string dirPath, Args... args) {
    return runProgram(out, err, "/bin/python3", "python3", (dirPath + "run-lit.py").c_str(), "-v", dirPath.c_str(),
                      args...);
}

/**
 * @brief Executes DAPHNE's command line interface with the given arguments and
 * captures `stdout`, `stderr`, and the status code.
 *
 * @param out The stream where to direct the program's standard output.
 * @param err The stream where to direct the program's standard error.
 * @param args The arguments including the script file. Despite the variadic
 * template, each element should be of type `char *`. The last one does *not*
 * need to be a null pointer.
 * @return The status code returned by the process, or `-1` if it did not exit
 * normally.
 */
template <typename... Args> int runDaphne(std::stringstream &out, std::stringstream &err, Args... args) {
    return runProgram(out, err, "bin/daphne", "daphne", args...);
}

/**
 * @brief Executes the given Python script with the `python3` interpreter and
 * captures `stdout`, `stderr`, and the status code.
 *
 * Typically the Python script will use DaphneLib, the Python API of DAPHNE.
 *
 * @param out The stream where to direct the program's standard output.
 * @param err The stream where to direct the program's standard error.
 * @param scriptPath The path to the Python script file to execute.
 * @param args The arguments to pass in addition to the script's path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 * @return The status code returned by the process, or `-1` if it did not exit
 * normally.
 */
template <typename... Args>
int runDaphneLib(std::stringstream &out, std::stringstream &err, const char *scriptPath, Args... args) {
    return runProgram(out, err, "/bin/python3", "python3", scriptPath, args...);
}

/**
 * @brief Checks whether executing the given DaphneDSL script with the command
 * line interface of DAPHNE returns the given status code.
 *
 * @param exp The expected status code.
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * @param args The arguments to pass in addition to the script's path. Note
 * that script arguments must be passed via the `--args` option for this
 * utility function. Despite the variadic template, each element should be of
 * type `char *`. The last one does *not* need to be a null pointer.
 */
template <typename... Args>
void checkDaphneStatusCode(StatusCode exp, const std::string &scriptFilePath, Args... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, args..., scriptFilePath.c_str());

    CHECK(status == exp);
}

template <typename... Args>
void checkDaphneStatusCodeSimple(StatusCode exp, const std::string &dirPath, const std::string &name, unsigned idx,
                                 Args... args) {
    checkDaphneStatusCode(exp, dirPath + name + '_' + std::to_string(idx) + ".daphne", args...);
}

/**
 * @brief Checks whether executing the given DaphneDSL script with the command
 * line interface of DAPHNE fails.
 *
 * This is the case when the return code is not `StatusCode::SUCCESS`.
 *
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * @param args The arguments to pass in addition to the script's path. Note
 * that script arguments must be passed via the `--args` option for this
 * utility function. Despite the variadic template, each element should be of
 * type `char *`. The last one does *not* need to be a null pointer.
 */
template <typename... Args> void checkDaphneFails(const std::string &scriptFilePath, Args... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, args..., scriptFilePath.c_str());

    CHECK(status != StatusCode::SUCCESS);
}

template <typename... Args>
void checkDaphneFailsSimple(const std::string &dirPath, const std::string &name, unsigned idx, Args... args) {
    checkDaphneFails(dirPath + name + '_' + std::to_string(idx) + ".daphne", args...);
}

/**
 * @brief Compares the standard output of executing the given DaphneDSL script
 * with the command line interface of DAPHNE to a reference text.
 *
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 *
 * @param exp The expected output on stdout.
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * @param args The arguments to pass in addition to the script's path. Note
 * that script arguments must be passed via the `--args` option for this
 * utility function. Despite the variadic template, each element should be of
 * type `char *`. The last one does *not* need to be a null pointer.
 */
template <typename... Args>
void compareDaphneToStr(const std::string &exp, const std::string &scriptFilePath, Args... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, args..., scriptFilePath.c_str());

    // Just CHECK (don't REQUIRE) success, such that in case of a failure, the
    // checks of out and err still run and provide useful messages. For err,
    // don't check empty(), because then catch2 doesn't display the error
    // output.
    CHECK(status == StatusCode::SUCCESS);
    CHECK(out.str() == exp);
    CHECK(err.str() == "");
}

/**
 * @brief Compares the standard output of executing the given Python/DaphneLib
 * script to a reference text.
 *
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 *
 * @param exp The expected output on stdout.
 * @param scriptFilePath The path to the Python/DaphneLib script file to
 * execute.
 * @param args The arguments to pass in addition to the script's path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 */
template <typename... Args>
void compareDaphneLibToStr(const std::string &exp, const std::string &scriptFilePath, Args... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphneLib(out, err, scriptFilePath.c_str(), args...);

    // Just CHECK (don't REQUIRE) success, such that in case of a failure, the
    // checks of out and err still run and provide useful messages. For err,
    // don't check empty(), because then catch2 doesn't display the error
    // output.
    CHECK(status == StatusCode::SUCCESS);
    CHECK(out.str() == exp);
    CHECK(err.str() == "");
}

/**
 * @brief Checks if the numerical values in the standard output of the two given
 * DaphneDSL script runs are within a relative distance to a reference text.
 *
 * @param left The output from a runDaphne().
 * @param right The output from a runDaphne().
 * @param ignore_lines How many lines in the beginning of the DaphneDSL output
 * do contain numerical values to compare.
 * @param epsilon The relative error that is acceptable.
 */
template <typename... Args>
void compareDaphneRunsNumerically(std::stringstream &left, std::stringstream &right, const int ignore_lines,
                                  const long double epsilon) {
    std::string s_left;
    std::string s_right;
    float f_left{0};
    float f_right{0};
    for (auto i = 0; i != ignore_lines; i++) {
        std::getline(left, s_left);
        std::getline(right, s_right);
    }
    bool correct_so_far = true;

    while (std::getline(left, s_left, ' ') && std::getline(right, s_right, ' ') && correct_so_far) {
        try {
            // Long double just to be sure
            f_left = std::stold(s_left);
            f_right = std::stold(s_right);
        } catch (std::invalid_argument const &) {
            FAIL("The result does not have the right number of outputs.\nLeft=" << left.str()
                                                                                << "\t Right=" << right.str());
        }
        correct_so_far = std::norm(f_left - f_right) < epsilon * std::norm(f_left);
    }
    CHECK(correct_so_far == true);
}

/**
 * @brief Checks if the numerical values in the standard output of the given
 * DaphneDSL script run with the command line interface of DAPHNE
 * are within a relative distance to a reference text.
 *
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 *
 * @param exp The expected output on stdout.
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * output.
 * @param ignore_lines How many lines in the beginning of the DaphneDSL output
 * do contain numerical values to compare.
 * @param epsilon The relative error that is acceptable.
 * @param args The arguments to pass in addition to the script's path. Note
 * that script arguments must be passed via the `--args` option for this
 * utility function. Despite the variadic template, each element should be of
 * type `char *`. The last one does *not* need to be a null pointer.
 */
template <typename... Args>
void compareDaphneToStringNumerically(const std::string &exp, const std::string &scriptFilePath, int ignore_lines,
                                      long double epsilon, Args... args) {
    std::stringstream out;
    std::stringstream err;
    int status = runDaphne(out, err, args..., scriptFilePath.c_str());
    CHECK(status == StatusCode::SUCCESS);
    std::stringstream exp_ss(exp);
    compareDaphneRunsNumerically(exp_ss, out, ignore_lines, epsilon);
    CHECK(err.str() == "");
}

/**
 * @brief Compares the standard output of executing the given DaphneDSL script
 * with the command line interface of DAPHNE to a reference text
 * file.
 *
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 *
 * @param refFilePath The path to the plain text file containing the reference
 * @param scriptFilePath The path to the DaphneDSL script file to execute.
 * output.
 * @param args The arguments to pass in addition to the script's path. Note
 * that script arguments must be passed via the `--args` option for this
 * utility function. Despite the variadic template, each element should be of
 * type `char *`. The last one does *not* need to be a null pointer.
 */
template <typename... Args>
void compareDaphneToRef(const std::string &refFilePath, const std::string &scriptFilePath, Args... args) {
    return compareDaphneToStr(readTextFile(refFilePath), scriptFilePath, args...);
}

/**
 * @brief Compares the standard output of the given DaphneDSL script with that
 * of the given Python/DaphneLib script.
 *
 * Also checks that the status codes indicate a successful execution for both
 * and that nothing was printed to standard error.
 *
 * @param pythonScriptFilePath
 * @param daphneDSLScriptFilePath
 * @param args The arguments to pass in addition to the scripts' path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 */
template <typename... Args>
void compareDaphneToDaphneLib(const std::string &pythonScriptFilePath, const std::string &daphneDSLScriptFilePath,
                              Args... args) {
    std::stringstream outDaphne;
    std::stringstream errDaphne;
    std::stringstream outDaphneLib;
    std::stringstream errDaphneLib;
    int statusDaphneLib = runDaphneLib(outDaphneLib, errDaphneLib, pythonScriptFilePath.c_str(), args...);
    int statusDaphne = runDaphne(outDaphne, errDaphne, daphneDSLScriptFilePath.c_str(), args...);

    // Just CHECK (don't REQUIRE) success, such that in case of a failure, the
    // checks of out and err still run and provide useful messages. For err,
    // don't check empty(), because then catch2 doesn't display the error
    // output.
    CHECK(statusDaphne == StatusCode::SUCCESS);
    CHECK(statusDaphneLib == 0);
    CHECK(outDaphne.str() == outDaphneLib.str());
    CHECK(errDaphne.str() == "");
    CHECK(errDaphneLib.str() == "");
}

/**
 * @brief Approximate floating point comparison of each line in the standard
 * output of the given DaphneDSL script with that of the given Python/DaphneLib
 * script.
 *
 * Also checks that the status codes indicate a successful execution for both
 * and that nothing was printed to standard error.
 *
 * @param pythonScriptFilePath
 * @param daphneDSLScriptFilePath
 * @param args The arguments to pass in addition to the scripts' path. Despite
 * the variadic template, each element should be of type `char *`. The last one
 * does *not* need to be a null pointer.
 */
template <typename... Args>
void compareDaphneToDaphneLibScalar(const std::string &pythonScriptFilePath, const std::string &daphneDSLScriptFilePath,
                                    Args... args) {
    std::stringstream outDaphne;
    std::stringstream errDaphne;
    std::stringstream outDaphneLib;
    std::stringstream errDaphneLib;
    std::string resultDaphne, resultDaphneLib;
    float epsilon = 0.1;
    int statusDaphneLib = runDaphneLib(outDaphneLib, errDaphneLib, pythonScriptFilePath.c_str(), args...);
    int statusDaphne = runDaphne(outDaphne, errDaphne, args..., daphneDSLScriptFilePath.c_str());

    // Just CHECK (don't REQUIRE) success, such that in case of a failure, the
    // checks of out and err still run and provide useful messages. For err,
    // don't check empty(), because then catch2 doesn't display the error
    // output.
    CHECK(statusDaphne == StatusCode::SUCCESS);
    CHECK(statusDaphneLib == 0);
    while (std::getline(outDaphneLib, resultDaphneLib) && std::getline(outDaphne, resultDaphne)) {
        CHECK(std::stof(resultDaphneLib) - std::stof(resultDaphne) <= epsilon);
    }
    CHECK(errDaphne.str() == "");
    CHECK(errDaphneLib.str() == "");
}

template <typename... Args>
void compareDaphneToRefSimple(const std::string &dirPath, const std::string &name, unsigned idx, Args... args) {
    const std::string filePath = dirPath + name + '_' + std::to_string(idx);
    compareDaphneToRef(filePath + ".txt", filePath + ".daphne", args...);
}

template <typename... Args>
void compareDaphneToDaphneLibSimple(const std::string &dirPath, const std::string &name, unsigned idx, Args... args) {
    const std::string filePath = dirPath + name + '_' + std::to_string(idx);
    compareDaphneToDaphneLib(filePath + ".py", args..., filePath + ".daphne");
}

/**
 * @brief Compares the standard output of executing a given DaphneDSL script
 * with the command line interface of DAPHNE, to a (simpler) DaphneDSL
 * script defining the expected behaviour.
 *
 * Also checks that the status code indicates a successful execution and that
 * nothing was printed to standard error.
 *
 * @param expScriptFilePath The path to the DaphneDSL script with expected
 * behaviour.
 * @param actScriptFilePath The path to the DaphneDSL script to check with
 * actual behaviour.
 * @param args The arguments to pass in addition to the script's path. Note
 * that script arguments must be passed via the `--args` option for this
 * utility function. Despite the variadic template, each element should be of
 * type `char *`. The last one does *not* need to be a null pointer.
 */
template <typename... Args>
void compareDaphneToSelfRef(const std::string &expScriptFilePath, const std::string &actScriptFilePath, Args... args) {
    std::stringstream expOut;
    std::stringstream expErr;
    int expStatus = runDaphne(expOut, expErr, args..., expScriptFilePath.c_str());
    std::stringstream actOut;
    std::stringstream actErr;
    int actStatus = runDaphne(actOut, actErr, args..., actScriptFilePath.c_str());

    // Just CHECK (don't REQUIRE) success, such that in case of a failure, the
    // checks of out and err still run and provide useful messages.
    CHECK(expStatus == actStatus);
    CHECK(expOut.str() == actOut.str());
    CHECK(expErr.str() == actErr.str());
}

template <typename... Args>
[[maybe_unused]] void compareDaphneToSelfRefSimple(const std::string &dirPath, const std::string &name, unsigned idx,
                                                   Args... args) {
    const std::string filePath = dirPath + name + '_' + std::to_string(idx);
    compareDaphneToSelfRef(filePath + ".ref.daphne", filePath + ".daphne", args...);
}

/**
 * @brief Compares the standard output of executing a given DaphneDSL script
 * with a reference script or text file, based on which file is found.
 *
 * @param dirPath
 * @param name
 * @param idx
 * @param args The arguments to pass in addition to the script's path. Note
 * that script arguments must be passed via the `--args` option for this
 * utility function. Despite the variadic template, each element should be of
 * type `char *`. The last one does *not* need to be a null pointer.
 */
template <typename... Args>
void compareDaphneToSomeRefSimple(const std::string &dirPath, const std::string &name, unsigned idx, Args... args) {
    const std::string filePath = dirPath + name + '_' + std::to_string(idx);
    if (std::filesystem::exists(filePath + ".ref.daphne")) {
        compareDaphneToSelfRef(filePath + ".ref.daphne", filePath + ".daphne", args...);
    } else if (std::filesystem::exists(filePath + ".txt")) {
        compareDaphneToRef(filePath + ".txt", filePath + ".daphne", args...);
    } else {
        throw std::runtime_error("Could not find any ref for file `" + filePath + ".daphne`");
    }
}

// TODO Ideally, we shouldn't need that. There should be a way to print data
// objects without technical information such as their physical data
// representation.
/**
 * @brief Replaces all occurrences of "DenseMatrix" and "CSRMatrix" in the
 * given string by "<SomeMatrix>".
 *
 * Can be used to prepare the outputs of a DaphneDSL script with two different
 * sets of arguments for string comparison.
 *
 * @param str
 * @return
 */
std::string generalizeDataTypes(const std::string &str);

/**
 * @brief Checks if the given string represents a floating-point number acceptable by `std::stod()` and nothing else (no
 * leading whitespace, no trailing characters).
 *
 * @param str The string to check.
 * @param res The parsed double value (if this function returns `true`) or unspecified (otherwise).
 * @return `true` if the given string fulfills the condition; or `false`, otherwise.
 */
bool isFloatRepr(const std::string &str, double &res);

#endif // TEST_API_CLI_UTILS_H
