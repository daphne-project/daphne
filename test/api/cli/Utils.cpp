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

#include <api/cli/Utils.h>
#include <runtime/distributed/worker/WorkerImpl.h>

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cctype>
#include <cstring>

std::string readTextFile(const std::string &filePath) {
    std::ifstream ifs(filePath, std::ios::in);
    if (!ifs.good())
        throw std::runtime_error("could not open file '" + filePath + "'");

    std::stringstream stream;
    stream << ifs.rdbuf();

    return stream.str();
}

std::string generalizeDataTypes(const std::string &str) {
    std::regex re("(DenseMatrix|CSRMatrix)");
    return std::regex_replace(str, re, "<SomeMatrix>");
}

bool isFloatRepr(const std::string &str, double &res) {
    // An empty string obviously doesn't represent any floating-point number.
    if (str.empty())
        return false;
    // std::stod() skips any whitespace at the beginning of the string, but we don't want to tolerate such whitespace.
    if (std::isspace(str[0]))
        return false;
    try {
        // Try to apply std::stod().
        size_t pos;
        res = std::stod(str, &pos);

        // std::stod() must have consumed the entire string, because we don't want to tolerate any left-over characters
        // after the floating-point number.
        if (pos != std::strlen(str.c_str()))
            return false;
    } catch (...) {
        // Applying std::stod() to the string must not throw.
        return false;
    }
    // All checks passed.
    return true;
}