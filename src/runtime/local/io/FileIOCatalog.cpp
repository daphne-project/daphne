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

#include <runtime/local/io/FileIOCatalog.h>

#include <stdexcept>
#include <string>

#include <cstddef>
#include <dlfcn.h>

// Helper: Merge a Frame* of column-label → single-row-value into IOOptions
IOOptions mergeOptionsFromFrame(const std::string &ext, PhyDataType dt, const std::string &engine,
                                const Frame *optsFrame, FileIOCatalog &cat) {
    // Ask the catalog for defaults for this (ext, dt, engine).
    // If engine == "", catalog should pick highest-priority impl.
    const IOOptions &defaults = cat.getOptions(ext, dt, engine);

    const size_t numRows = optsFrame->getNumRows();
    const size_t numCols = optsFrame->getNumCols();

    if (optsFrame == nullptr || (numRows == 0 && numCols == 0))
        return defaults;

    if (numRows != 1)
        throw std::runtime_error("the file reader/writer options must be either an empty frame (zero rows and zero "
                                 "columns) or a frame with exactly one row, but found " +
                                 std::to_string(numRows) + " rows");

    IOOptions merged = defaults;

    const std::string *labels = optsFrame->getLabels();
    for (size_t c = 0; c < numCols; ++c) {
        const std::string &key = labels[c];

        // Ignore the key "engine" because it is only used for optionally selecting a particular extension, but cannot
        // be used as the name of an extension-specific option.
        if (key == "engine")
            continue;

        std::string value;
        if (auto *strCol = dynamic_cast<const DenseMatrix<std::string> *>(optsFrame->getColumn<std::string>(c))) {
            value = strCol->get(0, 0);
        } else if (auto *boolCol = dynamic_cast<const DenseMatrix<bool> *>(optsFrame->getColumn<bool>(c))) {
            value = boolCol->get(0, 0) ? "true" : "false";
        } else if (auto *intCol = dynamic_cast<const DenseMatrix<int64_t> *>(optsFrame->getColumn<int64_t>(c))) {
            value = std::to_string(intCol->get(0, 0));
        } else if (auto *floatCol = dynamic_cast<const DenseMatrix<double> *>(optsFrame->getColumn<double>(c))) {
            value = std::to_string(floatCol->get(0, 0));
        } else
            throw std::runtime_error("unsupported column type for option `" + key +
                                     "`, expected either string, bool, int64_t, or double");

        // Only override known plugin options
        auto itKnown = merged.find(key);
        if (itKnown == merged.end()) {
            // silently ignore unknown keys instead of throwing if you prefer:
            // continue;
            throw std::runtime_error("unknown option: `" + key + "`");
        }

        merged[key] = value;
    }

    return merged;
}

// Extract "engine" (and ignore "priority") from the options Frame if present.
// Returns "" if not provided (so catalog picks highest-priority default).
std::string extractEngineFromFrame(const Frame *optsFrame) {
    const size_t numRows = optsFrame->getNumRows();
    const size_t numCols = optsFrame->getNumCols();

    if (optsFrame == nullptr || (numRows == 0 && numCols == 0))
        return "";

    if (numRows != 1)
        throw std::runtime_error("the file reader/writer options must be either an empty frame (zero rows and zero "
                                 "columns) or a frame with exactly one row, but found " +
                                 std::to_string(numRows) + " rows");

    const std::string *labels = optsFrame->getLabels();
    for (size_t c = 0; c < numCols; ++c)
        if (labels[c] == "engine") {
            ValueTypeCode vtc = optsFrame->getColumnType(c);
            if (vtc == ValueTypeCode::STR)
                return optsFrame->getColumn<std::string>(c)->get(0, 0);
            else
                throw std::runtime_error(
                    "the attribute `engine` of the reader/writer options must be of type string, but found `" +
                    ValueTypeUtils::cppNameForCode(vtc) + "`");
        }
    return "";
}