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

#pragma once

#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/MncSketch.h>

#include <string>
#include <utility>
#include <vector>
#include <optional>

struct HDFSMetaData {
    bool isHDFS = false;
    std::string HDFSFilename;
};

/**
 * @brief Very simple representation of basic file meta data.
 *
 * Currently tailored to frames.
 */
struct FileMetaData {
    const size_t numRows;
    const size_t numCols;
    bool isSingleValueType;
    std::vector<ValueTypeCode> schema;
    std::vector<std::string> labels;
    const ssize_t numNonZeros;
    HDFSMetaData hdfs;
    std::optional<MncSketch> mncSketch;

    /**
     * @brief Construct a new File Meta Data object for Frames
     */
    FileMetaData(size_t numRows, size_t numCols, bool isSingleValueType,
                std::vector<ValueTypeCode> schema,
                std::vector<std::string> labels,
                ssize_t numNonZeros = -1,
                HDFSMetaData hdfs = {},
                std::optional<MncSketch> mncSketch = std::nullopt)
        : numRows(numRows), numCols(numCols), isSingleValueType(isSingleValueType),
        schema(std::move(schema)), labels(std::move(labels)),
        numNonZeros(numNonZeros), hdfs(std::move(hdfs)),
        mncSketch(std::move(mncSketch)) {}

    FileMetaData(size_t numRows, size_t numCols, bool isSingleValueType,
                ValueTypeCode valueType,
                ssize_t numNonZeros = -1,
                HDFSMetaData hdfs = {},
                std::optional<MncSketch> mncSketch = std::nullopt)
        : numRows(numRows), numCols(numCols), isSingleValueType(isSingleValueType),
        numNonZeros(numNonZeros), hdfs(std::move(hdfs)),
        mncSketch(std::move(mncSketch)) {
        schema.emplace_back(valueType);
    }
};
