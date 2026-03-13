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

#ifndef SRC_PARSER_METADATA_JSONPARAMS_H
#define SRC_PARSER_METADATA_JSONPARAMS_H

#include <string>

/**
 * @brief A container that contains names of JSON keys for a file
 * metadata.
 *
 */
struct JsonKeys {

    // mandatory keys
    inline static const std::string NUM_ROWS = "numRows"; // int
    inline static const std::string NUM_COLS = "numCols"; // int

    // should always contain exactly one of the following keys
    inline static const std::string VALUE_TYPE = "valueType"; // string
    inline static const std::string SCHEMA = "schema";        // array of objects

    // optional key
    inline static const std::string NUM_NON_ZEROS = "numNonZeros"; // int (default: -1)
    inline static const std::string HDFS = "hdfs";                 // json
    struct HDFSKeys {
        inline static const std::string isHDFS = "isHDFS";
        inline static const std::string HDFSFilename = "HDFSFilename";
    };
    // MNC sketch metadata (optional)
    inline static const std::string MNC_SKETCH = "mncSketch"; // json object

    struct MNCKeys {
        inline static const std::string M = "m";
        inline static const std::string N = "n";

        // core counts
        inline static const std::string HR = "hr"; // array<uint32>
        inline static const std::string HC = "hc"; // array<uint32>

        // extended counts (optional, should appear together)
        inline static const std::string HER = "her"; // array<uint32>
        inline static const std::string HEC = "hec"; // array<uint32>

        // summary stats (optional)
        inline static const std::string MAX_HR = "maxHr";
        inline static const std::string MAX_HC = "maxHc";
        inline static const std::string NNZ_ROWS = "nnzRows";
        inline static const std::string NNZ_COLS = "nnzCols";
        inline static const std::string ROWS_EQ1 = "rowsEq1";
        inline static const std::string COLS_EQ1 = "colsEq1";
        inline static const std::string ROWS_GT_HALF = "rowsGtHalf";
        inline static const std::string COLS_GT_HALF = "colsGtHalf";

        // flags (optional)
        inline static const std::string IS_DIAGONAL = "isDiagonal";
    };

};

#endif