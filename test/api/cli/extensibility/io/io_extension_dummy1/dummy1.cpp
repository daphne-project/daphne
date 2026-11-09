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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/io/FileIOCatalog.h>
#include <runtime/local/io/FileMetaData.h>

#include <iostream>

/**
 * @brief Custom matrix reader for the fictional ".dummy" file format; only supports 0x0 matrices with value type f64.
 */
extern "C" void dummy1_read_DenseMatrix(Structure *&res, const FileMetaData &fmd, const char *filename,
                                        const IOOptions &opts, DaphneContext *ctx) {
    // Show which reader this is.
    std::cout << "hello from dummy1_read_DenseMatrix" << std::endl;

    // Check the meta data.
    if (!fmd.isSingleValueType || fmd.schema[0] != ValueTypeCode::F64)
        throw std::runtime_error("expected an input file with value type f64 for all values");
    if (fmd.numRows != 0 || fmd.numCols != 0)
        throw std::runtime_error("expected an input file with zero rows and columns");

    // Create the empty matrix.
    res = DataObjectFactory::create<DenseMatrix<double>>(0, 0, true);
}

/**
 * @brief Custom matrix writer for the fictional ".dummy" file format.
 */
extern "C" void dummy1_write_DenseMatrix(const Structure *arg, const FileMetaData &fmd, const char *filename,
                                         const IOOptions &opts, DaphneContext *ctx) {
    throw std::runtime_error("not implemented");
}