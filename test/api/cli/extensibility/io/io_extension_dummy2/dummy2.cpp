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

#include <fstream>
#include <iostream>
#include <map>
#include <string>

/**
 * @brief Prints the given reader/writer options as a comma-separated list of key-value pairs, sorted lexicographically
 * by key.
 */
void printOptions(const IOOptions &opts) {
    // Ensure that the key-value pairs are sorted lexicographically by key, no matter what internal representation is
    // used for IOOptions.
    std::map<std::string, std::string> optsSorted;
    optsSorted.insert(opts.begin(), opts.end());

    // Print the options.
    std::cout << "options: ";
    bool first = true;
    for (const auto &[opt, val] : optsSorted) {
        std::cout << (first ? "" : ",") << opt << '=' << '"' << val << '"';
        first = false;
    }
    std::cout << std::endl;
}

/**
 * @brief Custom matrix reader for the fictional ".dummy" file format; only supports 0x0 matrices with value type f64.
 */
extern "C" void dummy2_read_DenseMatrix(Structure *&res, const FileMetaData &fmd, const char *filename,
                                        const IOOptions &opts, DaphneContext *ctx) {
    // Show which reader this is and which options it received.
    std::cout << "hello from dummy2_read_DenseMatrix, ";
    printOptions(opts);

    // Check the meta data.
    if (!fmd.isSingleValueType || fmd.schema[0] != ValueTypeCode::F64)
        throw std::runtime_error("expected an input file with value type f64 for all values");
    if (fmd.numRows != 0 || fmd.numCols != 0)
        throw std::runtime_error("expected an input file with zero rows and columns");

    // Create the empty matrix.
    res = DataObjectFactory::create<DenseMatrix<double>>(0, 0, true);
}

/**
 * @brief Custom matrix writer for the fictional ".dummy" file format; only supports 0x0 matrices with value type f64.
 */
extern "C" void dummy2_write_DenseMatrix(const Structure *arg, const FileMetaData &fmd, const char *filename,
                                         const IOOptions &opts, DaphneContext *ctx) {
    // Show which writer this is.
    std::cout << "hello from dummy2_write_DenseMatrix" << std::endl;

    // Check the input.
    auto mat = dynamic_cast<const DenseMatrix<double> *>(arg);
    if (!mat)
        throw std::runtime_error("expected a DenseMatrix<double> as input");
    if (mat->getNumRows() != 0 || mat->getNumCols() != 0)
        throw std::runtime_error("expected a matrix with zero rows and columns as input");

    // Write the file.
    std::ofstream f(filename);
    f << "nothing" << std::endl;
    if (!f)
        throw std::runtime_error("write failed");
}

/**
 * @brief Custom frame reader for the fictional ".dummy" file format; only supports 0x0 frames.
 */
extern "C" void dummy2_read_Frame(Structure *&res, const FileMetaData &fmd, const char *filename, const IOOptions &opts,
                                  DaphneContext *ctx) {
    // Show which reader this is.
    std::cout << "hello from dummy2_read_Frame" << std::endl;

    // Check the meta data.
    if (fmd.numRows != 0 || fmd.numCols != 0)
        throw std::runtime_error("expected an input file with zero rows and columns");

    // Create the empty frame.
    ValueTypeCode schema[0];
    std::string labels[0];
    res = DataObjectFactory::create<Frame>(0, 0, schema, labels, true);
}

/**
 * @brief Custom frame writer for the fictional ".dummy" file format.
 */
extern "C" void dummy2_write_Frame(const Structure *arg, const FileMetaData &fmd, const char *filename,
                                   const IOOptions &opts, DaphneContext *ctx) {
    throw std::runtime_error("not implemented");
}