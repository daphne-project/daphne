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

#include "runtime/local/io/FileIORegistry.h"
#include <runtime/local/io/utils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/Frame.h>

#include <util/preprocessor_defs.h>


#include <type_traits>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct ReadCsvFile {
    static void apply(DTRes *&res, const FileMetaData& fmd, const char *filename, IOOptions &opts, DaphneContext *ctx) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes>
void readCsvFile(DTRes *&res, const FileMetaData& fmd, const char *filename, IOOptions &opts, DaphneContext *ctx) {
    ReadCsvFile<DTRes>::apply(res, fmd, filename, opts, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct ReadCsvFile<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const FileMetaData& fmd, const char *filename, IOOptions &opts, DaphneContext *ctx) {
        File *file = openFile(filename);
        // extract parameters
        size_t numRows = std::stoull(opts.extra.at("numRows"));
        size_t numCols = std::stoull(opts.extra.at("numCols"));
        char delim = ',';
        auto it = opts.extra.find("delimiter");
        if(it != opts.extra.end()) {
            const auto &val = it->second;
            if(val.size() != 1) throw std::runtime_error("Invalid delimiter");
            delim = val[0];
        }
        if (!file) throw std::runtime_error("ReadCsvFile: requires a file (must not be nullptr)");
        if (numRows == 0) throw std::runtime_error("ReadCsvFile: numRows must be > 0");
        if (numCols == 0) throw std::runtime_error("ReadCsvFile: numCols must be > 0");
        if (!res) {
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        }
        size_t cell = 0;
        VT *valuesRes = res->getValues();
        for (size_t r = 0; r < numRows; ++r) {
            if (getFileLine(file) == -1)
                throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");
            size_t pos = 0;
            for (size_t c = 0; c < numCols; ++c) {
                VT val;
                convertCstr(file->line + pos, &val);
                valuesRes[cell++] = val;
                if (c + 1 < numCols) {
                    while (file->line[pos] != delim) ++pos;
                    ++pos;
                }
            }
        }
        closeFile(file);
    }
};

template <>
struct ReadCsvFile<DenseMatrix<std::string>> {
    static void apply(DenseMatrix<std::string> *&res, const FileMetaData& fmd, const char *filename, IOOptions &opts, DaphneContext *ctx) {
        File *file = openFile(filename);
        size_t numRows = std::stoull(opts.extra.at("numRows"));
        size_t numCols = std::stoull(opts.extra.at("numCols"));
        char delim = ',';
        auto it = opts.extra.find("delimiter");
        if(it != opts.extra.end()) {
            const auto &val = it->second;
            if(val.size() != 1) throw std::runtime_error("Invalid delimiter");
            delim = val[0];
        }
        if (!file) throw std::runtime_error("ReadCsvFile: requires a file (must not be nullptr)");
        if (numRows == 0) throw std::runtime_error("ReadCsvFile: numRows must be > 0");
        if (numCols == 0) throw std::runtime_error("ReadCsvFile: numCols must be > 0");
        if (!res) {
            res = DataObjectFactory::create<DenseMatrix<std::string>>(numRows, numCols, false);
        }
        size_t cell = 0;
        std::string *valuesRes = res->getValues();
        for (size_t r = 0; r < numRows; ++r) {
            if (getFileLine(file) == -1)
                throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");
            size_t pos = 0;
            for (size_t c = 0; c < numCols; ++c) {
                std::string val;
                pos = setCString(file, pos, &val, delim) + 1;
                valuesRes[cell++] = std::move(val);
            }
        }
        closeFile(file);
    }
};

template <>
struct ReadCsvFile<DenseMatrix<FixedStr16>> {
    static void apply(DenseMatrix<FixedStr16> *&res, const FileMetaData& fmd, const char *filename, IOOptions &opts, DaphneContext *ctx) {
        File *file = openFile(filename);
        size_t numRows = std::stoull(opts.extra.at("numRows"));
        size_t numCols = std::stoull(opts.extra.at("numCols"));
        char delim = ',';
        auto it = opts.extra.find("delimiter");
        if(it != opts.extra.end()) {
            const auto &val = it->second;
            if(val.size() != 1) throw std::runtime_error("Invalid delimiter");
            delim = val[0];
        }
        if (!file) throw std::runtime_error("ReadCsvFile: requires a file (must not be nullptr)");
        if (numRows == 0) throw std::runtime_error("ReadCsvFile: numRows must be > 0");
        if (numCols == 0) throw std::runtime_error("ReadCsvFile: numCols must be > 0");
        if (!res) {
            res = DataObjectFactory::create<DenseMatrix<FixedStr16>>(numRows, numCols, false);
        }
        size_t cell = 0;
        FixedStr16 *valuesRes = res->getValues();
        for (size_t r = 0; r < numRows; ++r) {
            if (getFileLine(file) == -1)
                throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");
            size_t pos = 0;
            for (size_t c = 0; c < numCols; ++c) {
                std::string tmp;
                pos = setCString(file, pos, &tmp, delim) + 1;
                valuesRes[cell++].set(tmp.c_str());
            }
        }
        closeFile(file);
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT>
struct ReadCsvFile<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const FileMetaData& fmd, const char *filename, IOOptions &opts, DaphneContext *ctx) {
        File *file = openFile(filename);
        size_t numRows = std::stoull(opts.extra.at("numRows"));
        size_t numCols = std::stoull(opts.extra.at("numCols"));
        ssize_t numNonZeros = opts.extra.count("numNonZeros") ? std::stoll(opts.extra.at("numNonZeros")) : -1;
        bool sorted = opts.extra.count("sorted") && opts.extra.at("sorted") == "true";
        char delim = ',';
        auto it = opts.extra.find("delimiter");
        if(it != opts.extra.end()) {
            const auto &val = it->second;
            if(val.size() != 1) throw std::runtime_error("Invalid delimiter");
            delim = val[0];
        }
        if (numNonZeros < 0)
            throw std::runtime_error("ReadCsvFile: sparse reader requires numNonZeros >= 0");
        if (!res)
            res = DataObjectFactory::create<CSRMatrix<VT>>(numRows, numCols, numNonZeros, false);
        if (sorted) {
            readCOOSorted(res, file, numRows, numCols, static_cast<size_t>(numNonZeros), delim);
        } else {
            DenseMatrix<uint64_t> *rowColumnPairs = nullptr;
            readCsvFile(rowColumnPairs, fmd, filename, opts, ctx);
            readCOOUnsorted(res, rowColumnPairs, numRows, numCols, static_cast<size_t>(numNonZeros));
            DataObjectFactory::destroy(rowColumnPairs);
        }
        closeFile(file);
    }

    private:
    static void readCOOSorted(CSRMatrix<VT> *&res, File *file, size_t numRows, [[maybe_unused]] size_t numCols,
                              size_t numNonZeros, char delim) {
        auto *rowOffsets = res->getRowOffsets();
        // we first write number of non zeros for each row and then compute the
        // cumulative sum
        std::memset(rowOffsets, 0, (numRows + 1) * sizeof(size_t));
        auto *colIdxs = res->getColIdxs();
        auto *values = res->getValues();

        size_t pos;
        uint64_t row;
        uint64_t col;
        for (size_t i = 0; i < numNonZeros; ++i) {
            if (getFileLine(file) == -1)
                throw std::runtime_error("ReadCOOSorted::apply: getFileLine failed");
            convertCstr(file->line, &row);
            pos = 0;
            while (file->line[pos] != delim)
                pos++;
            pos++; // skip delimiter
            convertCstr(file->line + pos, &col);

            rowOffsets[row + 1] += 1;
            values[i] = 1;
            colIdxs[i] = col;
        }
        //        #pragma clang loop vectorize(enable)
        PRAGMA_LOOP_VECTORIZE
        for (size_t r = 1; r <= numRows; ++r) {
            rowOffsets[r] += rowOffsets[r - 1];
        }
    }

    static void readCOOUnsorted(CSRMatrix<VT> *&res, DenseMatrix<uint64_t> *rowColumnPairs, size_t numRows,
                                size_t numCols, size_t numNonZeros) {
        // pairs are ordered by first then by second argument (row, then col)
        using RowColPos = std::pair<size_t, size_t>;
        std::priority_queue<RowColPos, std::vector<RowColPos>, std::greater<>> positions;
        for (auto r = 0u; r < rowColumnPairs->getNumRows(); ++r) {
            positions.emplace(rowColumnPairs->get(r, 0), rowColumnPairs->get(r, 1));
        }

        auto *rowOffsets = res->getRowOffsets();
        rowOffsets[0] = 0;
        auto *colIdxs = res->getColIdxs();
        auto *values = res->getValues();
        size_t currValIdx = 0;
        size_t rowIdx = 0;
        while (!positions.empty()) {
            auto pos = positions.top();
            if (pos.first >= res->getNumRows() || pos.second >= res->getNumCols()) {
                throw std::runtime_error("Position [" + std::to_string(pos.first) + ", " + std::to_string(pos.second) +
                                         "] is not part of matrix<" + std::to_string(res->getNumRows()) + ", " +
                                         std::to_string(res->getNumCols()) + ">");
            }
            while (rowIdx < pos.first) {
                rowOffsets[rowIdx + 1] = currValIdx;
                rowIdx++;
            }
            // TODO: valued COO files?
            values[currValIdx] = 1;
            colIdxs[currValIdx] = pos.second;
            currValIdx++;
            positions.pop();
        }
        while (rowIdx < numRows) {
            rowOffsets[rowIdx + 1] = currValIdx;
            rowIdx++;
        }
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <>
struct ReadCsvFile<Frame> {
    static void apply(Frame *&res, const FileMetaData& fmd, const char *filename, IOOptions &opts, DaphneContext *ctx) {
        File *file = openFile(filename);
        size_t numRows = std::stoull(opts.extra.at("numRows"));
        size_t numCols = std::stoull(opts.extra.at("numCols"));
        char delim = ',';
        auto it = opts.extra.find("delimiter");
        if(it != opts.extra.end()) {
            const auto &val = it->second;
            if(val.size() != 1) throw std::runtime_error("Invalid delimiter");
            delim = val[0];
        }
        ValueTypeCode* schema = nullptr;
        if(opts.extra.count("schema")) {
            schema = reinterpret_cast<ValueTypeCode*>(std::stoull(opts.extra.at("schema")));
        }
        if (numRows == 0) throw std::runtime_error("ReadCsvFile: numRows must be > 0");
        if (numCols == 0) throw std::runtime_error("ReadCsvFile: numCols must be > 0");
        if (res == nullptr) {
            res = DataObjectFactory::create<Frame>(numRows, numCols, schema, nullptr, false);
        }

        size_t row = 0, col = 0;

        uint8_t **rawCols = new uint8_t *[numCols];
        ValueTypeCode *colTypes = new ValueTypeCode[numCols];
        for (size_t i = 0; i < numCols; i++) {
            rawCols[i] = reinterpret_cast<uint8_t *>(res->getColumnRaw(i));
            colTypes[i] = res->getColumnType(i);
        }

        while (1) {
            ssize_t ret = getFileLine(file);
            if (file->read == EOF)
                break;
            if (file->line == NULL)
                break;
            if (ret == -1)
                throw std::runtime_error("ReadCsvFile::apply: getFileLine failed");

            size_t pos = 0;
            while (1) {
                switch (colTypes[col]) {
                case ValueTypeCode::SI8:
                    int8_t val_si8;
                    convertCstr(file->line + pos, &val_si8);
                    reinterpret_cast<int8_t *>(rawCols[col])[row] = val_si8;
                    break;
                case ValueTypeCode::SI32:
                    int32_t val_si32;
                    convertCstr(file->line + pos, &val_si32);
                    reinterpret_cast<int32_t *>(rawCols[col])[row] = val_si32;
                    break;
                case ValueTypeCode::SI64:
                    int64_t val_si64;
                    convertCstr(file->line + pos, &val_si64);
                    reinterpret_cast<int64_t *>(rawCols[col])[row] = val_si64;
                    break;
                case ValueTypeCode::UI8:
                    uint8_t val_ui8;
                    convertCstr(file->line + pos, &val_ui8);
                    reinterpret_cast<uint8_t *>(rawCols[col])[row] = val_ui8;
                    break;
                case ValueTypeCode::UI32:
                    uint32_t val_ui32;
                    convertCstr(file->line + pos, &val_ui32);
                    reinterpret_cast<uint32_t *>(rawCols[col])[row] = val_ui32;
                    break;
                case ValueTypeCode::UI64:
                    uint64_t val_ui64;
                    convertCstr(file->line + pos, &val_ui64);
                    reinterpret_cast<uint64_t *>(rawCols[col])[row] = val_ui64;
                    break;
                case ValueTypeCode::F32:
                    float val_f32;
                    convertCstr(file->line + pos, &val_f32);
                    reinterpret_cast<float *>(rawCols[col])[row] = val_f32;
                    break;
                case ValueTypeCode::F64:
                    double val_f64;
                    convertCstr(file->line + pos, &val_f64);
                    reinterpret_cast<double *>(rawCols[col])[row] = val_f64;
                    break;
                case ValueTypeCode::STR: {
                    std::string val_str = "";
                    pos = setCString(file, pos, &val_str, delim);
                    reinterpret_cast<std::string *>(rawCols[col])[row] = val_str;
                    break;
                }
                case ValueTypeCode::FIXEDSTR16: {
                    std::string val_str = "";
                    pos = setCString(file, pos, &val_str, delim);
                    reinterpret_cast<FixedStr16 *>(rawCols[col])[row] = FixedStr16(val_str);
                    break;
                }
                default:
                    throw std::runtime_error("ReadCsvFile::apply: unknown value type code");
                }

                if (++col >= numCols) {
                    break;
                }

                // TODO We could even exploit the fact that the strtoX functions
                // can return a pointer to the first character after the parsed
                // input, then we wouldn't have to search for that ourselves,
                // just would need to check if it is really the delimiter.
                while (file->line[pos] != delim)
                    pos++;
                pos++; // skip delimiter
            }

            if (++row >= numRows) {
                break;
            }
            col = 0;
        }

        delete[] rawCols;
        delete[] colTypes;
        closeFile(file);
    }
};
