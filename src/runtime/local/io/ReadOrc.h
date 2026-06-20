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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/io/FileMetaData.h>

#include <arrow/api.h>
#include <arrow/adapters/orc/adapter.h>
#include <arrow/io/file.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

// ****************************************************************************
// readOrc — reader for Apache ORC.
//
// Signature matches the reader/writer extensibility interface introduced in
// PR #993 (daphne-eu/daphne). Until PR #993 is merged, this function is
// invoked from the if/else dispatcher in `Read.h` based on the `.orc` file
// extension.
//
// Uses Apache Arrow's ORC adapter (arrow::adapters::orc::ORCFileReader),
// which requires Arrow to be built with -DARROW_ORC=ON (see build.sh).
// ****************************************************************************

namespace daphne_orc_detail {

inline std::shared_ptr<arrow::Table> openAndReadOrc(const char *filename) {
    auto fileOr = arrow::io::ReadableFile::Open(std::string(filename));
    if (!fileOr.ok())
        throw std::runtime_error(std::string("ORC reader: failed to open file '") + filename + "' (" +
                                 fileOr.status().ToString() + ")");

    auto readerOr =
        arrow::adapters::orc::ORCFileReader::Open(fileOr.ValueOrDie(), arrow::default_memory_pool());
    if (!readerOr.ok())
        throw std::runtime_error(std::string("ORC reader: failed to parse ORC structure of '") + filename + "' (" +
                                 readerOr.status().ToString() + ")");

    auto tableOr = readerOr.ValueOrDie()->Read();
    if (!tableOr.ok())
        throw std::runtime_error(std::string("ORC reader: failed to read ORC data from '") + filename + "' (" +
                                 tableOr.status().ToString() + ")");

    return tableOr.ValueOrDie();
}

inline void validateShape(const std::shared_ptr<arrow::Table> &table, const FileMetaData &fmd) {
    if (static_cast<size_t>(table->num_rows()) != fmd.numRows)
        throw std::runtime_error("ORC reader: row count mismatch — meta says " + std::to_string(fmd.numRows) +
                                 ", file has " + std::to_string(table->num_rows()));
    if (static_cast<size_t>(table->num_columns()) != fmd.numCols)
        throw std::runtime_error("ORC reader: column count mismatch — meta says " + std::to_string(fmd.numCols) +
                                 ", file has " + std::to_string(table->num_columns()));
}

inline void expectArrowType(const std::shared_ptr<arrow::Table> &table, size_t colIdx, arrow::Type::type expectedId,
                            const char *expectedLabel) {
    const auto &actualType = table->schema()->field(static_cast<int>(colIdx))->type();
    if (actualType->id() != expectedId)
        throw std::runtime_error("ORC reader: column " + std::to_string(colIdx) + " type mismatch — expected " +
                                 expectedLabel + ", got " + actualType->ToString());
}

inline void rejectStringsInSchema(const FileMetaData &fmd) {
    for (size_t i = 0; i < fmd.schema.size(); ++i)
        if (fmd.schema[i] == ValueTypeCode::STR)
            throw std::runtime_error("ORC reader: reading string-valued ORC files is not supported (yet)");
}

inline void rejectNulls(const std::shared_ptr<arrow::Table> &table) {
    for (int c = 0; c < table->num_columns(); ++c)
        if (table->column(c)->null_count() > 0)
            throw std::runtime_error("ORC reader: null values not supported (yet) — column " + std::to_string(c));
}

// Copy a chunked column into a row-strided DenseMatrix destination
// (column index `c`, with row stride `numCols`).
template <typename ArrowArrayT, typename VT>
inline void copyChunkedToDense(const std::shared_ptr<arrow::ChunkedArray> &chunked, VT *vals, size_t c,
                               size_t numCols) {
    size_t row = 0;
    for (int ci = 0; ci < chunked->num_chunks(); ++ci) {
        auto arr = std::static_pointer_cast<ArrowArrayT>(chunked->chunk(ci));
        const auto *raw = arr->raw_values();
        const int64_t n = arr->length();
        for (int64_t i = 0; i < n; ++i, ++row)
            vals[row * numCols + c] = raw[i];
    }
}

// Copy a chunked column into a contiguous (column-major) destination buffer.
template <typename ArrowArrayT, typename VT>
inline void copyChunkedToContig(const std::shared_ptr<arrow::ChunkedArray> &chunked, VT *dst) {
    size_t row = 0;
    for (int ci = 0; ci < chunked->num_chunks(); ++ci) {
        auto arr = std::static_pointer_cast<ArrowArrayT>(chunked->chunk(ci));
        const int64_t n = arr->length();
        std::memcpy(dst + row, arr->raw_values(), static_cast<size_t>(n) * sizeof(VT));
        row += static_cast<size_t>(n);
    }
}

} // namespace daphne_orc_detail

// ----------------------------------------------------------------------------
// readOrc — entry point matching PR #993 reader signature.
// ----------------------------------------------------------------------------

inline void readOrc(void *res, const FileMetaData &fmd, const char *filename,
                    const std::map<std::string, std::string> &options, DaphneContext *ctx) {
    (void)options; // reserved for future push-down hints (column selection, etc.)
    (void)ctx;

    daphne_orc_detail::rejectStringsInSchema(fmd);
    auto table = daphne_orc_detail::openAndReadOrc(filename);
    daphne_orc_detail::validateShape(table, fmd);
    daphne_orc_detail::rejectNulls(table);

    if (fmd.isSingleValueType) {
        const ValueTypeCode vt = fmd.schema.empty() ? ValueTypeCode::F64 : fmd.schema[0];

        // ----- DenseMatrix<double> -----
        if (vt == ValueTypeCode::F64) {
            for (size_t c = 0; c < fmd.numCols; ++c)
                daphne_orc_detail::expectArrowType(table, c, arrow::Type::DOUBLE, "F64");

            auto **out = reinterpret_cast<DenseMatrix<double> **>(res);
            double *vals = (*out)->getValues();
            for (size_t c = 0; c < fmd.numCols; ++c)
                daphne_orc_detail::copyChunkedToDense<arrow::DoubleArray, double>(table->column(static_cast<int>(c)),
                                                                                  vals, c, fmd.numCols);
            return;
        }

        // ----- DenseMatrix<int64_t> -----
        if (vt == ValueTypeCode::SI64) {
            for (size_t c = 0; c < fmd.numCols; ++c)
                daphne_orc_detail::expectArrowType(table, c, arrow::Type::INT64, "SI64");

            auto **out = reinterpret_cast<DenseMatrix<int64_t> **>(res);
            int64_t *vals = (*out)->getValues();
            for (size_t c = 0; c < fmd.numCols; ++c)
                daphne_orc_detail::copyChunkedToDense<arrow::Int64Array, int64_t>(table->column(static_cast<int>(c)),
                                                                                  vals, c, fmd.numCols);
            return;
        }

        throw std::runtime_error("ORC reader: value type not supported by ORC reader (yet) — only F64 and SI64 are "
                                 "implemented");
    }

    // ----- Frame: per-column copy, type-dispatched per column -----
    auto **outFrame = reinterpret_cast<Frame **>(res);
    Frame *f = *outFrame;

    for (size_t c = 0; c < fmd.numCols; ++c) {
        const ValueTypeCode vtc = fmd.schema[c];
        if (vtc == ValueTypeCode::F64) {
            daphne_orc_detail::expectArrowType(table, c, arrow::Type::DOUBLE, "F64");
            double *buf = f->getColumn<double>(c)->getValues();
            daphne_orc_detail::copyChunkedToContig<arrow::DoubleArray, double>(table->column(static_cast<int>(c)),
                                                                               buf);
        } else if (vtc == ValueTypeCode::SI64) {
            daphne_orc_detail::expectArrowType(table, c, arrow::Type::INT64, "SI64");
            int64_t *buf = f->getColumn<int64_t>(c)->getValues();
            daphne_orc_detail::copyChunkedToContig<arrow::Int64Array, int64_t>(table->column(static_cast<int>(c)),
                                                                               buf);
        } else {
            throw std::runtime_error("ORC reader: value type not supported by ORC reader (yet) — only F64 and SI64 "
                                     "are implemented (column " +
                                     std::to_string(c) + ")");
        }
    }
}
