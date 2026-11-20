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

#include <orc/OrcFile.hh>
#include <orc/Reader.hh>
#include <orc/Type.hh>
#include <orc/Vector.hh>
#include <orc/sargs/SearchArgument.hh>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ****************************************************************************
// readOrc - reader for Apache ORC.
//
// Signature matches the reader/writer extensibility interface introduced in
// PR #993 (daphne-eu/daphne). Until PR #993 is merged, this function is
// invoked from the if/else dispatcher in `Read.h` based on the `.orc` file
// extension.
//
// Internally uses Apache ORC's C++ library directly (not Arrow's ORC adapter,
// since the dev container's Arrow is built without -DARROW_ORC=ON).
// ****************************************************************************

namespace daphne_orc_detail {

inline std::unique_ptr<orc::Reader> openOrcReader(const char *filename) {
    try {
        orc::ReaderOptions opts;
        return orc::createReader(orc::readLocalFile(std::string(filename)), opts);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("ORC reader: failed to open file '") + filename + "' (" + e.what() + ")");
    }
}

// Parsed representation of a single-clause predicate:
//   <col> <op> <literal>
// Only one clause is supported. Compound predicates (AND/OR/NOT) are out of
// scope for this initial implementation.
struct ParsedPredicate {
    enum class Op { EQ, NEQ, LT, LTE, GT, GTE };
    enum class LiteralKind { INT, FLOAT, STRING };

    std::string column;
    Op op;
    LiteralKind kind;
    int64_t intVal = 0;
    double floatVal = 0.0;
    std::string stringVal;
};

inline std::string trimAscii(const std::string &t) {
    const auto b = t.find_first_not_of(" \t");
    if (b == std::string::npos)
        return "";
    const auto e = t.find_last_not_of(" \t");
    return t.substr(b, e - b + 1);
}

// Parse a predicate string like `age > 60`, `dept = 'eng'`, `salary <= 72.25`.
// Throws with the offending fragment quoted on any failure.
inline ParsedPredicate parsePredicate(const std::string &s) {
    const std::string trimmed = trimAscii(s);
    if (trimmed.empty())
        throw std::runtime_error("ORC reader: predicate option is empty");

    size_t i = 0;
    if (!(std::isalpha(static_cast<unsigned char>(trimmed[i])) || trimmed[i] == '_'))
        throw std::runtime_error("ORC reader: predicate must start with a column name, got '" + trimmed + "'");
    const size_t colStart = i;
    while (i < trimmed.size() &&
           (std::isalnum(static_cast<unsigned char>(trimmed[i])) || trimmed[i] == '_'))
        ++i;
    ParsedPredicate p;
    p.column = trimmed.substr(colStart, i - colStart);

    while (i < trimmed.size() && (trimmed[i] == ' ' || trimmed[i] == '\t'))
        ++i;
    if (i >= trimmed.size())
        throw std::runtime_error("ORC reader: predicate is missing an operator after column '" + p.column + "'");

    auto matchOp = [&](const char *lit, ParsedPredicate::Op v) -> bool {
        const size_t n = std::strlen(lit);
        if (trimmed.compare(i, n, lit) == 0) {
            p.op = v;
            i += n;
            return true;
        }
        return false;
    };
    if (!matchOp(">=", ParsedPredicate::Op::GTE) && !matchOp("<=", ParsedPredicate::Op::LTE) &&
        !matchOp("!=", ParsedPredicate::Op::NEQ) && !matchOp("=", ParsedPredicate::Op::EQ) &&
        !matchOp(">", ParsedPredicate::Op::GT) && !matchOp("<", ParsedPredicate::Op::LT))
        throw std::runtime_error("ORC reader: predicate has an unknown operator near '" + trimmed.substr(i) + "'");

    while (i < trimmed.size() && (trimmed[i] == ' ' || trimmed[i] == '\t'))
        ++i;
    if (i >= trimmed.size())
        throw std::runtime_error("ORC reader: predicate is missing a literal after the operator");

    if (trimmed[i] == '\'') {
        const size_t litStart = i + 1;
        const size_t litEnd = trimmed.find('\'', litStart);
        if (litEnd == std::string::npos)
            throw std::runtime_error("ORC reader: predicate string literal missing closing quote near '" +
                                     trimmed.substr(i) + "'");
        p.kind = ParsedPredicate::LiteralKind::STRING;
        p.stringVal = trimmed.substr(litStart, litEnd - litStart);
        i = litEnd + 1;
    } else {
        const std::string litStr = trimAscii(trimmed.substr(i));
        i = trimmed.size();
        if (litStr.empty())
            throw std::runtime_error("ORC reader: predicate literal is missing");
        if (litStr.find('.') != std::string::npos) {
            try {
                p.kind = ParsedPredicate::LiteralKind::FLOAT;
                p.floatVal = std::stod(litStr);
            } catch (const std::exception &) {
                throw std::runtime_error("ORC reader: predicate has an invalid float literal '" + litStr + "'");
            }
        } else {
            try {
                p.kind = ParsedPredicate::LiteralKind::INT;
                p.intVal = std::stoll(litStr);
            } catch (const std::exception &) {
                throw std::runtime_error("ORC reader: predicate has an invalid integer literal '" + litStr + "'");
            }
        }
    }

    while (i < trimmed.size() && (trimmed[i] == ' ' || trimmed[i] == '\t'))
        ++i;
    if (i != trimmed.size())
        throw std::runtime_error("ORC reader: predicate has unexpected trailing text near '" +
                                 trimmed.substr(i) + "'");

    return p;
}

// Verify the predicate's column is present in the output and the literal
// kind matches the column type. Returns the OUTPUT-side column index.
inline uint64_t resolveAndValidatePredicate(const ParsedPredicate &p, const FileMetaData &fmd) {
    for (uint64_t c = 0; c < fmd.numCols; ++c) {
        const std::string &label = (c < fmd.labels.size()) ? fmd.labels[c] : std::string();
        if (label != p.column)
            continue;
        const ValueTypeCode vtc = fmd.schema[c];
        const bool okInt = (vtc == ValueTypeCode::SI64 && p.kind == ParsedPredicate::LiteralKind::INT);
        const bool okFloat = (vtc == ValueTypeCode::F64 && p.kind == ParsedPredicate::LiteralKind::FLOAT);
        const bool okStr = (vtc == ValueTypeCode::STR && p.kind == ParsedPredicate::LiteralKind::STRING);
        if (!(okInt || okFloat || okStr))
            throw std::runtime_error("ORC reader: predicate literal type does not match column '" + p.column + "'");
        return c;
    }
    throw std::runtime_error("ORC reader: predicate column '" + p.column + "' is not in the output schema");
}

// Build a liborc SearchArgument from a parsed predicate.
inline std::unique_ptr<orc::SearchArgument> buildSearchArgument(const ParsedPredicate &p) {
    auto builder = orc::SearchArgumentFactory::newBuilder();
    orc::PredicateDataType pdt = orc::PredicateDataType::LONG;
    orc::Literal lit(int64_t{0});
    switch (p.kind) {
    case ParsedPredicate::LiteralKind::INT:
        pdt = orc::PredicateDataType::LONG;
        lit = orc::Literal(p.intVal);
        break;
    case ParsedPredicate::LiteralKind::FLOAT:
        pdt = orc::PredicateDataType::FLOAT;
        lit = orc::Literal(p.floatVal);
        break;
    case ParsedPredicate::LiteralKind::STRING:
        pdt = orc::PredicateDataType::STRING;
        lit = orc::Literal(p.stringVal.c_str(), p.stringVal.size());
        break;
    }

    builder->startAnd();
    switch (p.op) {
    case ParsedPredicate::Op::EQ:
        builder->equals(p.column, pdt, lit);
        break;
    case ParsedPredicate::Op::NEQ:
        builder->startNot().equals(p.column, pdt, lit).end();
        break;
    case ParsedPredicate::Op::LT:
        builder->lessThan(p.column, pdt, lit);
        break;
    case ParsedPredicate::Op::LTE:
        builder->lessThanEquals(p.column, pdt, lit);
        break;
    case ParsedPredicate::Op::GT:
        builder->startNot().lessThanEquals(p.column, pdt, lit).end();
        break;
    case ParsedPredicate::Op::GTE:
        builder->startNot().lessThan(p.column, pdt, lit).end();
        break;
    }
    builder->end();
    return builder->build();
}

// Per-row correctness filter. `col` is the decoded batch column for the
// predicate's column; `rowIdx` is the row within the batch.
inline bool matchesRow(const ParsedPredicate &p, const orc::ColumnVectorBatch *col, uint64_t rowIdx) {
    switch (p.kind) {
    case ParsedPredicate::LiteralKind::INT: {
        const auto *lng = dynamic_cast<const orc::LongVectorBatch *>(col);
        if (!lng)
            throw std::runtime_error("ORC reader: predicate on '" + p.column +
                                     "' - decoded batch is not a LongVectorBatch");
        const int64_t v = lng->data[rowIdx];
        switch (p.op) {
        case ParsedPredicate::Op::EQ:  return v == p.intVal;
        case ParsedPredicate::Op::NEQ: return v != p.intVal;
        case ParsedPredicate::Op::LT:  return v <  p.intVal;
        case ParsedPredicate::Op::LTE: return v <= p.intVal;
        case ParsedPredicate::Op::GT:  return v >  p.intVal;
        case ParsedPredicate::Op::GTE: return v >= p.intVal;
        }
    } break;
    case ParsedPredicate::LiteralKind::FLOAT: {
        const auto *dbl = dynamic_cast<const orc::DoubleVectorBatch *>(col);
        if (!dbl)
            throw std::runtime_error("ORC reader: predicate on '" + p.column +
                                     "' - decoded batch is not a DoubleVectorBatch");
        const double v = dbl->data[rowIdx];
        switch (p.op) {
        case ParsedPredicate::Op::EQ:  return v == p.floatVal;
        case ParsedPredicate::Op::NEQ: return v != p.floatVal;
        case ParsedPredicate::Op::LT:  return v <  p.floatVal;
        case ParsedPredicate::Op::LTE: return v <= p.floatVal;
        case ParsedPredicate::Op::GT:  return v >  p.floatVal;
        case ParsedPredicate::Op::GTE: return v >= p.floatVal;
        }
    } break;
    case ParsedPredicate::LiteralKind::STRING: {
        const auto *sc = dynamic_cast<const orc::StringVectorBatch *>(col);
        if (!sc)
            throw std::runtime_error("ORC reader: predicate on '" + p.column +
                                     "' - decoded batch is not a StringVectorBatch");
        const size_t vlen = static_cast<size_t>(sc->length[rowIdx]);
        const char *vptr = sc->data[rowIdx];
        int cmp;
        if (vlen == p.stringVal.size() && std::memcmp(vptr, p.stringVal.data(), vlen) == 0) {
            cmp = 0;
        } else {
            const size_t minLen = std::min(vlen, p.stringVal.size());
            cmp = std::memcmp(vptr, p.stringVal.data(), minLen);
            if (cmp == 0)
                cmp = (vlen < p.stringVal.size()) ? -1 : 1;
        }
        switch (p.op) {
        case ParsedPredicate::Op::EQ:  return cmp == 0;
        case ParsedPredicate::Op::NEQ: return cmp != 0;
        case ParsedPredicate::Op::LT:  return cmp <  0;
        case ParsedPredicate::Op::LTE: return cmp <= 0;
        case ParsedPredicate::Op::GT:  return cmp >  0;
        case ParsedPredicate::Op::GTE: return cmp >= 0;
        }
    } break;
    }
    throw std::runtime_error("ORC reader: unreachable in matchesRow");
}

inline std::unique_ptr<orc::RowReader> openRowReader(orc::Reader &reader, const std::vector<std::string> &names) {
    orc::RowReaderOptions rowOpts;
    if (!names.empty()) {
        std::list<std::string> nameList(names.begin(), names.end());
        rowOpts.include(nameList);
    }
    return reader.createRowReader(rowOpts);
}

// Validate top-level shape (rows, cols, struct-rooted schema). When
// projection is active, the file's total column count is expected to be
// >= fmd.numCols and is not checked here.
inline void validateShape(const orc::Reader &reader, const FileMetaData &fmd, bool projected) {
    if (reader.getNumberOfRows() != fmd.numRows)
        throw std::runtime_error("ORC reader: row count mismatch - meta says " + std::to_string(fmd.numRows) +
                                 ", file has " + std::to_string(reader.getNumberOfRows()));
    const orc::Type &root = reader.getType();
    if (root.getKind() != orc::STRUCT)
        throw std::runtime_error("ORC reader: expected top-level struct in ORC file");
    if (!projected && root.getSubtypeCount() != fmd.numCols)
        throw std::runtime_error("ORC reader: column count mismatch - meta says " + std::to_string(fmd.numCols) +
                                 ", file has " + std::to_string(root.getSubtypeCount()));
}

// Split "a, b,c" into ["a", "b", "c"]. Trim ASCII whitespace around each
// entry, throw on empty entries, throw on duplicates.
inline std::vector<std::string> parseColumnList(const std::string &s) {
    auto trim = [](std::string t) -> std::string {
        const auto b = t.find_first_not_of(" \t");
        if (b == std::string::npos)
            return "";
        const auto e = t.find_last_not_of(" \t");
        return t.substr(b, e - b + 1);
    };

    if (trim(s).empty())
        throw std::runtime_error("ORC reader: columns option must be a non-empty comma-separated list");

    std::vector<std::string> result;
    size_t start = 0;
    while (true) {
        const size_t comma = s.find(',', start);
        const size_t end = (comma == std::string::npos) ? s.size() : comma;
        std::string tok = trim(s.substr(start, end - start));
        if (tok.empty())
            throw std::runtime_error("ORC reader: columns option has an empty entry");
        for (const auto &existing : result)
            if (existing == tok)
                throw std::runtime_error("ORC reader: columns option has duplicate entry '" + tok + "'");
        result.push_back(std::move(tok));
        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }
    return result;
}

// Given the file's root struct type and a list of requested column names,
// return the mapping from output index to file column index. Throws with
// a human-readable list of available names if any requested name is
// missing.
inline std::vector<uint64_t> resolveProjection(const orc::Type &root, const std::vector<std::string> &names) {
    std::vector<uint64_t> result;
    result.reserve(names.size());
    for (const auto &name : names) {
        bool found = false;
        for (uint64_t i = 0; i < root.getSubtypeCount(); ++i) {
            if (root.getFieldName(i) == name) {
                result.push_back(i);
                found = true;
                break;
            }
        }
        if (!found) {
            std::string avail;
            for (uint64_t i = 0; i < root.getSubtypeCount(); ++i) {
                if (i > 0)
                    avail += ", ";
                avail += root.getFieldName(i);
            }
            throw std::runtime_error("ORC reader: column '" + name + "' not found in file (available: " + avail + ")");
        }
    }
    return result;
}

// Throw if the column at index `c` does not have the expected ORC TypeKind.
inline void expectColumnKind(const orc::Type &root, uint64_t c, orc::TypeKind expected, const char *expectedLabel) {
    orc::TypeKind got = root.getSubtype(c)->getKind();
    if (got != expected)
        throw std::runtime_error(std::string("ORC reader: column ") + std::to_string(c) + " type mismatch - expected " +
                                 expectedLabel + ", got " + root.getSubtype(c)->toString());
}

// Reject a batch column that contains null entries.
inline void rejectNulls(const orc::ColumnVectorBatch *col, uint64_t cIdx) {
    if (col->hasNulls)
        throw std::runtime_error("ORC reader: null values not supported (yet) - column " + std::to_string(cIdx));
}

} // namespace daphne_orc_detail

// ----------------------------------------------------------------------------
// readOrc - entry point matching PR #993 reader signature.
// ----------------------------------------------------------------------------

inline void readOrc(void *res, const FileMetaData &fmd, const char *filename,
                    const std::map<std::string, std::string> &options, DaphneContext *ctx) {
    (void)ctx;

    // Parse `columns` option if present. Empty vector means no projection.
    std::vector<std::string> projNames;
    if (auto it = options.find("columns"); it != options.end())
        projNames = daphne_orc_detail::parseColumnList(it->second);
    const bool projected = !projNames.empty();

    // Parse `predicate` option if present.
    daphne_orc_detail::ParsedPredicate parsedPred;
    bool hasPredicate = false;
    uint64_t predOutColIdx = 0;
    if (auto it = options.find("predicate"); it != options.end()) {
        parsedPred = daphne_orc_detail::parsePredicate(it->second);
        predOutColIdx = daphne_orc_detail::resolveAndValidatePredicate(parsedPred, fmd);
        hasPredicate = true;
    }

    auto reader = daphne_orc_detail::openOrcReader(filename);
    const orc::Type &root = reader->getType();
    daphne_orc_detail::validateShape(*reader, fmd, projected);

    // Resolve output-column -> file-column index mapping.
    std::vector<uint64_t> fileColIdx;
    if (projected) {
        fileColIdx = daphne_orc_detail::resolveProjection(root, projNames);
        if (fileColIdx.size() != fmd.numCols)
            throw std::runtime_error("ORC reader: columns option requests " + std::to_string(fileColIdx.size()) +
                                     " columns but meta says numCols=" + std::to_string(fmd.numCols));
    } else {
        fileColIdx.reserve(fmd.numCols);
        for (uint64_t i = 0; i < fmd.numCols; ++i)
            fileColIdx.push_back(i);
    }

    // liborc's include() compacts the batch's fields[] into file-schema
    // order, not user-requested order. batchColIdx maps each output
    // column c to its slot within the compacted batch.
    std::vector<uint64_t> batchColIdx;
    batchColIdx.reserve(fmd.numCols);
    if (projected) {
        std::vector<uint64_t> sorted = fileColIdx;
        std::sort(sorted.begin(), sorted.end());
        for (uint64_t c = 0; c < fmd.numCols; ++c) {
            const auto it = std::find(sorted.begin(), sorted.end(), fileColIdx[c]);
            batchColIdx.push_back(static_cast<uint64_t>(std::distance(sorted.begin(), it)));
        }
    } else {
        for (uint64_t i = 0; i < fmd.numCols; ++i)
            batchColIdx.push_back(i);
    }

    orc::RowReaderOptions rowOpts;
    if (!projNames.empty()) {
        std::list<std::string> nameList(projNames.begin(), projNames.end());
        rowOpts.include(nameList);
    }
    if (hasPredicate)
        rowOpts.searchArgument(daphne_orc_detail::buildSearchArgument(parsedPred));
    auto rowReader = reader->createRowReader(rowOpts);
    constexpr uint64_t kBatchSize = 1024;
    auto batch = rowReader->createRowBatch(kBatchSize);

    // ----- Predicate branch (Option C: reader allocates final result) -----
    if (hasPredicate) {
        if (fmd.isSingleValueType) {
            const ValueTypeCode vt = fmd.schema.empty() ? ValueTypeCode::F64 : fmd.schema[0];
            if (vt != ValueTypeCode::F64 && vt != ValueTypeCode::SI64)
                throw std::runtime_error("ORC reader: value type not supported by ORC reader (yet) - "
                                         "only F64 and SI64 are implemented");
            for (uint64_t c = 0; c < fmd.numCols; ++c)
                daphne_orc_detail::expectColumnKind(
                    root, fileColIdx[c],
                    (vt == ValueTypeCode::F64) ? orc::DOUBLE : orc::LONG,
                    (vt == ValueTypeCode::F64) ? "F64" : "SI64");

            if (vt == ValueTypeCode::F64) {
                auto **out = reinterpret_cast<DenseMatrix<double> **>(res);
                if (*out != nullptr)
                    throw std::runtime_error("ORC reader: predicate requires *res == nullptr; caller pre-allocated");
                std::vector<double> vals;
                vals.reserve(static_cast<size_t>(fmd.numRows) * fmd.numCols);
                uint64_t matched = 0;
                while (rowReader->next(*batch)) {
                    auto *sb = dynamic_cast<orc::StructVectorBatch *>(batch.get());
                    if (!sb)
                        throw std::runtime_error("ORC reader: top-level batch is not a struct");
                    for (uint64_t c = 0; c < fmd.numCols; ++c)
                        daphne_orc_detail::rejectNulls(sb->fields[batchColIdx[c]], c);
                    for (uint64_t i = 0; i < batch->numElements; ++i) {
                        if (!daphne_orc_detail::matchesRow(parsedPred, sb->fields[batchColIdx[predOutColIdx]], i))
                            continue;
                        for (uint64_t c = 0; c < fmd.numCols; ++c) {
                            const auto *dbl =
                                dynamic_cast<orc::DoubleVectorBatch *>(sb->fields[batchColIdx[c]]);
                            vals.push_back(dbl->data[i]);
                        }
                        ++matched;
                    }
                }
                *out = DataObjectFactory::create<DenseMatrix<double>>(matched, fmd.numCols, false);
                std::copy(vals.begin(), vals.end(), (*out)->getValues());
            } else { // SI64
                auto **out = reinterpret_cast<DenseMatrix<int64_t> **>(res);
                if (*out != nullptr)
                    throw std::runtime_error("ORC reader: predicate requires *res == nullptr; caller pre-allocated");
                std::vector<int64_t> vals;
                vals.reserve(static_cast<size_t>(fmd.numRows) * fmd.numCols);
                uint64_t matched = 0;
                while (rowReader->next(*batch)) {
                    auto *sb = dynamic_cast<orc::StructVectorBatch *>(batch.get());
                    if (!sb)
                        throw std::runtime_error("ORC reader: top-level batch is not a struct");
                    for (uint64_t c = 0; c < fmd.numCols; ++c)
                        daphne_orc_detail::rejectNulls(sb->fields[batchColIdx[c]], c);
                    for (uint64_t i = 0; i < batch->numElements; ++i) {
                        if (!daphne_orc_detail::matchesRow(parsedPred, sb->fields[batchColIdx[predOutColIdx]], i))
                            continue;
                        for (uint64_t c = 0; c < fmd.numCols; ++c) {
                            const auto *lng =
                                dynamic_cast<orc::LongVectorBatch *>(sb->fields[batchColIdx[c]]);
                            vals.push_back(lng->data[i]);
                        }
                        ++matched;
                    }
                }
                *out = DataObjectFactory::create<DenseMatrix<int64_t>>(matched, fmd.numCols, false);
                std::copy(vals.begin(), vals.end(), (*out)->getValues());
            }
            return;
        }

        // Frame path with predicate: one growth buffer per column, typed by schema.
        auto **outFrame = reinterpret_cast<Frame **>(res);
        if (*outFrame != nullptr)
            throw std::runtime_error("ORC reader: predicate requires *res == nullptr; caller pre-allocated");

        for (uint64_t c = 0; c < fmd.numCols; ++c) {
            const ValueTypeCode vtc = fmd.schema[c];
            if (vtc == ValueTypeCode::F64)
                daphne_orc_detail::expectColumnKind(root, fileColIdx[c], orc::DOUBLE, "F64");
            else if (vtc == ValueTypeCode::SI64)
                daphne_orc_detail::expectColumnKind(root, fileColIdx[c], orc::LONG, "SI64");
            else if (vtc == ValueTypeCode::STR)
                daphne_orc_detail::expectColumnKind(root, fileColIdx[c], orc::STRING, "STR");
            else
                throw std::runtime_error("ORC reader: value type not supported by ORC reader (yet) - "
                                         "only F64, SI64, and STR are implemented (column " +
                                         std::to_string(c) + ")");
        }

        std::vector<std::vector<double>> f64_bufs(fmd.numCols);
        std::vector<std::vector<int64_t>> si64_bufs(fmd.numCols);
        std::vector<std::vector<std::string>> str_bufs(fmd.numCols);
        for (uint64_t c = 0; c < fmd.numCols; ++c) {
            if (fmd.schema[c] == ValueTypeCode::F64)
                f64_bufs[c].reserve(fmd.numRows);
            else if (fmd.schema[c] == ValueTypeCode::SI64)
                si64_bufs[c].reserve(fmd.numRows);
            else
                str_bufs[c].reserve(fmd.numRows);
        }

        uint64_t matched = 0;
        while (rowReader->next(*batch)) {
            auto *sb = dynamic_cast<orc::StructVectorBatch *>(batch.get());
            if (!sb)
                throw std::runtime_error("ORC reader: top-level batch is not a struct");
            for (uint64_t c = 0; c < fmd.numCols; ++c)
                daphne_orc_detail::rejectNulls(sb->fields[batchColIdx[c]], c);
            for (uint64_t i = 0; i < batch->numElements; ++i) {
                if (!daphne_orc_detail::matchesRow(parsedPred, sb->fields[batchColIdx[predOutColIdx]], i))
                    continue;
                for (uint64_t c = 0; c < fmd.numCols; ++c) {
                    auto *col = sb->fields[batchColIdx[c]];
                    const ValueTypeCode vtc = fmd.schema[c];
                    if (vtc == ValueTypeCode::F64) {
                        auto *dbl = dynamic_cast<orc::DoubleVectorBatch *>(col);
                        f64_bufs[c].push_back(dbl->data[i]);
                    } else if (vtc == ValueTypeCode::SI64) {
                        auto *lng = dynamic_cast<orc::LongVectorBatch *>(col);
                        si64_bufs[c].push_back(lng->data[i]);
                    } else { // STR
                        auto *sc = dynamic_cast<orc::StringVectorBatch *>(col);
                        str_bufs[c].push_back(sc->length[i] > 0
                                                  ? std::string(sc->data[i], static_cast<size_t>(sc->length[i]))
                                                  : std::string());
                    }
                }
                ++matched;
            }
        }

        std::vector<ValueTypeCode> outSchema(fmd.schema.begin(), fmd.schema.end());
        std::vector<std::string> outLabels(fmd.labels.begin(), fmd.labels.end());
        *outFrame = DataObjectFactory::create<Frame>(matched, fmd.numCols, outSchema.data(),
                                                    outLabels.empty() ? nullptr : outLabels.data(), false);
        for (uint64_t c = 0; c < fmd.numCols; ++c) {
            const ValueTypeCode vtc = fmd.schema[c];
            if (vtc == ValueTypeCode::F64) {
                double *dst = (*outFrame)->getColumn<double>(c)->getValues();
                std::copy(f64_bufs[c].begin(), f64_bufs[c].end(), dst);
            } else if (vtc == ValueTypeCode::SI64) {
                int64_t *dst = (*outFrame)->getColumn<int64_t>(c)->getValues();
                std::copy(si64_bufs[c].begin(), si64_bufs[c].end(), dst);
            } else {
                std::string *dst = (*outFrame)->getColumn<std::string>(c)->getValues();
                std::copy(str_bufs[c].begin(), str_bufs[c].end(), dst);
            }
        }
        return;
    }

    if (fmd.isSingleValueType) {
        const ValueTypeCode vt = fmd.schema.empty() ? ValueTypeCode::F64 : fmd.schema[0];

        // ----- DenseMatrix<double> -----
        if (vt == ValueTypeCode::F64) {
            for (uint64_t c = 0; c < fmd.numCols; ++c)
                daphne_orc_detail::expectColumnKind(root, fileColIdx[c], orc::DOUBLE, "F64");

            auto **out = reinterpret_cast<DenseMatrix<double> **>(res);
            DenseMatrix<double> *m = *out;
            double *vals = m->getValues();
            uint64_t rowOffset = 0;
            while (rowReader->next(*batch)) {
                auto *sb = dynamic_cast<orc::StructVectorBatch *>(batch.get());
                if (!sb)
                    throw std::runtime_error("ORC reader: top-level batch is not a struct");
                for (uint64_t c = 0; c < fmd.numCols; ++c) {
                    auto *col = sb->fields[batchColIdx[c]];
                    daphne_orc_detail::rejectNulls(col, c);
                    auto *dbl = dynamic_cast<orc::DoubleVectorBatch *>(col);
                    if (!dbl)
                        throw std::runtime_error("ORC reader: column " + std::to_string(c) +
                                                 " type mismatch - expected F64 (DoubleVectorBatch)");
                    for (uint64_t i = 0; i < batch->numElements; ++i)
                        vals[(rowOffset + i) * fmd.numCols + c] = dbl->data[i];
                }
                rowOffset += batch->numElements;
            }
            if (rowOffset != fmd.numRows)
                throw std::runtime_error("ORC reader: read " + std::to_string(rowOffset) + " rows; expected " +
                                         std::to_string(fmd.numRows));
            return;
        }

        // ----- DenseMatrix<int64_t> -----
        if (vt == ValueTypeCode::SI64) {
            for (uint64_t c = 0; c < fmd.numCols; ++c)
                daphne_orc_detail::expectColumnKind(root, fileColIdx[c], orc::LONG, "SI64");

            auto **out = reinterpret_cast<DenseMatrix<int64_t> **>(res);
            DenseMatrix<int64_t> *m = *out;
            int64_t *vals = m->getValues();
            uint64_t rowOffset = 0;
            while (rowReader->next(*batch)) {
                auto *sb = dynamic_cast<orc::StructVectorBatch *>(batch.get());
                if (!sb)
                    throw std::runtime_error("ORC reader: top-level batch is not a struct");
                for (uint64_t c = 0; c < fmd.numCols; ++c) {
                    auto *col = sb->fields[batchColIdx[c]];
                    daphne_orc_detail::rejectNulls(col, c);
                    auto *lng = dynamic_cast<orc::LongVectorBatch *>(col);
                    if (!lng)
                        throw std::runtime_error("ORC reader: column " + std::to_string(c) +
                                                 " type mismatch - expected SI64 (LongVectorBatch)");
                    for (uint64_t i = 0; i < batch->numElements; ++i)
                        vals[(rowOffset + i) * fmd.numCols + c] = lng->data[i];
                }
                rowOffset += batch->numElements;
            }
            if (rowOffset != fmd.numRows)
                throw std::runtime_error("ORC reader: read " + std::to_string(rowOffset) + " rows; expected " +
                                         std::to_string(fmd.numRows));
            return;
        }

        throw std::runtime_error("ORC reader: value type not supported by ORC reader (yet) - only F64 and SI64 are "
                                 "implemented");
    }

    // ----- Frame: per-column copy, type-dispatched per column -----
    {
        auto **outFrame = reinterpret_cast<Frame **>(res);
        Frame *f = *outFrame;

        // Pre-validate each output column's expected type against the file's TypeKind.
        for (uint64_t c = 0; c < fmd.numCols; ++c) {
            const ValueTypeCode vtc = fmd.schema[c];
            if (vtc == ValueTypeCode::F64)
                daphne_orc_detail::expectColumnKind(root, fileColIdx[c], orc::DOUBLE, "F64");
            else if (vtc == ValueTypeCode::SI64)
                daphne_orc_detail::expectColumnKind(root, fileColIdx[c], orc::LONG, "SI64");
            else if (vtc == ValueTypeCode::STR)
                daphne_orc_detail::expectColumnKind(root, fileColIdx[c], orc::STRING, "STR");
            else
                throw std::runtime_error("ORC reader: value type not supported by ORC reader (yet) - only F64, SI64, "
                                         "and STR are implemented (column " +
                                         std::to_string(c) + ")");
        }

        uint64_t rowOffset = 0;
        while (rowReader->next(*batch)) {
            auto *sb = dynamic_cast<orc::StructVectorBatch *>(batch.get());
            if (!sb)
                throw std::runtime_error("ORC reader: top-level batch is not a struct");
            for (uint64_t c = 0; c < fmd.numCols; ++c) {
                auto *col = sb->fields[batchColIdx[c]];
                daphne_orc_detail::rejectNulls(col, c);
                const ValueTypeCode vtc = fmd.schema[c];
                if (vtc == ValueTypeCode::F64) {
                    auto *dbl = dynamic_cast<orc::DoubleVectorBatch *>(col);
                    if (!dbl)
                        throw std::runtime_error("ORC reader: column " + std::to_string(c) +
                                                 " type mismatch - expected F64");
                    double *buf = f->getColumn<double>(c)->getValues();
                    for (uint64_t i = 0; i < batch->numElements; ++i)
                        buf[rowOffset + i] = dbl->data[i];
                } else if (vtc == ValueTypeCode::SI64) {
                    auto *lng = dynamic_cast<orc::LongVectorBatch *>(col);
                    if (!lng)
                        throw std::runtime_error("ORC reader: column " + std::to_string(c) +
                                                 " type mismatch - expected SI64");
                    int64_t *buf = f->getColumn<int64_t>(c)->getValues();
                    for (uint64_t i = 0; i < batch->numElements; ++i)
                        buf[rowOffset + i] = lng->data[i];
                } else { // STR (others rejected in pre-validation)
                    auto *sc = dynamic_cast<orc::StringVectorBatch *>(col);
                    if (!sc)
                        throw std::runtime_error("ORC reader: column " + std::to_string(c) +
                                                 " type mismatch - expected STR (StringVectorBatch)");
                    std::string *buf = f->getColumn<std::string>(c)->getValues();
                    for (uint64_t i = 0; i < batch->numElements; ++i)
                        buf[rowOffset + i] = (sc->length[i] > 0)
                                                 ? std::string(sc->data[i], static_cast<size_t>(sc->length[i]))
                                                 : std::string();
                }
            }
            rowOffset += batch->numElements;
        }
        if (rowOffset != fmd.numRows)
            throw std::runtime_error("ORC reader: read " + std::to_string(rowOffset) + " rows; expected " +
                                     std::to_string(fmd.numRows));
    }
}
