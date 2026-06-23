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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_READ_H
#define SRC_RUNTIME_LOCAL_KERNELS_READ_H

#include "runtime/local/datastructures/ValueTypeCode.h"
#include "runtime/local/datastructures/ValueTypeUtils.h"
#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/FileIORegistry.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/io/ReadDaphne.h>
#include <runtime/local/io/ReadMM.h>
#include <runtime/local/io/ReadParquet.h>
#if USE_HDFS
#include <runtime/local/io/HDFS/ReadHDFS.h>
#endif

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cstddef>

// ****************************************************************************
// Helper: Merge a Frame* of column-label → single-row-value into IOOptions
// ****************************************************************************

static IOOptions mergeOptionsFromFrame(const std::string &ext, PhyDataType dt, const std::string &engine,
                                       const Frame *optsFrame, DCTX(ctx)) {
    auto &reg = ctx ? ctx->config.registry : FileIORegistry::instance();

    // Ask the registry for defaults for this (ext, dt, engine).
    // If engine == "", registry should pick highest-priority impl.
    const IOOptions &defaults = reg.getOptions(ext, dt, engine);

    IOOptions merged = defaults;

    const size_t numRows = optsFrame->getNumRows();
    const size_t numCols = optsFrame->getNumCols();

    if (numRows != 1)
        throw std::runtime_error("the file reader/writer options must be a frame with exactly one row, but found " +
                                 std::to_string(numRows) + " rows");

    if (optsFrame == nullptr || numCols == 0)
        return merged;

    const std::string *labels = optsFrame->getLabels();

    for (size_t c = 0; c < numCols; ++c) {
        const std::string &key = labels[c];

        // Ignore non-plugin selection knobs if user sent them in the frame.
        if (key == "engine" || key == "priority")
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
// Returns "" if not provided (so registry picks highest-priority default).
static std::string extractEngineFromFrame(const Frame *optsFrame) {
    if (!optsFrame)
        return "";

    const size_t numRows = optsFrame->getNumRows();
    const size_t numCols = optsFrame->getNumCols();

    if (numRows != 1)
        throw std::runtime_error("the file reader/writer options must be a frame with exactly one row, but found " +
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

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct Read {
    static void apply(DTRes *&res, const char *filename, const Frame *opts, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes> void read(DTRes *&res, const char *filename, const Frame *opts, DCTX(ctx)) {
    Read<DTRes>::apply(res, filename, opts, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Read<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const char *filename, const Frame *optsFrame, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());
        PhyDataType dt = PhyDataType::DENSEMATRIX;
        try {
            auto &registry = ctx ? ctx->config.registry : FileIORegistry::instance();

            // Get the engine (may be "")
            std::string engine = extractEngineFromFrame(optsFrame);

            // Select reader with engine hint
            auto reader = registry.getReader(ext, dt, engine);

            // Merge user overrides using defaults for that engine
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, dt, engine, optsFrame, ctx);

            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        } catch (const std::out_of_range &e) {
            std::cerr << "no suitable reader found in the registry";
        }
#if USE_HDFS
        if (ext == ".hdfs") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued HDFS files is not supported (yet)");
            else {
                if (res == nullptr)
                    res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
                readHDFS(res, filename, ctx);
                return;
            }
        }
#endif
        throw std::runtime_error("no suitable writer found in the registry");
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Read<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const char *filename, const Frame *optsFrame, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());
        PhyDataType dt = PhyDataType::CSRMATRIX;
        try {
            auto &registry = ctx ? ctx->config.registry : FileIORegistry::instance();

            // Get the engine (may be "")
            std::string engine = extractEngineFromFrame(optsFrame);

            // Select reader with engine hint
            auto reader = registry.getReader(ext, dt, engine);

            // Merge user overrides using defaults for that engine
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, dt, engine, optsFrame, ctx);

            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        } catch (const std::out_of_range &) {
            throw std::runtime_error("no suitable reader found in the registry");
        }
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct Read<Frame> {
    static void apply(Frame *&res, const char *filename, const Frame *optsFrame, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());
        PhyDataType dt = PhyDataType::FRAME;
        try {
            auto &registry = ctx ? ctx->config.registry : FileIORegistry::instance();

            // Get the engine (may be "")
            std::string engine = extractEngineFromFrame(optsFrame);

            // Select reader with engine hint
            auto reader = registry.getReader(ext, dt, engine);

            // Merge user overrides using defaults for that engine
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, dt, engine, optsFrame, ctx);

            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        } catch (const std::out_of_range &) {
            throw std::runtime_error("no suitable reader found in the registry");
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_READ_H