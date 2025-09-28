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

#include "runtime/local/io/FileIORegistry.h"
#include <cstddef>
#include <iostream>
#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/io/ReadDaphne.h>
#include <runtime/local/io/ReadMM.h>
#include <runtime/local/io/ReadParquet.h>
#if USE_HDFS
#include <runtime/local/io/HDFS/ReadHDFS.h>
#endif

#include <filesystem>
#include <string>
#include <stdexcept>

static Frame *dummyFrame = DataObjectFactory::create<Frame>(0, 0, nullptr, nullptr, false);


// ****************************************************************************
// Helper: Merge a Frame* of column-label → single-row-value into IOOptions
// ****************************************************************************
static IOOptions mergeOptionsFromFrame(const std::string &ext,
                                       IODataType         dt,
                                       const std::string &engine, // NEW
                                       Frame             *optsFrame,
                                       DCTX(ctx))
{
    auto& reg = ctx ? ctx->config.registry : FileIORegistry::instance();

    // Ask the registry for defaults for this (ext, dt, engine).
    // If engine == "", registry should pick highest-priority impl.
    const IOOptions &defaults = reg.getOptions(ext, dt, engine);

    IOOptions merged = defaults;

    //reg.dumpReaders();

    if(optsFrame != nullptr && optsFrame->getLabels()[0] != "dummy") {
        const size_t nRows = optsFrame->getNumRows();
        const size_t nCols = optsFrame->getNumCols();
        if(nRows == 0) return merged;

        const std::string* labels = optsFrame->getLabels();

        for(size_t colIdx = 0; colIdx < nCols; ++colIdx) {
            const std::string &key = labels[colIdx];

            // Ignore non-plugin selection knobs if user sent them in the frame.
            if(key == "engine" || key == "priority")
                continue;

            std::string value;
            if(auto* strCol = dynamic_cast<DenseMatrix<std::string>*>(optsFrame->getColumn<std::string>(colIdx))) {
                value = strCol->get(0, 0);
            }
            else if(auto* boolCol = dynamic_cast<DenseMatrix<bool>*>(optsFrame->getColumn<bool>(colIdx))) {
                value = boolCol->get(0, 0) ? "true" : "false";
            }
            else if(auto* intCol = dynamic_cast<DenseMatrix<int64_t>*>(optsFrame->getColumn<int64_t>(colIdx))) {
                value = std::to_string(intCol->get(0, 0));
            }
            else if(auto* floatCol = dynamic_cast<DenseMatrix<double>*>(optsFrame->getColumn<double>(colIdx))) {
                value = std::to_string(floatCol->get(0, 0));
            }
            else {
                throw std::runtime_error("Unsupported column type for option: " + key);
            }

            // Only override known plugin options
            auto itKnown = merged.extra.find(key);
            if(itKnown == merged.extra.end()) {
                // silently ignore unknown keys instead of throwing if you prefer:
                // continue;
                throw std::runtime_error("Unknown option '" + key + "'");
            }

            merged.extra[key] = value;
        }
    }

    return merged;
}


// Extract "engine" (and ignore "priority") from the options Frame if present.
// Returns "" if not provided (so registry picks highest-priority default).
static std::string extractEngineFromFrame(Frame *optsFrame) {
    
    if(!optsFrame) return "";
    if(optsFrame->getNumRows() == 0) return "";
    const auto *labels = optsFrame->getLabels();
    const size_t nCols = optsFrame->getNumCols();

    for(size_t c = 0; c < nCols; ++c) {
        if(labels[c] == "engine") {
            if(auto* strCol = dynamic_cast<DenseMatrix<std::string>*>(optsFrame->getColumn<std::string>(c)))
                return strCol->get(0, 0);
            // allow non-string columns too (we’ll stringify)
            if(auto* boolCol = dynamic_cast<DenseMatrix<bool>*>(optsFrame->getColumn<bool>(c)))
                return boolCol->get(0, 0) ? "true" : "false";
            if(auto* intCol = dynamic_cast<DenseMatrix<int64_t>*>(optsFrame->getColumn<int64_t>(c)))
                return std::to_string(intCol->get(0, 0));
            if(auto* floatCol = dynamic_cast<DenseMatrix<double>*>(optsFrame->getColumn<double>(c)))
                return std::to_string(floatCol->get(0, 0));
        }
    }
    return "";
}


// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct Read {
    static void apply(DTRes *&res, const char *filename, Frame* opts, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes> void read(DTRes *&res, const char *filename,Frame *opts, DCTX(ctx)) {
    Read<DTRes>::apply(res, filename, opts, ctx);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Read<DenseMatrix<VT>> {
    // ------------------------------------------------------------------------
    // Overload: DenseMatrix with an options‐Frame
    // ------------------------------------------------------------------------
    static void apply(DenseMatrix<VT> *&res,
                      const char       *filename,
                      Frame            *optsFrame,
                      DCTX(ctx))
    {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());
        IODataType typeHash = DENSEMATRIX;
        try {
            auto& registry = ctx ? ctx->config.registry : FileIORegistry::instance();

            //registry.dumpReaders();
            //registry.dumpWriters();

            // NEW: get the engine (may be "")
            std::string engine = extractEngineFromFrame(optsFrame);

            // NEW: select reader with engine hint
            auto reader = registry.getReader(ext, typeHash, engine);

            // Merge user overrides using defaults for that engine
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, typeHash, engine, optsFrame, ctx);

            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        }
        catch (const std::out_of_range &e) {
            std::cerr << "no suitable reader found in the registry";
        }
        //std::cout << "d";

        if (ext == ".dbdf") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued DAPHNE binary format files is not supported (yet)");
            else
                readDaphne(res, filename);
        }
#if USE_HDFS
        else if (ext == ".hdfs") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued HDFS files is not supported (yet)");
            else {
                if (res == nullptr)
                    res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
                readHDFS(res, filename, ctx);
            }
        }
#endif
        else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }
}; // end Read<DenseMatrix<VT>>

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Read<CSRMatrix<VT>> {
    // ------------------------------------------------------------------------
    // Overload: CSRMatrix with an options‐Frame
    // ------------------------------------------------------------------------
    static void apply(CSRMatrix<VT> *&res,
                      const char     *filename,
                      Frame          *optsFrame,
                      DCTX(ctx))
    {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());
        IODataType typeHash = CSRMATRIX;
        try {
            auto& registry = ctx ? ctx->config.registry : FileIORegistry::instance();

            // NEW: get the engine (may be "")
            std::string engine = extractEngineFromFrame(optsFrame);

            // NEW: select reader with engine hint
            auto reader = registry.getReader(ext, typeHash, engine);

            // Merge user overrides using defaults for that engine
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, typeHash, engine, optsFrame, ctx);

            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        }
        catch (const std::out_of_range &) {
            // no plugin, fall back to built-in
        }
        //std::cout << "using default";

        if (ext == ".csv") {
            if (fmd.numNonZeros == -1)
                throw std::runtime_error("currently reading of sparse matrices requires a number of "
                                         "non zeros to be defined");

            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(fmd.numRows, fmd.numCols, fmd.numNonZeros, false);

            readCsv(res, filename, fmd.numRows, fmd.numCols, ',', fmd.numNonZeros, true);
        } else if (ext == ".mtx") {
            readMM(res, filename);
        } else if (ext == ".parquet") {
            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(fmd.numRows, fmd.numCols, fmd.numNonZeros, false);
            readParquet(res, filename, fmd.numRows, fmd.numCols, fmd.numNonZeros, false);
        } else if (ext == ".dbdf") {
            readDaphne(res, filename);
        } else {
            throw std::runtime_error("file extension not supported: '" + ext + "'");
        }
    }
}; // end Read<CSRMatrix<VT>>

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct Read<Frame> {
    // ------------------------------------------------------------------------
    // Overload: Frame with an options‐Frame
    // ------------------------------------------------------------------------
    static void apply(Frame *&res,
                      const char *filename,
                      Frame      *optsFrame,
                      DCTX(ctx))
    {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());
        IODataType typeHash = FRAME;
        std::string engine;
        try {
            auto& registry = ctx ? ctx->config.registry : FileIORegistry::instance();

            //registry.dumpReaders();
            //registry.dumpWriters();
            
            // NEW: get the engine (may be "")
            engine = extractEngineFromFrame(optsFrame);

            // NEW: select reader with engine hint
            auto reader = registry.getReader(ext, typeHash, engine);

            // Merge user overrides using defaults for that engine
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, typeHash, engine, optsFrame, ctx);

            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        } catch (const std::out_of_range &) {
            throw std::runtime_error("No suitable reader found in the registry");
        }
        //std::cout << "d";

    }
}; // end Read<Frame>

#endif // SRC_RUNTIME_LOCAL_KERNELS_READ_H