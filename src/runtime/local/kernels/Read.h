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

#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/FileIOCatalog.h>
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
        try {
            FileMetaData fmd = MetaDataParser::readMetaData(filename);
            std::string ext(std::filesystem::path(filename).extension());
            // TODO Support HDFS through a file IO extension and remove this special case.
#if USE_HDFS
            if (ext == ".hdfs") {
                if constexpr (std::is_same<VT, std::string>::value)
                    throw std::runtime_error("reading string-valued HDFS files is not supported (yet)");
                else {
                    if (res == nullptr)
                        res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
                    readHDFS(res, filename, ctx);
                }
            } else
#endif
            {
                PhyDataType dt = PhyDataType::DENSEMATRIX;
                auto &catalog = ctx ? ctx->config.fileioCatalog : FileIOCatalog::instance();

                // Get the engine (may be "")
                std::string engine = extractEngineFromFrame(optsFrame);

                // Select reader with engine hint
                auto reader = catalog.getReader(ext, dt, engine);

                // Merge user overrides using defaults for that engine
                IOOptions mergedOpts = mergeOptionsFromFrame(ext, dt, engine, optsFrame, catalog);

                reader(&res, fmd, filename, mergedOpts, ctx);
            }
        } catch (const std::exception &e) {
            throw std::runtime_error("error while reading file `" + std::string(filename) + "`: " + e.what());
        }
    }
};

// ----------------------------------------------------------------------------
// CSRMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Read<CSRMatrix<VT>> {
    static void apply(CSRMatrix<VT> *&res, const char *filename, const Frame *optsFrame, DCTX(ctx)) {
        try {
            FileMetaData fmd = MetaDataParser::readMetaData(filename);
            std::string ext(std::filesystem::path(filename).extension());
            PhyDataType dt = PhyDataType::CSRMATRIX;
            auto &catalog = ctx ? ctx->config.fileioCatalog : FileIOCatalog::instance();

            // Get the engine (may be "")
            std::string engine = extractEngineFromFrame(optsFrame);

            // Select reader with engine hint
            auto reader = catalog.getReader(ext, dt, engine);

            // Merge user overrides using defaults for that engine
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, dt, engine, optsFrame, catalog);

            reader(&res, fmd, filename, mergedOpts, ctx);
        } catch (const std::exception &e) {
            throw std::runtime_error("error while reading file `" + std::string(filename) + "`: " + e.what());
        }
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct Read<Frame> {
    static void apply(Frame *&res, const char *filename, const Frame *optsFrame, DCTX(ctx)) {
        try {
            FileMetaData fmd = MetaDataParser::readMetaData(filename);
            std::string ext(std::filesystem::path(filename).extension());
            PhyDataType dt = PhyDataType::FRAME;
            auto &catalog = ctx ? ctx->config.fileioCatalog : FileIOCatalog::instance();

            // Get the engine (may be "")
            std::string engine = extractEngineFromFrame(optsFrame);

            // Select reader with engine hint
            auto reader = catalog.getReader(ext, dt, engine);

            // Merge user overrides using defaults for that engine
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, dt, engine, optsFrame, catalog);

            reader(&res, fmd, filename, mergedOpts, ctx);
        } catch (const std::exception &e) {
            throw std::runtime_error("error while reading file `" + std::string(filename) + "`: " + e.what());
        }
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_READ_H