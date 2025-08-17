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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_WRITE_H
#define SRC_RUNTIME_LOCAL_KERNELS_WRITE_H

#include "runtime/local/io/FileIORegistry.h"
#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/io/WriteCsv.h>
#include <runtime/local/io/WriteDaphne.h>
#include <runtime/local/kernels/Read.h>

#if USE_HDFS
#include <runtime/local/io/HDFS/WriteHDFS.h>
#endif

#include <filesystem>
#include <string>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTArg> struct Write {
    static void apply(const DTArg *arg, const char *filename, Frame* opts, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTArg> void write(const DTArg *arg, const char *filename, Frame *opts, DCTX(ctx)) {
    Write<DTArg>::apply(arg, filename, opts, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Write<DenseMatrix<VT>> {
    static void apply(const DenseMatrix<VT> *arg, const char *filename, Frame *optsFrame, DCTX(ctx)) {
        std::string ext(std::filesystem::path(filename).extension());
        try {
            auto& registry = ctx ? ctx->config.registry : FileIORegistry::instance();  
            IODataType typeHash = DENSEMATRIX;
            auto writer = registry.getWriter(ext, typeHash);
            FileMetaData fmd(arg->getNumRows(), arg->getNumCols(), true, ValueTypeUtils::codeFor<VT>);

            MetaDataParser::writeMetaData(filename, fmd);

            // Merge user overrides from optsFrame
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, typeHash, optsFrame,ctx);
            
            writer(arg, fmd, filename, mergedOpts, ctx);
            //std::cout << "using registry\n";
            return;
        }
        catch (const std::out_of_range &e) {
        }

        if (ext == ".csv") {
            File *file = openFileForWrite(filename);
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), true, ValueTypeUtils::codeFor<VT>);
            MetaDataParser::writeMetaData(filename, metaData);
            writeCsv(arg, file);
            closeFile(file);
        } else if (ext == ".dbdf") {
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), true, ValueTypeUtils::codeFor<VT>);
            MetaDataParser::writeMetaData(filename, metaData);
            writeDaphne(arg, filename);
#if USE_HDFS
        } else if (ext == ".hdfs") {
            HDFSMetaData hdfs = {true, filename};
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), true, ValueTypeUtils::codeFor<VT>, -1, hdfs);
            // Get file extension before .hdfs (e.g. file.csv.hdfs)
            std::string nestedExt(
                std::filesystem::path(std::string(filename).substr(0, std::string(filename).size() - ext.size()))
                    .extension());
            MetaDataParser::writeMetaData(filename, metaData);

            // call WriteHDFS
            writeHDFS(arg, filename, ctx);
#endif
        } else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }
};

// ----------------------------------------------------------------------------
// Frame
// ----------------------------------------------------------------------------

template <> struct Write<Frame> {
    static void apply(const Frame *arg, const char *filename, Frame *optsFrame, DCTX(ctx)) {
        std::string ext(std::filesystem::path(filename).extension());

        try {
            auto& registry = ctx ? ctx->config.registry : FileIORegistry::instance();  
            IODataType typeHash = FRAME;
            auto writer = registry.getWriter(ext, typeHash);
            std::vector<ValueTypeCode> vtcs;
            std::vector<std::string> labels;
            for (size_t i = 0; i < arg->getNumCols(); i++) {
                vtcs.push_back(arg->getSchema()[i]);
                labels.push_back(arg->getLabels()[i]);
            }
            FileMetaData fmd(arg->getNumRows(), arg->getNumCols(), false, vtcs, labels);            
            MetaDataParser::writeMetaData(filename, fmd);

            // Merge user overrides from optsFrame
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, typeHash, optsFrame,ctx);

            writer(arg, fmd, filename, mergedOpts, ctx);
            return;
        }
        catch (const std::out_of_range &e) {
        }

        if (ext == ".csv") {
            File *file = openFileForWrite(filename);
            std::vector<ValueTypeCode> vtcs;
            std::vector<std::string> labels;
            for (size_t i = 0; i < arg->getNumCols(); i++) {
                vtcs.push_back(arg->getSchema()[i]);
                labels.push_back(arg->getLabels()[i]);
            }
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), false, vtcs, labels);
            MetaDataParser::writeMetaData(filename, metaData);
            writeCsv(arg, file);
            closeFile(file);
        } else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }
};

// ----------------------------------------------------------------------------
// Matrix
// ----------------------------------------------------------------------------

template <typename VT> struct Write<Matrix<VT>> {
    static void apply(const Matrix<VT> *arg, const char *filename, Frame *optsFrame, DCTX(ctx)) {
        std::string ext(std::filesystem::path(filename).extension());

        if (ext == ".csv") {
            File *file = openFileForWrite(filename);
            FileMetaData metaData(arg->getNumRows(), arg->getNumCols(), true, ValueTypeUtils::codeFor<VT>);
            MetaDataParser::writeMetaData(filename, metaData);
            writeCsv(arg, file);
            closeFile(file);
        } else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }
};

#endif // SRC_RUNTIME_LOCAL_KERNELS_WRITE_H
