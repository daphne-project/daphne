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

// ****************************************************************************
// Helper: Merge a Frame* of key→value overrides into the default IOOptions
// ****************************************************************************
static IOOptions mergeOptionsFromFrame(const std::string &ext,
                                       IODataType         dt,
                                       Frame             *optsFrame)
{
    // 1) Retrieve the plugin's default options
    const IOOptions &defaults =
        FileIORegistry::instance().getOptions(ext, dt);

    // 2) Copy defaults into merged
    IOOptions merged = defaults;

    // 3) If optsFrame is non-null, convert each row (key,value) and override
    if(optsFrame != nullptr) {
        const size_t nRows = optsFrame->getNumRows();
        const size_t nCols = optsFrame->getNumCols();
        if(nCols < 2)
            throw std::runtime_error(
                "Options frame must have at least 2 columns (key, value)"
            );
        size_t temp = 0;
        auto* keyCol = dynamic_cast<DenseMatrix<std::string>*>(optsFrame->getColumn<std::string>(temp));
        auto* valCol = dynamic_cast<DenseMatrix<std::string>*>(optsFrame->getColumn<std::string>(1));

        if(!keyCol || !valCol)
            throw std::runtime_error("Expected string columns for key/value in options Frame");

        for(size_t i = 0; i < nRows; ++i) {
            
            const std::string& key = keyCol->get(i, 0);
            const std::string& value = valCol->get(i, 0);

            // 4) Verify this key exists in the plugin defaults
            if(merged.extra.find(key) == merged.extra.end()) {
                throw std::runtime_error(
                    std::string("Unknown option '") + key + "'"
                );
            }
            // 5) Override
            merged.extra[key] = value;
        }
    }

    return merged;
}

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DTRes> struct Read {
    static void apply(DTRes *&res, const char *filename, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DTRes> void read(DTRes *&res, const char *filename, DCTX(ctx)) {
    Read<DTRes>::apply(res, filename, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------

template <typename VT> struct Read<DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *&res, const char *filename, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());

        try {
            auto &registry = FileIORegistry::instance();
            IODataType typeHash = DENSEMATRIX;
            auto reader = registry.getReader(ext, typeHash);
            IOOptions opts = registry.getOptions(ext, typeHash);
            reader(&res, fmd, filename, opts, ctx);
            return;
        } catch (const std::out_of_range &) {
            // no plugin, fall back to built-in
        }

        if (ext == ".csv") {
            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
            readCsv(res, filename, fmd.numRows, fmd.numCols, ',');
        } else if (ext == ".mtx") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued MatrixMarket files is not supported (yet)");
            else
                readMM(res, filename);
        } else if (ext == ".parquet") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued Parquet files is not supported (yet)");
            else {
                if (res == nullptr)
                    res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
                readParquet(res, filename, fmd.numRows, fmd.numCols);
            }
        } else if (ext == ".dbdf") {
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

        try {
            auto &registry = FileIORegistry::instance();
            IODataType typeHash = DENSEMATRIX;

            // Merge user overrides from optsFrame
            IOOptions mergedOpts = mergeOptionsFromFrame(ext, typeHash, optsFrame);

            auto reader = registry.getReader(ext, typeHash);
            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        }
        catch (const std::out_of_range &) {
            // no plugin, fall back to built-in
        }

        if (ext == ".csv") {
            if (res == nullptr)
                res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
            readCsv(res, filename, fmd.numRows, fmd.numCols, ',');
        } else if (ext == ".mtx") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued MatrixMarket files is not supported (yet)");
            else
                readMM(res, filename);
        } else if (ext == ".parquet") {
            if constexpr (std::is_same<VT, std::string>::value)
                throw std::runtime_error("reading string-valued Parquet files is not supported (yet)");
            else {
                if (res == nullptr)
                    res = DataObjectFactory::create<DenseMatrix<VT>>(fmd.numRows, fmd.numCols, false);
                readParquet(res, filename, fmd.numRows, fmd.numCols);
            }
        } else if (ext == ".dbdf") {
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
    static void apply(CSRMatrix<VT> *&res, const char *filename, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());

        try {
            auto &registry = FileIORegistry::instance();
            IODataType typeHash = CSRMATRIX;
            auto reader = registry.getReader(ext, typeHash);
            IOOptions opts = registry.getOptions(ext, typeHash);

            reader(&res, fmd, filename, opts, ctx);
            return;
        } catch (const std::out_of_range &) {
            // no plugin, fall back to built-in
        }

        if (ext == ".csv") {
            if (fmd.numNonZeros == -1)
                throw std::runtime_error("currently reading of sparse matrices requires a number of "
                                         "non zeros to be defined");

            if (res == nullptr)
                res = DataObjectFactory::create<CSRMatrix<VT>>(fmd.numRows, fmd.numCols, fmd.numNonZeros, false);

            // FIXME: ensure file is sorted, or set `sorted` argument correctly
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

        try {
            auto &registry = FileIORegistry::instance();
            IODataType typeHash = CSRMATRIX;

            // Merge user overrides from optsFrame
            IOOptions mergedOpts =
                mergeOptionsFromFrame(ext, typeHash, optsFrame);

            auto reader = registry.getReader(ext, typeHash);
            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        }
        catch (const std::out_of_range &) {
            // no plugin, fall back to built-in
        }

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
    static void apply(Frame *&res, const char *filename, DCTX(ctx)) {
        FileMetaData fmd = MetaDataParser::readMetaData(filename);
        std::string ext(std::filesystem::path(filename).extension());

        try {
            auto &registry = FileIORegistry::instance();
            IODataType typeHash = FRAME;
            auto reader = registry.getReader(ext, typeHash);
            IOOptions opts = registry.getOptions(ext, typeHash);
            reader(&res, fmd, filename, opts, ctx);
            return;
        } catch (const std::out_of_range &) {
            // no plugin, fall back to built-in
        }

        if (ext == ".csv") {
            ValueTypeCode *schema;
            if (fmd.isSingleValueType) {
                schema = new ValueTypeCode[fmd.numCols];
                for (size_t i = 0; i < fmd.numCols; i++)
                    schema[i] = fmd.schema[0];
            } else
                schema = fmd.schema.data();

            std::string *labels;
            if (fmd.labels.empty())
                labels = nullptr;
            else
                labels = fmd.labels.data();

            if (res == nullptr)
                res = DataObjectFactory::create<Frame>(fmd.numRows, fmd.numCols, schema, labels, false);

            readCsv(res, filename, fmd.numRows, fmd.numCols, ',', schema);

            if (fmd.isSingleValueType)
                delete[] schema;
        } else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }

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

        try {
            auto &registry = FileIORegistry::instance();
            IODataType typeHash = FRAME;

            // Merge user overrides from optsFrame
            IOOptions mergedOpts =
                mergeOptionsFromFrame(ext, typeHash, optsFrame);

            auto reader = registry.getReader(ext, typeHash);
            reader(&res, fmd, filename, mergedOpts, ctx);
            return;
        } catch (const std::out_of_range &) {
            // no plugin, fall back to built-in
        }

        if (ext == ".csv") {
            ValueTypeCode *schema;
            if (fmd.isSingleValueType) {
                schema = new ValueTypeCode[fmd.numCols];
                for (size_t i = 0; i < fmd.numCols; i++)
                    schema[i] = fmd.schema[0];
            } else
                schema = fmd.schema.data();

            std::string *labels;
            if (fmd.labels.empty())
                labels = nullptr;
            else
                labels = fmd.labels.data();

            if (res == nullptr)
                res = DataObjectFactory::create<Frame>(fmd.numRows, fmd.numCols, schema, labels, false);

            readCsv(res, filename, fmd.numRows, fmd.numCols, ',', schema);

            if (fmd.isSingleValueType)
                delete[] schema;
        } else
            throw std::runtime_error("file extension not supported: '" + ext + "'");
    }
}; // end Read<Frame>

// ****************************************************************************
// Convenience overloads for calling read(..., optsFrame)
// ****************************************************************************

// For DenseMatrix<VT>
template <typename VT>
inline void read(DenseMatrix<VT> *&res,
                 const char       *filename,
                 Frame            *optsFrame,
                 DCTX(ctx))
{
    Read<DenseMatrix<VT>>::apply(res, filename, optsFrame, ctx);
}

// For CSRMatrix<VT>
template <typename VT>
inline void read(CSRMatrix<VT> *&res,
                 const char     *filename,
                 Frame          *optsFrame,
                 DCTX(ctx))
{
    Read<CSRMatrix<VT>>::apply(res, filename, optsFrame, ctx);
}

// For Frame
inline void read(Frame  *&res,
                 const char *filename,
                 Frame      *optsFrame,
                 DCTX(ctx))
{
    Read<Frame>::apply(res, filename, optsFrame, ctx);
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_READ_H