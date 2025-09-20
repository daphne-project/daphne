// parquetPlugIn.cpp
// Compile with:
//   g++ -std=c++17 -fPIC -shared \
//     parquetPlugIn.cpp -o libparquetPlugIn.so \
//     -I/home/yazan/daphneFork/daphne/src \
//     -I/home/yazan/daphneFork/daphne/thirdparty/installed/include \
//     -L/home/yazan/daphneFork/daphne/thirdparty/installed/lib \
//     -larrow -lparquet -lMLIRIR -lMLIRSupport -lLLVMOption \
//     -lLLVMSupport -lLLVMDemangle -lLLVMCore \
//     -lDataStructures -lspdlog -lfmt

#include <iostream>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/io/FileMetaData.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/kernels/CreateFrame.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/arrow/writer.h>

#include <thread>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <mutex>
#include <exception>

extern "C" {

void parquet_read(
    Structure *&res,
    const FileMetaData &fmd,
    const char *filename,
    IOOptions &opts,
    DaphneContext *ctx
) {
    using arrow::io::MemoryMappedFile;
    using arrow::io::FileMode;
    using parquet::ParquetFileReader;


    // 1) mmap + metadata
    auto mmr = MemoryMappedFile::Open(filename, FileMode::READ);
    if (!mmr.ok())
        throw std::runtime_error("mmap failed: " + mmr.status().ToString());
    auto infile = *mmr;

    std::unique_ptr<ParquetFileReader> pf = ParquetFileReader::Open(infile);
    if (!pf)
        throw std::runtime_error("ParquetFileReader::Open failed");
    auto meta = pf->metadata();

    int numRG = meta->num_row_groups();
    int cols  = meta->num_columns();
    int64_t rows = fmd.numRows;
    if (cols != (int)fmd.numCols)
        throw std::runtime_error("column count mismatch");

    // 2) compute row-group offsets
    std::vector<int64_t> rgOffset(numRG+1, 0);
    for (int rg = 0; rg < numRG; ++rg)
        rgOffset[rg+1] = rgOffset[rg] + meta->RowGroup(rg)->num_rows();
    if (rgOffset[numRG] != rows)
        throw std::runtime_error("row count mismatch");

    // 3) thread count
    int threads = 1;
    if (auto it = opts.extra.find("threads"); it != opts.extra.end())
        threads = std::max(1, std::stoi(it->second));

    // 4) dispatch by type
    if (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::SI32) {
        // allocate one big row-major matrix
        auto *mat = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);
        int32_t *data = mat->getValues();

        // parallel RG scatter
        std::mutex  mx; std::exception_ptr err;
        int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            int rg0 = t * rgPer;
            int rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;

            workers.emplace_back([&, rg0, rg1](){
                try {
                    // each thread opens its own Arrow FileReader
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(
                        parquet::arrow::OpenFile(infile, pool, &rdr)
                    );

                    // prepare column list
                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    // read & scatter
                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(
                            rdr->ReadRowGroup(rg, all_cols, &tbl)
                        );
                        int64_t base = rgOffset[rg];
                        int64_t nRG  = tbl->num_rows();

                        // for each column, memcpy into row-major buffer
                        for (int c = 0; c < cols; ++c) {
                            auto arr = std::static_pointer_cast<arrow::Int32Array>(
                                           tbl->column(c)->chunk(0));
                            const int32_t *src = arr->raw_values();
                            int32_t *dst = data + base*cols + c;
                            // now we need to copy nRG values at stride=cols:
                            for (int64_t i = 0; i < nRG; ++i) {
                                dst[i*cols] = src[i];
                            }
                        }
                    }
                }
                catch(...) {
                    std::lock_guard<std::mutex> guard(mx);
                    if (!err) err = std::current_exception();
                }
            });
        }

        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);

        res = mat;
    }
    else if (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::F64) {
        auto *mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
        double *data = mat->getValues();

        std::mutex mx; std::exception_ptr err;
        int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            int rg0 = t*rgPer, rg1 = std::min(numRG, rg0+rgPer);
            if (rg0 >= rg1) break;
            workers.emplace_back([&, rg0, rg1](){
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(
                        parquet::arrow::OpenFile(infile, pool, &rdr)
                    );

                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(
                            rdr->ReadRowGroup(rg, all_cols, &tbl)
                        );
                        int64_t base = rgOffset[rg];
                        int64_t nRG  = tbl->num_rows();

                        for (int c = 0; c < cols; ++c) {
                            auto arr = std::static_pointer_cast<arrow::DoubleArray>(
                                           tbl->column(c)->chunk(0));
                            const double *src = arr->raw_values();
                            double *dst = data + base*cols + c;
                            for (int64_t i = 0; i < nRG; ++i)
                                dst[i*cols] = src[i];
                        }
                    }
                }
                catch(...) {
                    std::lock_guard<std::mutex> guard(mx);
                    if (!err) err = std::current_exception();
                }
            });
        }
        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);

        res = mat;
    }
    else {
        // string case: unchanged per-element copy
        auto *mat = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
        std::string *data = mat->getValues();

        std::mutex mx; std::exception_ptr err;
        int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;
            workers.emplace_back([&, rg0, rg1](){
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(
                        parquet::arrow::OpenFile(infile, pool, &rdr)
                    );

                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(
                            rdr->ReadRowGroup(rg, all_cols, &tbl)
                        );
                        int64_t base = rgOffset[rg];
                        int64_t nRG  = tbl->num_rows();

                        for (int c = 0; c < cols; ++c) {
                            auto arr = std::static_pointer_cast<arrow::StringArray>(
                                           tbl->column(c)->chunk(0));
                            for (int64_t i = 0; i < nRG; ++i)
                                data[(base + i)*cols + c] = arr->GetString(i);
                        }
                    }
                }
                catch(...) {
                    std::lock_guard<std::mutex> guard(mx);
                    if (!err) err = std::current_exception();
                }
            });
        }
        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);

        res = mat;
    }
}

void parquet_read_frame(
    Frame *&res,
    const FileMetaData &fmd,
    const char *filename,
    IOOptions &opts,
    DaphneContext *ctx
) {
    using arrow::io::MemoryMappedFile;
    using arrow::io::FileMode;
    using parquet::ParquetFileReader;

    // 1) mmap + metadata
    auto mmr = MemoryMappedFile::Open(filename, FileMode::READ);
    if (!mmr.ok())
        throw std::runtime_error("mmap failed: " + mmr.status().ToString());
    auto infile = *mmr;

    std::unique_ptr<ParquetFileReader> pf = ParquetFileReader::Open(infile);
    if (!pf)
        throw std::runtime_error("ParquetFileReader::Open failed");
    auto meta = pf->metadata();

    int numRG = meta->num_row_groups();
    int cols  = meta->num_columns();
    int64_t rows = fmd.numRows;
    if (cols != (int)fmd.numCols)
        throw std::runtime_error("column count mismatch");

    // compute row-group row offsets
    std::vector<int64_t> rgOffset(numRG+1,0);
    for (int rg = 0; rg < numRG; ++rg)
        rgOffset[rg+1] = rgOffset[rg] + meta->RowGroup(rg)->num_rows();
    if (rgOffset[numRG] != rows)
        throw std::runtime_error("row count mismatch");

    // 2) labels from fmd
    std::vector<const char*> colLabels(cols);
    if ((int)fmd.labels.size() < cols){
        throw std::runtime_error("Not enough labels in FileMetaData");
    } else {
        for (int c = 0; c < cols; ++c)colLabels[c] = fmd.labels[c].c_str();
    }
    // 3) allocate one  rows×1 matrix per column
    std::vector<Structure*> columns(cols);
    for (int c = 0; c < cols; ++c) {
        switch(fmd.schema[c]) {
            case ValueTypeCode::SI32:
                columns[c] = DataObjectFactory::create<DenseMatrix<int32_t>>(rows,1,true);
                break;
            case ValueTypeCode::F64:
                columns[c] = DataObjectFactory::create<DenseMatrix<double>>(rows,1,true);
                break;
            default:
                columns[c] = DataObjectFactory::create<DenseMatrix<std::string>>(rows,1,true);
        }
    }

    // 4) threading
    int threads = 1;
    if (auto it = opts.extra.find("threads"); it != opts.extra.end())
        threads = std::max(1, std::stoi(it->second));

    std::mutex mtx;
    std::exception_ptr eptr;
    int rgPer = (numRG + threads - 1)/threads;
    std::vector<std::thread> workers;
    workers.reserve(threads);

    // all column indices
    std::vector<int> all_cols(cols);
    std::iota(all_cols.begin(), all_cols.end(), 0);

    for(int t=0; t<threads; ++t) {
        int startRG = t*rgPer;
        int endRG   = std::min(numRG, startRG+rgPer);
        if(startRG>=endRG) break;

        workers.emplace_back([&,startRG,endRG](){
            try {
                // each thread: fresh FileReader
                arrow::MemoryPool* pool = arrow::default_memory_pool();
                std::unique_ptr<parquet::arrow::FileReader> reader;
                PARQUET_THROW_NOT_OK(
                    parquet::arrow::OpenFile(infile, pool, &reader)
                );

                for(int rg=startRG; rg<endRG; ++rg) {
                    std::shared_ptr<arrow::Table> tbl;
                    PARQUET_THROW_NOT_OK(
                        reader->ReadRowGroup(rg, all_cols, &tbl)
                    );
                    int64_t base = rgOffset[rg];
                    int64_t nRG  = tbl->num_rows();

                    for(int c=0; c<cols; ++c) {
                        if(fmd.schema[c]==ValueTypeCode::SI32) {
                            auto arr = std::static_pointer_cast<arrow::Int32Array>(
                                         tbl->column(c)->chunk(0));
                            int32_t *dst = static_cast<DenseMatrix<int32_t>*>(columns[c])
                                              ->getValues();
                            const int32_t *src = arr->raw_values();
                            // memcpy contiguous block into column-matrix
                            std::memcpy(
                                dst + base,
                                src,
                                nRG * sizeof(int32_t)
                            );
                        }
                        else if(fmd.schema[c]==ValueTypeCode::F64) {
                            auto arr = std::static_pointer_cast<arrow::DoubleArray>(
                                         tbl->column(c)->chunk(0));
                            double *dst = static_cast<DenseMatrix<double>*>(columns[c])
                                              ->getValues();
                            const double *src = arr->raw_values();
                            std::memcpy(
                                dst + base,
                                src,
                                nRG * sizeof(double)
                            );
                        }
                        else {
                            auto arr = std::static_pointer_cast<arrow::StringArray>(
                                         tbl->column(c)->chunk(0));
                            auto *dstMat = static_cast<DenseMatrix<std::string>*>(columns[c]);
                            std::string *dst = dstMat->getValues();
                            for(int64_t i=0; i<nRG; ++i)
                                dst[base + i] = arr->GetString(i);
                        }
                    }
                }
            }
            catch(...) {
                std::lock_guard<std::mutex> lock(mtx);
                if(!eptr) eptr = std::current_exception();
            }
        });
    }

    for(auto &w : workers) w.join();
    if(eptr) std::rethrow_exception(eptr);

    // 5) assemble Frame
    createFrame(
        res,
        columns.data(),
        cols,
        colLabels.data(),
        cols,
        ctx
    );
}

void parquet_write(
    const void *data,
    const FileMetaData &fmd,
    const char *filename,
    const IOOptions &opts,
    DaphneContext * ctx
) {
    using arrow::Array;
    using arrow::DoubleBuilder;
    using arrow::Int32Builder;
    using arrow::StringBuilder;
    using arrow::default_memory_pool;
    using arrow::io::FileOutputStream;

    const int64_t rows = static_cast<int64_t>(fmd.numRows);
    const int64_t cols = static_cast<int64_t>(fmd.numCols);
    if(rows < 0 || cols <= 0)
        throw std::runtime_error("parquet_write: invalid shape");

    // Row group / chunk size (optional)
    int64_t chunk_size = 1024 * 1024; // default 1M rows per row group
    if (auto it = opts.extra.find("row_group_size"); it != opts.extra.end()) {
        try { chunk_size = std::max<int64_t>(1, std::stoll(it->second)); } catch(...) {}
    }

    // Compression (optional): snappy|gzip|zstd|brotli|lz4|none
    parquet::Compression::type comp = parquet::Compression::SNAPPY;
    if (auto it = opts.extra.find("compression"); it != opts.extra.end()) {
        std::string s = it->second; std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        if(s == "none"   || s == "uncompressed") comp = parquet::Compression::UNCOMPRESSED;
        else if(s == "gzip" || s == "zlib")      comp = parquet::Compression::GZIP;
        else if(s == "zstd")                     comp = parquet::Compression::ZSTD;
        else if(s == "brotli")                   comp = parquet::Compression::BROTLI;
        else if(s == "lz4")                      comp = parquet::Compression::LZ4;
        else if(s == "snappy")                   comp = parquet::Compression::SNAPPY;
    }
    auto props = parquet::WriterProperties::Builder().compression(comp)->build();

    // Column labels
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(cols);
    std::vector<std::shared_ptr<Array>> columns;
    columns.reserve(cols);

    // Helper to pick a name
    auto col_name = [&](int64_t c)->std::string {
        if (c < (int64_t)fmd.labels.size() && !fmd.labels[c].empty())
            return fmd.labels[c];
        return "col_" + std::to_string(c);
    };

    // Build arrays column-wise based on value type
    if (fmd.isSingleValueType && fmd.schema.size() >= 1 && fmd.schema[0] == ValueTypeCode::SI32) {
        auto mat = static_cast<const DenseMatrix<int32_t>*>(data);
        const int32_t *vals = mat->getValues();
        for (int64_t c = 0; c < cols; ++c) {
            Int32Builder b(default_memory_pool());
            PARQUET_THROW_NOT_OK(b.Reserve(rows));
            for (int64_t r = 0; r < rows; ++r)
                b.UnsafeAppend(vals[r * cols + c]);
            std::shared_ptr<Array> arr;
            PARQUET_THROW_NOT_OK(b.Finish(&arr));
            fields.push_back(arrow::field(col_name(c), arrow::int32()));
            columns.push_back(std::move(arr));
        }
    }
    else if (fmd.isSingleValueType && fmd.schema.size() >= 1 && fmd.schema[0] == ValueTypeCode::F64) {
        auto mat = static_cast<const DenseMatrix<double>*>(data);
        const double *vals = mat->getValues();
        for (int64_t c = 0; c < cols; ++c) {
            DoubleBuilder b(default_memory_pool());
            PARQUET_THROW_NOT_OK(b.Reserve(rows));
            for (int64_t r = 0; r < rows; ++r)
                b.UnsafeAppend(vals[r * cols + c]);
            std::shared_ptr<Array> arr;
            PARQUET_THROW_NOT_OK(b.Finish(&arr));
            fields.push_back(arrow::field(col_name(c), arrow::float64()));
            columns.push_back(std::move(arr));
        }
    }
    else {
        // Treat as string matrix
        auto mat = static_cast<const DenseMatrix<std::string>*>(data);
        const std::string *vals = mat->getValues();
        for (int64_t c = 0; c < cols; ++c) {
            StringBuilder b(default_memory_pool());
            for (int64_t r = 0; r < rows; ++r)
                PARQUET_THROW_NOT_OK(b.Append(vals[r * cols + c]));
            std::shared_ptr<Array> arr;
            PARQUET_THROW_NOT_OK(b.Finish(&arr));
            fields.push_back(arrow::field(col_name(c), arrow::utf8()));
            columns.push_back(std::move(arr));
        }
    }

    auto sch = arrow::schema(fields);
    auto tbl = arrow::Table::Make(sch, columns, rows);

    auto out_res = FileOutputStream::Open(filename);
    if (!out_res.ok())
        throw std::runtime_error("parquet_write: cannot open file: " + out_res.status().ToString());
    std::shared_ptr<arrow::io::OutputStream> out = *out_res;

    // Write table
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*tbl, default_memory_pool(), out, chunk_size, props));
}

void parquet_write_frame(
    const void *data,
    const FileMetaData &fmd,
    const char *filename,
    const IOOptions &opts,
    DaphneContext * ctx
) {
    using arrow::Array;
    using arrow::DoubleBuilder;
    using arrow::Int32Builder;
    using arrow::StringBuilder;
    using arrow::default_memory_pool;
    using arrow::io::FileOutputStream;

    const auto *frame = static_cast<const Frame*>(data);
    const int64_t rows = static_cast<int64_t>(fmd.numRows);
    const int64_t cols = static_cast<int64_t>(fmd.numCols);
    if (rows < 0 || cols <= 0)
        throw std::runtime_error("parquet_write_frame: invalid shape");

    // Options: row group size + compression
    int64_t row_group_size = 1024 * 1024; // default 1M rows/rg
    if (auto it = opts.extra.find("row_group_size"); it != opts.extra.end()) {
        try { row_group_size = std::max<int64_t>(1, std::stoll(it->second)); } catch(...) {}
    }

    parquet::Compression::type comp = parquet::Compression::SNAPPY;
    if (auto it = opts.extra.find("compression"); it != opts.extra.end()) {
        std::string s = it->second; std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        if      (s == "none" || s == "uncompressed") comp = parquet::Compression::UNCOMPRESSED;
        else if (s == "gzip" || s == "zlib")         comp = parquet::Compression::GZIP;
        else if (s == "zstd")                         comp = parquet::Compression::ZSTD;
        else if (s == "brotli")                       comp = parquet::Compression::BROTLI;
        else if (s == "lz4")                          comp = parquet::Compression::LZ4;
        else if (s == "snappy")                       comp = parquet::Compression::SNAPPY;
    }
    auto props = parquet::WriterProperties::Builder().compression(comp)->build();

    // Build Arrow schema + columns
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(cols);
    std::vector<std::shared_ptr<Array>> arrays;
    arrays.reserve(cols);

    auto col_name = [&](int64_t c)->std::string {
        if (c < (int64_t)fmd.labels.size() && !fmd.labels[c].empty())
            return fmd.labels[c];
        return "col_" + std::to_string(c);
    };

    for (int64_t c = 0; c < cols; ++c) {
        switch (fmd.schema[c]) {
            case ValueTypeCode::SI32: {
                // Expect column stored as DenseMatrix<int32_t> of shape rows x 1
                const auto *col = frame->getColumn<int32_t>(static_cast<size_t>(c));
                const int32_t *vals = col->getValues();
                Int32Builder b(default_memory_pool());
                PARQUET_THROW_NOT_OK(b.Reserve(rows));
                for (int64_t r = 0; r < rows; ++r)
                    b.UnsafeAppend(vals[r]); // contiguous column vector
                std::shared_ptr<Array> arr;
                PARQUET_THROW_NOT_OK(b.Finish(&arr));
                fields.push_back(arrow::field(col_name(c), arrow::int32()));
                arrays.push_back(std::move(arr));
                break;
            }
            case ValueTypeCode::F64: {
                const auto *col = frame->getColumn<double>(static_cast<size_t>(c));
                const double *vals = col->getValues();
                DoubleBuilder b(default_memory_pool());
                PARQUET_THROW_NOT_OK(b.Reserve(rows));
                for (int64_t r = 0; r < rows; ++r)
                    b.UnsafeAppend(vals[r]);
                std::shared_ptr<Array> arr;
                PARQUET_THROW_NOT_OK(b.Finish(&arr));
                fields.push_back(arrow::field(col_name(c), arrow::float64()));
                arrays.push_back(std::move(arr));
                break;
            }
            default: {
                // Treat any other type as string (mirrors your reader)
                const auto *col = frame->getColumn<std::string>(static_cast<size_t>(c));
                const std::string *vals = col->getValues();
                StringBuilder b(default_memory_pool());
                for (int64_t r = 0; r < rows; ++r)
                    PARQUET_THROW_NOT_OK(b.Append(vals[r]));
                std::shared_ptr<Array> arr;
                PARQUET_THROW_NOT_OK(b.Finish(&arr));
                fields.push_back(arrow::field(col_name(c), arrow::utf8()));
                arrays.push_back(std::move(arr));
                break;
            }
        }
    }

    auto schema = arrow::schema(fields);
    auto table  = arrow::Table::Make(schema, arrays, rows);

    auto out_res = FileOutputStream::Open(filename);
    if (!out_res.ok())
        throw std::runtime_error("parquet_write_frame: cannot open file: " + out_res.status().ToString());
    std::shared_ptr<arrow::io::OutputStream> out = *out_res;

    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, default_memory_pool(), out, row_group_size, props));
}

} // extern "C"