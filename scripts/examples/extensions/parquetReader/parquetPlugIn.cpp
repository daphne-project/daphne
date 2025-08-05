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
    if ((int)fmd.labels.size() < cols)
        throw std::runtime_error("Not enough labels in FileMetaData");
    std::vector<const char*> colLabels(cols);
    for (int c = 0; c < cols; ++c)
        colLabels[c] = fmd.labels[c].c_str();

    // 3) allocate one  rows×1 matrix per column
    std::vector<Structure*> columns(cols);
    for (int c = 0; c < cols; ++c) {
        switch(fmd.schema[c]) {
            case ValueTypeCode::SI32:
                columns[c] = DataObjectFactory::create<DenseMatrix<int32_t>>(rows,1,false);
                break;
            case ValueTypeCode::F64:
                columns[c] = DataObjectFactory::create<DenseMatrix<double>>(rows,1,false);
                break;
            default:
                columns[c] = DataObjectFactory::create<DenseMatrix<std::string>>(rows,1,false);
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
} // extern "C"