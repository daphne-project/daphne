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
#include <arrow/compute/api.h>

#include <thread>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <mutex>
#include <exception>

// Cast an Arrow Array to target type if needed (returns original if already matching).
static std::shared_ptr<arrow::Array> cast_array_if_needed(
    const std::shared_ptr<arrow::Array>& arr,
    const std::shared_ptr<arrow::DataType>& target_type,
    arrow::MemoryPool* pool)
{
    if (arr->type()->Equals(target_type)) {
        return arr;
    }
    arrow::compute::CastOptions copts;
    copts.to_type = target_type;
    copts.allow_int_overflow = true;
    copts.allow_time_truncate = true;
    auto res = arrow::compute::Cast(arr, copts);
    if (!res.ok()) {
        throw std::runtime_error("Arrow cast failed: " + res.status().ToString());
    }
    return res.ValueOrDie().make_array();
}

// Return Parquet leaf column names in canonical (left-to-right) order as dot paths, e.g. "a.b.c".
static std::vector<std::string> parquet_leaf_names(const parquet::FileMetaData* meta) {
    std::vector<std::string> names;
    auto sch = meta->schema(); // parquet::schema::SchemaDescriptor
    const int leaves = sch->num_columns();
    names.reserve(leaves);
    for (int i = 0; i < leaves; ++i) {
        names.emplace_back(sch->Column(i)->path()->ToDotString());
    }
    return names;
}

// Materialize a value from a dictionary's data array at 'code' as std::string.
static std::string materialize_dict_value(const std::shared_ptr<arrow::Array>& dict, int64_t code) {
    using T = arrow::Type;
    switch (dict->type_id()) {
        case T::STRING: {
            auto d = std::static_pointer_cast<arrow::StringArray>(dict);
            return (code >= 0 && code < d->length() && d->IsValid(code)) ? d->GetString(code) : std::string{};
        }
        case T::LARGE_STRING: {
            auto d = std::static_pointer_cast<arrow::LargeStringArray>(dict);
            return (code >= 0 && code < d->length() && d->IsValid(code)) ? d->GetString(code) : std::string{};
        }
        case T::BINARY: {
            auto d = std::static_pointer_cast<arrow::BinaryArray>(dict);
            if (!(code >= 0 && code < d->length()) || !d->IsValid(code)) return {};
            auto v = d->GetView(code);
            return std::string(v.data(), v.size());
        }
        case T::LARGE_BINARY: {
            auto d = std::static_pointer_cast<arrow::LargeBinaryArray>(dict);
            if (!(code >= 0 && code < d->length()) || !d->IsValid(code)) return {};
            auto v = d->GetView(code);
            return std::string(v.data(), v.size());
        }
        default: {
            auto sres = dict->GetScalar(code);
            if (sres.ok() && sres.ValueOrDie()) return sres.ValueOrDie()->ToString();
            return {};
        }
    }
}

// Extract a single row i from arbitrary Arrow array as std::string (handles DICTIONARY safely).
static std::string get_string_at_any(const std::shared_ptr<arrow::Array>& arr, int64_t i, arrow::MemoryPool* pool) {
    using T = arrow::Type;
    if (!arr->IsValid(i)) return {}; // empty for nulls

    switch (arr->type_id()) {
        case T::STRING: {
            auto a = std::static_pointer_cast<arrow::StringArray>(arr);
            return a->GetString(i);
        }
        case T::LARGE_STRING: {
            auto a = std::static_pointer_cast<arrow::LargeStringArray>(arr);
            return a->GetString(i);
        }
        case T::BINARY: {
            auto a = std::static_pointer_cast<arrow::BinaryArray>(arr);
            auto v = a->GetView(i);
            return std::string(v.data(), v.size());
        }
        case T::LARGE_BINARY: {
            auto a = std::static_pointer_cast<arrow::LargeBinaryArray>(arr);
            auto v = a->GetView(i);
            return std::string(v.data(), v.size());
        }
        case T::DICTIONARY: {
            auto dict_arr = std::static_pointer_cast<arrow::DictionaryArray>(arr);

            // Cast indices to int32 (supports int8/16/32/64 signed/unsigned)
            arrow::compute::CastOptions copts;
            copts.to_type = arrow::int32();
            copts.allow_int_overflow = true;

            auto casted_idx_res = arrow::compute::Cast(dict_arr->indices(), copts);
            if (!casted_idx_res.ok()) {
                // fallback: stringify index scalar if casting fails
                auto s = dict_arr->indices()->GetScalar(i);
                return (s.ok() && s.ValueOrDie()) ? s.ValueOrDie()->ToString() : std::string{};
            }
            auto idx32 = std::static_pointer_cast<arrow::Int32Array>(casted_idx_res.ValueOrDie().make_array());
            if (!idx32->IsValid(i)) return {};
            int32_t code = idx32->Value(i);
            return materialize_dict_value(dict_arr->dictionary(), code);
        }
        default: {
            // Numeric/bool/timestamp/nested: fallback to Scalar::ToString()
            auto sres = arr->GetScalar(i);
            return (sres.ok() && sres.ValueOrDie()) ? sres.ValueOrDie()->ToString() : std::string{};
        }
    }
}

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
    std::shared_ptr<arrow::io::RandomAccessFile> infile = *mmr;

    std::unique_ptr<ParquetFileReader> pf = ParquetFileReader::Open(infile);
    if (!pf)
        throw std::runtime_error("ParquetFileReader::Open failed");
    std::shared_ptr<parquet::FileMetaData> meta = pf->metadata();

    const int numRG = meta->num_row_groups();
    const int cols  = meta->num_columns();           // PARQUET LEAF COLUMNS
    const int64_t rows = fmd.numRows;

    if (cols != static_cast<int>(fmd.numCols)) {
        throw std::runtime_error("column count mismatch: parquet(leaves)=" + std::to_string(cols) +
                                 " expected(meta)=" + std::to_string(fmd.numCols));
    }

    // 2) row-group offsets
    std::vector<int64_t> rgOffset(numRG + 1, 0);
    for (int rg = 0; rg < numRG; ++rg)
        rgOffset[rg + 1] = rgOffset[rg] + meta->RowGroup(rg)->num_rows();
    if (rgOffset[numRG] != rows)
        throw std::runtime_error("row count mismatch: parquet=" +
                                 std::to_string(rgOffset[numRG]) + " expected(meta)=" + std::to_string(rows));

    // Canonical leaf names in Parquet order; used to align RGs with differing visible fields
    const auto leafNames = parquet_leaf_names(meta.get()); // size == cols

    // 3) thread count
    int threads = 1;
    if (auto it = opts.extra.find("threads"); it != opts.extra.end())
        threads = std::max(1, std::stoi(it->second));

    const bool is_si8  = (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::SI8);
    const bool is_si32 = (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::SI32);
    const bool is_si64 = (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::SI64);

    const bool is_ui8  = (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::UI8);
    const bool is_ui32 = (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::UI32);
    const bool is_ui64 = (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::UI64);

    const bool is_f32  = (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::F32);
    const bool is_f64  = (fmd.isSingleValueType && fmd.schema[0] == ValueTypeCode::F64);

    if (is_si8) {
        auto *mat = DataObjectFactory::create<DenseMatrix<int8_t>>(rows, cols, false);
        int8_t *data = mat->getValues();

        std::mutex mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;

            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false);

                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        {
                            auto cc = tbl->CombineChunks(pool);
                            if (!cc.ok()) throw std::runtime_error(cc.status().ToString());
                            tbl = cc.ValueOrDie();
                            auto fl = tbl->Flatten();
                            if (!fl.ok()) throw std::runtime_error(fl.status().ToString());
                            tbl = fl.ValueOrDie();
                        }

                        const int64_t base = rgOffset[rg];
                        const int64_t nRG  = tbl->num_rows();

                        std::unordered_map<std::string, int> name2idx;
                        for (int j = 0; j < tbl->num_columns(); ++j)
                            name2idx.emplace(tbl->field(j)->name(), j);

                        for (int c = 0; c < cols; ++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it == name2idx.end()) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }
                            auto chunked = tbl->column(it->second);
                            if (!chunked) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }

                            int64_t logical_row = 0;
                            for (int k = 0; k < chunked->num_chunks(); ++k) {
                                auto arr0 = chunked->chunk(k);
                                if (!arr0) continue;
                                auto arr = std::static_pointer_cast<arrow::Int8Array>(
                                    cast_array_if_needed(arr0, arrow::int8(), pool)
                                );
                                const int64_t len = arr->length();
                                int8_t *dst_col0 = data + (base + logical_row) * cols + c;
                                if (arr->null_count() == 0) {
                                    const int8_t *src = arr->raw_values();
                                    for (int64_t i=0;i<len;++i) dst_col0[i*cols]=src[i];
                                } else {
                                    for (int64_t i=0;i<len;++i) dst_col0[i*cols] = arr->IsValid(i) ? arr->Value(i) : 0;
                                }
                                logical_row += len;
                            }
                            if (logical_row != nRG) throw std::runtime_error("RG row mismatch (int8)");
                        }
                    }
                } catch (...) { std::lock_guard<std::mutex> g(mx); if (!err) err = std::current_exception(); }
            });
        }
        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);
        res = mat;
        return;
    }


    // ---------- INT32 path ----------
    if (is_si32) {
        auto *mat = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);
        int32_t *data = mat->getValues();

        std::mutex  mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer;
            const int rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;

            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false);

                    // We will read all LEAF columns; later we align by leafNames
                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        if (!tbl) throw std::runtime_error("ReadRowGroup returned null Table");

                        // Combine chunks + flatten (structs -> leaves as fields)
                        {
                            auto cc = tbl->CombineChunks(pool);
                            if (!cc.ok()) throw std::runtime_error("CombineChunks failed: " + cc.status().ToString());
                            tbl = cc.ValueOrDie();

                            auto fl = tbl->Flatten();
                            if (!fl.ok()) throw std::runtime_error("Flatten failed: " + fl.status().ToString());
                            tbl = fl.ValueOrDie();
                        }

                        const int64_t base = rgOffset[rg];
                        const int64_t nRG  = tbl->num_rows();

                        // Map present field names -> index
                        std::unordered_map<std::string, int> name2idx;
                        name2idx.reserve(static_cast<size_t>(tbl->num_columns()));
                        for (int j = 0; j < tbl->num_columns(); ++j) {
                            name2idx.emplace(tbl->field(j)->name(), j);
                        }

                        // Iterate canonical Parquet leaves
                        for (int c = 0; c < cols; ++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it == name2idx.end()) {
                                // Missing in this RG: fill zeros
                                for (int64_t i = 0; i < nRG; ++i)
                                    data[(base + i) * cols + c] = 0;
                                continue;
                            }

                            auto chunked = tbl->column(it->second);
                            if (!chunked) {
                                for (int64_t i = 0; i < nRG; ++i)
                                    data[(base + i) * cols + c] = 0;
                                continue;
                            }

                            int64_t logical_row = 0;
                            const int n_chunks = chunked->num_chunks();
                            for (int k = 0; k < n_chunks; ++k) {
                                auto arr = chunked->chunk(k);
                                if (!arr) continue;

                                auto arr32 = std::static_pointer_cast<arrow::Int32Array>(
                                    cast_array_if_needed(arr, arrow::int32(), pool)
                                );

                                const int64_t len = arr32->length();
                                const int32_t* src = arr32->raw_values();
                                int32_t *dst_col0 = data + (base + logical_row) * cols + c;

                                if (arr32->null_count() == 0) {
                                    for (int64_t i = 0; i < len; ++i) dst_col0[i * cols] = src[i];
                                } else {
                                    for (int64_t i = 0; i < len; ++i)
                                        dst_col0[i * cols] = arr32->IsValid(i) ? src[i] : 0;
                                }
                                logical_row += len;
                            }
                            if (logical_row != nRG)
                                throw std::runtime_error("RG row mismatch after chunking (int32)");
                        }
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> guard(mx);
                    if (!err) err = std::current_exception();
                }
            });
        }

        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);
        res = mat;
        return;
    }

    if (is_si64) {
        auto *mat = DataObjectFactory::create<DenseMatrix<int64_t>>(rows, cols, false);
        int64_t *data = mat->getValues();

        std::mutex mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;

            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false);

                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        {
                            auto cc = tbl->CombineChunks(pool);
                            if (!cc.ok()) throw std::runtime_error(cc.status().ToString());
                            tbl = cc.ValueOrDie();
                            auto fl = tbl->Flatten();
                            if (!fl.ok()) throw std::runtime_error(fl.status().ToString());
                            tbl = fl.ValueOrDie();
                        }

                        const int64_t base = rgOffset[rg];
                        const int64_t nRG  = tbl->num_rows();

                        std::unordered_map<std::string, int> name2idx;
                        for (int j = 0; j < tbl->num_columns(); ++j)
                            name2idx.emplace(tbl->field(j)->name(), j);

                        for (int c = 0; c < cols; ++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it == name2idx.end()) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }
                            auto chunked = tbl->column(it->second);
                            if (!chunked) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }

                            int64_t logical_row = 0;
                            for (int k = 0; k < chunked->num_chunks(); ++k) {
                                auto arr0 = chunked->chunk(k);
                                if (!arr0) continue;
                                auto arr = std::static_pointer_cast<arrow::Int64Array>(
                                    cast_array_if_needed(arr0, arrow::int64(), pool)
                                );
                                const int64_t len = arr->length();
                                int64_t *dst_col0 = data + (base + logical_row) * cols + c;
                                if (arr->null_count() == 0) {
                                    const int64_t *src = arr->raw_values();
                                    for (int64_t i=0;i<len;++i) dst_col0[i*cols]=src[i];
                                } else {
                                    for (int64_t i=0;i<len;++i) dst_col0[i*cols] = arr->IsValid(i) ? arr->Value(i) : 0;
                                }
                                logical_row += len;
                            }
                            if (logical_row != nRG) throw std::runtime_error("RG row mismatch (int64)");
                        }
                    }
                } catch (...) { std::lock_guard<std::mutex> g(mx); if (!err) err = std::current_exception(); }
            });
        }
        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);
        res = mat;
        return;
    }

    // ---------- UINT8 ----------
    if (is_ui8) {
        auto *mat = DataObjectFactory::create<DenseMatrix<uint8_t>>(rows, cols, false);
        uint8_t *data = mat->getValues();
        std::mutex mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;
            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false);
                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);
                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        auto cc = tbl->CombineChunks(pool); if (!cc.ok()) throw std::runtime_error(cc.status().ToString()); tbl = cc.ValueOrDie();
                        auto fl = tbl->Flatten(); if (!fl.ok()) throw std::runtime_error(fl.status().ToString()); tbl = fl.ValueOrDie();
                        const int64_t base = rgOffset[rg], nRG = tbl->num_rows();
                        std::unordered_map<std::string,int> name2idx; for (int j=0;j<tbl->num_columns();++j) name2idx.emplace(tbl->field(j)->name(), j);
                        for (int c=0;c<cols;++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it==name2idx.end()) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }
                            auto chunked = tbl->column(it->second); if (!chunked) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }
                            int64_t logical_row = 0;
                            for (int k=0;k<chunked->num_chunks();++k) {
                                auto arr0 = chunked->chunk(k); if (!arr0) continue;
                                auto arr = std::static_pointer_cast<arrow::UInt8Array>(cast_array_if_needed(arr0, arrow::uint8(), pool));
                                const int64_t len = arr->length();
                                uint8_t *dst_col0 = data + (base + logical_row) * cols + c;
                                if (arr->null_count()==0) { const uint8_t *src = arr->raw_values(); for (int64_t i=0;i<len;++i) dst_col0[i*cols]=src[i]; }
                                else { for (int64_t i=0;i<len;++i) dst_col0[i*cols] = arr->IsValid(i) ? arr->Value(i) : 0; }
                                logical_row += len;
                            }
                            if (logical_row != nRG) throw std::runtime_error("RG row mismatch (uint8)");
                        }
                    }
                } catch (...) { std::lock_guard<std::mutex> g(mx); if (!err) err = std::current_exception(); }
            });
        }
        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);
        res = mat;
        return;
    }

    if (is_ui32) {
        auto *mat = DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, cols, false);
        uint32_t *data = mat->getValues();
        std::mutex mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;
            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false);
                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);
                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        auto cc = tbl->CombineChunks(pool); if (!cc.ok()) throw std::runtime_error(cc.status().ToString()); tbl = cc.ValueOrDie();
                        auto fl = tbl->Flatten(); if (!fl.ok()) throw std::runtime_error(fl.status().ToString()); tbl = fl.ValueOrDie();
                        const int64_t base = rgOffset[rg], nRG = tbl->num_rows();
                        std::unordered_map<std::string,int> name2idx; for (int j=0;j<tbl->num_columns();++j) name2idx.emplace(tbl->field(j)->name(), j);
                        for (int c=0;c<cols;++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it==name2idx.end()) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }
                            auto chunked = tbl->column(it->second); if (!chunked) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }
                            int64_t logical_row = 0;
                            for (int k=0;k<chunked->num_chunks();++k) {
                                auto arr0 = chunked->chunk(k); if (!arr0) continue;
                                auto arr = std::static_pointer_cast<arrow::UInt32Array>(cast_array_if_needed(arr0, arrow::uint32(), pool));
                                const int64_t len = arr->length();
                                uint32_t *dst_col0 = data + (base + logical_row) * cols + c;
                                if (arr->null_count()==0) { const uint32_t *src = arr->raw_values(); for (int64_t i=0;i<len;++i) dst_col0[i*cols]=src[i]; }
                                else { for (int64_t i=0;i<len;++i) dst_col0[i*cols] = arr->IsValid(i) ? arr->Value(i) : 0; }
                                logical_row += len;
                            }
                            if (logical_row != nRG) throw std::runtime_error("RG row mismatch (uint32)");
                        }
                    }
                } catch (...) { std::lock_guard<std::mutex> g(mx); if (!err) err = std::current_exception(); }
            });
        }
        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);
        res = mat;
        return;
    }

    // ---------- UINT64 ----------
    if (is_ui64) {
        auto *mat = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false);
        uint64_t *data = mat->getValues();
        std::mutex mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;
            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false);
                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);
                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        auto cc = tbl->CombineChunks(pool); if (!cc.ok()) throw std::runtime_error(cc.status().ToString()); tbl = cc.ValueOrDie();
                        auto fl = tbl->Flatten(); if (!fl.ok()) throw std::runtime_error(fl.status().ToString()); tbl = fl.ValueOrDie();
                        const int64_t base = rgOffset[rg], nRG = tbl->num_rows();
                        std::unordered_map<std::string,int> name2idx; for (int j=0;j<tbl->num_columns();++j) name2idx.emplace(tbl->field(j)->name(), j);
                        for (int c=0;c<cols;++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it==name2idx.end()) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }
                            auto chunked = tbl->column(it->second); if (!chunked) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=0; continue; }
                            int64_t logical_row = 0;
                            for (int k=0;k<chunked->num_chunks();++k) {
                                auto arr0 = chunked->chunk(k); if (!arr0) continue;
                                auto arr = std::static_pointer_cast<arrow::UInt64Array>(cast_array_if_needed(arr0, arrow::uint64(), pool));
                                const int64_t len = arr->length();
                                uint64_t *dst_col0 = data + (base + logical_row) * cols + c;
                                if (arr->null_count()==0) { const uint64_t *src = arr->raw_values(); for (int64_t i=0;i<len;++i) dst_col0[i*cols]=src[i]; }
                                else { for (int64_t i=0;i<len;++i) dst_col0[i*cols] = arr->IsValid(i) ? arr->Value(i) : 0; }
                                logical_row += len;
                            }
                            if (logical_row != nRG) throw std::runtime_error("RG row mismatch (uint64)");
                        }
                    }
                } catch (...) { std::lock_guard<std::mutex> g(mx); if (!err) err = std::current_exception(); }
            });
        }
        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);
        res = mat;
        return;
    }

    // ---------- FLOAT (F32) ----------
    if (is_f32) {
        auto *mat = DataObjectFactory::create<DenseMatrix<float>>(rows, cols, false);
        float *data = mat->getValues();

        std::mutex mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;

            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false);

                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        {
                            auto cc = tbl->CombineChunks(pool);
                            if (!cc.ok()) throw std::runtime_error(cc.status().ToString());
                            tbl = cc.ValueOrDie();
                            auto fl = tbl->Flatten();
                            if (!fl.ok()) throw std::runtime_error(fl.status().ToString());
                            tbl = fl.ValueOrDie();
                        }

                        const int64_t base = rgOffset[rg];
                        const int64_t nRG  = tbl->num_rows();

                        std::unordered_map<std::string, int> name2idx;
                        for (int j = 0; j < tbl->num_columns(); ++j)
                            name2idx.emplace(tbl->field(j)->name(), j);

                        for (int c = 0; c < cols; ++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it == name2idx.end()) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=std::numeric_limits<float>::quiet_NaN(); continue; }
                            auto chunked = tbl->column(it->second);
                            if (!chunked) { for (int64_t i=0;i<nRG;++i) data[(base+i)*cols+c]=std::numeric_limits<float>::quiet_NaN(); continue; }

                            int64_t logical_row = 0;
                            for (int k = 0; k < chunked->num_chunks(); ++k) {
                                auto arr0 = chunked->chunk(k);
                                if (!arr0) continue;
                                auto arr = std::static_pointer_cast<arrow::FloatArray>(
                                    cast_array_if_needed(arr0, arrow::float32(), pool)
                                );
                                const int64_t len = arr->length();
                                float *dst_col0 = data + (base + logical_row) * cols + c;
                                if (arr->null_count() == 0) {
                                    const float *src = arr->raw_values();
                                    for (int64_t i=0;i<len;++i) dst_col0[i*cols]=src[i];
                                } else {
                                    for (int64_t i=0;i<len;++i)
                                        dst_col0[i*cols] = arr->IsValid(i) ? arr->Value(i) : std::numeric_limits<float>::quiet_NaN();
                                }
                                logical_row += len;
                            }
                            if (logical_row != nRG) throw std::runtime_error("RG row mismatch (float)");
                        }
                    }
                } catch (...) { std::lock_guard<std::mutex> g(mx); if (!err) err = std::current_exception(); }
            });
        }
        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);
        res = mat;
        return;
    }

    // ---------- DOUBLE path ----------
    if (is_f64) {
        auto *mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
        double *data = mat->getValues();

        std::mutex  mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;

            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false);

                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        if (!tbl) throw std::runtime_error("ReadRowGroup returned null Table");

                        // Combine chunks + flatten
                        {
                            auto cc = tbl->CombineChunks(pool);
                            if (!cc.ok()) throw std::runtime_error("CombineChunks failed: " + cc.status().ToString());
                            tbl = cc.ValueOrDie();

                            auto fl = tbl->Flatten();
                            if (!fl.ok()) throw std::runtime_error("Flatten failed: " + fl.status().ToString());
                            tbl = fl.ValueOrDie();
                        }

                        const int64_t base = rgOffset[rg];
                        const int64_t nRG  = tbl->num_rows();

                        std::unordered_map<std::string, int> name2idx;
                        name2idx.reserve(static_cast<size_t>(tbl->num_columns()));
                        for (int j = 0; j < tbl->num_columns(); ++j) {
                            name2idx.emplace(tbl->field(j)->name(), j);
                        }

                        for (int c = 0; c < cols; ++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it == name2idx.end()) {
                                for (int64_t i = 0; i < nRG; ++i)
                                    data[(base + i) * cols + c] = std::nan("");
                                continue;
                            }

                            auto chunked = tbl->column(it->second);
                            if (!chunked) {
                                for (int64_t i = 0; i < nRG; ++i)
                                    data[(base + i) * cols + c] = std::nan("");
                                continue;
                            }

                            int64_t logical_row = 0;
                            const int n_chunks = chunked->num_chunks();
                            for (int k = 0; k < n_chunks; ++k) {
                                auto arr = chunked->chunk(k);
                                if (!arr) continue;

                                auto arr64 = std::static_pointer_cast<arrow::DoubleArray>(
                                    cast_array_if_needed(arr, arrow::float64(), pool)
                                );

                                const int64_t len = arr64->length();
                                const double* src = arr64->raw_values();
                                double *dst_col0 = data + (base + logical_row) * cols + c;

                                if (arr64->null_count() == 0) {
                                    for (int64_t i = 0; i < len; ++i) dst_col0[i * cols] = src[i];
                                } else {
                                    for (int64_t i = 0; i < len; ++i)
                                        dst_col0[i * cols] = arr64->IsValid(i) ? src[i] : std::nan("");
                                }
                                logical_row += len;
                            }
                            if (logical_row != nRG)
                                throw std::runtime_error("RG row mismatch after chunking (double)");
                        }
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> guard(mx);
                    if (!err) err = std::current_exception();
                }
            });
        }

        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);
        res = mat;
        return;
    }

    // ---------- STRING (robust) path ----------
    {
        auto *mat = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
        std::string *data = mat->getValues();

        std::mutex mx; std::exception_ptr err;
        const int rgPer = (numRG + threads - 1) / threads;
        std::vector<std::thread> workers;

        for (int t = 0; t < threads; ++t) {
            const int rg0 = t * rgPer, rg1 = std::min(numRG, rg0 + rgPer);
            if (rg0 >= rg1) break;

            workers.emplace_back([&, rg0, rg1]() {
                try {
                    arrow::MemoryPool *pool = arrow::default_memory_pool();
                    std::unique_ptr<parquet::arrow::FileReader> rdr;
                    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, pool, &rdr));
                    rdr->set_use_threads(false); // external parallelism

                    std::vector<int> all_cols(cols);
                    std::iota(all_cols.begin(), all_cols.end(), 0);

                    for (int rg = rg0; rg < rg1; ++rg) {
                        std::shared_ptr<arrow::Table> tbl;
                        PARQUET_THROW_NOT_OK(rdr->ReadRowGroup(rg, all_cols, &tbl));
                        if (!tbl) throw std::runtime_error("ReadRowGroup returned null Table");

                        // 1) Merge chunks; 2) Flatten structs into leaves
                        {
                            auto cc = tbl->CombineChunks(pool);
                            if (!cc.ok()) throw std::runtime_error("CombineChunks failed: " + cc.status().ToString());
                            tbl = cc.ValueOrDie();

                            auto fl = tbl->Flatten();
                            if (!fl.ok()) throw std::runtime_error("Flatten failed: " + fl.status().ToString());
                            tbl = fl.ValueOrDie();
                        }

                        const int64_t base = rgOffset[rg];
                        const int64_t nRG  = tbl->num_rows();

                        // Build name -> index for present fields in this RG (flattened)
                        std::unordered_map<std::string, int> name2idx;
                        name2idx.reserve(static_cast<size_t>(tbl->num_columns()));
                        for (int j = 0; j < tbl->num_columns(); ++j) {
                            name2idx.emplace(tbl->field(j)->name(), j);
                        }

                        // Iterate canonical Parquet leaf names (len == meta->num_columns())
                        for (int c = 0; c < cols; ++c) {
                            auto it = name2idx.find(leafNames[c]);
                            if (it == name2idx.end()) {
                                // Missing column in this RG -> fill empties
                                for (int64_t i = 0; i < nRG; ++i) {
                                    data[(base + i) * cols + c].clear();
                                }
                                continue;
                            }

                            auto chunked = tbl->column(it->second);
                            if (!chunked) {
                                for (int64_t i = 0; i < nRG; ++i) data[(base + i) * cols + c].clear();
                                continue;
                            }

                            int64_t logical_row = 0;
                            const int n_chunks = chunked->num_chunks();
                            for (int k = 0; k < n_chunks; ++k) {
                                auto arr = chunked->chunk(k);
                                if (!arr) continue;

                                const int64_t len = arr->length();
                                for (int64_t i = 0; i < len; ++i) {
                                    data[(base + logical_row + i) * cols + c] = get_string_at_any(arr, i, pool);
                                }
                                logical_row += len;
                            }
                            if (logical_row != nRG)
                                throw std::runtime_error("RG row mismatch after chunking (string)");
                        }
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> g(mx);
                    if (!err) err = std::current_exception();
                }
            });
        }

        for (auto &w : workers) w.join();
        if (err) std::rethrow_exception(err);

        res = mat;
        return;
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
            case ValueTypeCode::SI8:   columns[c] = DataObjectFactory::create<DenseMatrix<int8_t  >>(rows,1,true);  break;
            case ValueTypeCode::SI32:  columns[c] = DataObjectFactory::create<DenseMatrix<int32_t >>(rows,1,true);  break;
            case ValueTypeCode::SI64:  columns[c] = DataObjectFactory::create<DenseMatrix<int64_t >>(rows,1,true);  break;
            case ValueTypeCode::UI8:   columns[c] = DataObjectFactory::create<DenseMatrix<uint8_t >>(rows,1,true);  break;
            case ValueTypeCode::UI32:  columns[c] = DataObjectFactory::create<DenseMatrix<uint32_t>>(rows,1,true);  break;
            case ValueTypeCode::UI64:  columns[c] = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows,1,true);  break;
            case ValueTypeCode::F32:   columns[c] = DataObjectFactory::create<DenseMatrix<float   >>(rows,1,true);  break;
            case ValueTypeCode::F64:   columns[c] = DataObjectFactory::create<DenseMatrix<double  >>(rows,1,true);  break;
            default:                   columns[c] = DataObjectFactory::create<DenseMatrix<std::string>>(rows,1,true);
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
                        switch (fmd.schema[c]) {
                            case ValueTypeCode::SI8: {
                                auto arr = std::static_pointer_cast<arrow::Int8Array>(
                                    cast_array_if_needed(tbl->column(c)->chunk(0), arrow::int8(), pool)
                                );
                                auto *dst = static_cast<DenseMatrix<int8_t>*>(columns[c])->getValues();
                                if (arr->null_count() == 0) {
                                    const int8_t *src = arr->raw_values();
                                    std::memcpy(dst + base, src, nRG * sizeof(int8_t));
                                } else {
                                    for (int64_t i=0;i<nRG;++i) dst[base+i] = arr->IsValid(i) ? arr->Value(i) : int8_t(0);
                                }
                                break;
                            }
                            case ValueTypeCode::SI32: {
                                auto arr = std::static_pointer_cast<arrow::Int32Array>(
                                    cast_array_if_needed(tbl->column(c)->chunk(0), arrow::int32(), pool)
                                );
                                auto *dst = static_cast<DenseMatrix<int32_t>*>(columns[c])->getValues();
                                if (arr->null_count() == 0) {
                                    const int32_t *src = arr->raw_values();
                                    std::memcpy(dst + base, src, nRG * sizeof(int32_t));
                                } else {
                                    for (int64_t i=0;i<nRG;++i) dst[base+i] = arr->IsValid(i) ? arr->Value(i) : int32_t(0);
                                }
                                break;
                            }
                            case ValueTypeCode::SI64: {
                                auto arr = std::static_pointer_cast<arrow::Int64Array>(
                                    cast_array_if_needed(tbl->column(c)->chunk(0), arrow::int64(), pool)
                                );
                                auto *dst = static_cast<DenseMatrix<int64_t>*>(columns[c])->getValues();
                                if (arr->null_count() == 0) {
                                    const int64_t *src = arr->raw_values();
                                    std::memcpy(dst + base, src, nRG * sizeof(int64_t));
                                } else {
                                    for (int64_t i=0;i<nRG;++i) dst[base+i] = arr->IsValid(i) ? arr->Value(i) : int64_t(0);
                                }
                                break;
                            }
                            case ValueTypeCode::UI8: {
                                auto arr = std::static_pointer_cast<arrow::UInt8Array>(
                                    cast_array_if_needed(tbl->column(c)->chunk(0), arrow::uint8(), pool)
                                );
                                auto *dst = static_cast<DenseMatrix<uint8_t>*>(columns[c])->getValues();
                                if (arr->null_count() == 0) {
                                    const uint8_t *src = arr->raw_values();
                                    std::memcpy(dst + base, src, nRG * sizeof(uint8_t));
                                } else {
                                    for (int64_t i=0;i<nRG;++i) dst[base+i] = arr->IsValid(i) ? arr->Value(i) : uint8_t(0);
                                }
                                break;
                            }
                            case ValueTypeCode::UI32: {
                                auto arr = std::static_pointer_cast<arrow::UInt32Array>(
                                    cast_array_if_needed(tbl->column(c)->chunk(0), arrow::uint32(), pool)
                                );
                                auto *dst = static_cast<DenseMatrix<uint32_t>*>(columns[c])->getValues();
                                if (arr->null_count() == 0) {
                                    const uint32_t *src = arr->raw_values();
                                    std::memcpy(dst + base, src, nRG * sizeof(uint32_t));
                                } else {
                                    for (int64_t i=0;i<nRG;++i) dst[base+i] = arr->IsValid(i) ? arr->Value(i) : uint32_t(0);
                                }
                                break;
                            }
                            case ValueTypeCode::UI64: {
                                auto arr = std::static_pointer_cast<arrow::UInt64Array>(
                                    cast_array_if_needed(tbl->column(c)->chunk(0), arrow::uint64(), pool)
                                );
                                auto *dst = static_cast<DenseMatrix<uint64_t>*>(columns[c])->getValues();
                                if (arr->null_count() == 0) {
                                    const uint64_t *src = arr->raw_values();
                                    std::memcpy(dst + base, src, nRG * sizeof(uint64_t));
                                } else {
                                    for (int64_t i=0;i<nRG;++i) dst[base+i] = arr->IsValid(i) ? arr->Value(i) : uint64_t(0);
                                }
                                break;
                            }
                            case ValueTypeCode::F32: {
                                auto arr = std::static_pointer_cast<arrow::FloatArray>(
                                    cast_array_if_needed(tbl->column(c)->chunk(0), arrow::float32(), pool)
                                );
                                auto *dst = static_cast<DenseMatrix<float>*>(columns[c])->getValues();
                                if (arr->null_count() == 0) {
                                    const float *src = arr->raw_values();
                                    std::memcpy(dst + base, src, nRG * sizeof(float));
                                } else {
                                    for (int64_t i=0;i<nRG;++i)
                                        dst[base+i] = arr->IsValid(i) ? arr->Value(i) : std::numeric_limits<float>::quiet_NaN();
                                }
                                break;
                            }
                            case ValueTypeCode::F64: {
                                auto arr = std::static_pointer_cast<arrow::DoubleArray>(
                                    cast_array_if_needed(tbl->column(c)->chunk(0), arrow::float64(), pool)
                                );
                                auto *dst = static_cast<DenseMatrix<double>*>(columns[c])->getValues();
                                if (arr->null_count() == 0) {
                                    const double *src = arr->raw_values();
                                    std::memcpy(dst + base, src, nRG * sizeof(double));
                                } else {
                                    for (int64_t i=0;i<nRG;++i)
                                        dst[base+i] = arr->IsValid(i) ? arr->Value(i) : std::numeric_limits<double>::quiet_NaN();
                                }
                                break;
                            }
                            default: {
                                // robust string: cast anything to utf8 (handles DICTIONARY/BINARY/etc.)
                                auto arr = cast_array_if_needed(tbl->column(c)->chunk(0), arrow::utf8(), pool);
                                auto s   = std::static_pointer_cast<arrow::StringArray>(arr);
                                auto *dst = static_cast<DenseMatrix<std::string>*>(columns[c])->getValues();
                                for (int64_t i=0;i<nRG;++i)
                                    dst[base+i] = s->IsValid(i) ? s->GetString(i) : std::string{};
                                break;
                            }
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

// Helper: map ValueTypeCode to a tag
enum class VTTag { F64, F32, SI64, SI32, SI8, UI64, UI32, UI8, STR };

static VTTag tag_from_schema(const FileMetaData &fmd) {
    if(!(fmd.isSingleValueType) || fmd.schema.empty())
        throw std::runtime_error("parquet_write: expected single value type schema");
    switch(fmd.schema[0]) {
        case ValueTypeCode::F64: return VTTag::F64;
        case ValueTypeCode::F32: return VTTag::F32;
        case ValueTypeCode::SI64: return VTTag::SI64;
        case ValueTypeCode::SI32: return VTTag::SI32;
        case ValueTypeCode::SI8:  return VTTag::SI8;
        case ValueTypeCode::UI64: return VTTag::UI64;
        case ValueTypeCode::UI32: return VTTag::UI32;
        case ValueTypeCode::UI8:  return VTTag::UI8;
        case ValueTypeCode::STR:  return VTTag::STR;
        default: throw std::runtime_error("parquet_write: unsupported VT in schema[0]");
    }
}

extern "C" void parquet_write(
    const void *data,
    const FileMetaData &fmd,
    const char *filename,
    const IOOptions &opts,
    DaphneContext * /*ctx*/
) {
    using namespace arrow;

    const int64_t rows = static_cast<int64_t>(fmd.numRows);
    const int64_t cols = static_cast<int64_t>(fmd.numCols);
    if(rows < 0 || cols <= 0)
        throw std::runtime_error("parquet_write: invalid shape");

    // Chunking to avoid Arrow 2GB-per-array limit
    int64_t chunk_rows = 1024 * 1024; // default: 1M rows
    if (auto it = opts.extra.find("row_group_size"); it != opts.extra.end()) {
        try { chunk_rows = std::max<int64_t>(1, std::stoll(it->second)); } catch(...) {}
    }

    // Compression (snappy|gzip|zstd|brotli|lz4|none)
    parquet::Compression::type comp = parquet::Compression::SNAPPY;
    if (auto it = opts.extra.find("compression"); it != opts.extra.end()) {
        std::string s = it->second; std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        if(s=="none"||s=="uncompressed") comp = parquet::Compression::UNCOMPRESSED;
        else if(s=="gzip"||s=="zlib")    comp = parquet::Compression::GZIP;
        else if(s=="zstd")               comp = parquet::Compression::ZSTD;
        else if(s=="brotli")             comp = parquet::Compression::BROTLI;
        else if(s=="lz4")                comp = parquet::Compression::LZ4;
        else if(s=="snappy")             comp = parquet::Compression::SNAPPY;
    }
    auto props = parquet::WriterProperties::Builder().compression(comp)->build();

    // large string option (use 64-bit offsets)
    bool use_large_str = false;
    if (auto it = opts.extra.find("large_strings"); it != opts.extra.end()) {
        std::string s = it->second; std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        use_large_str = (s == "1" || s == "true" || s == "yes");
    }

    // Column names
    auto col_name = [&](int64_t c)->std::string {
        if (c < static_cast<int64_t>(fmd.labels.size()) && !fmd.labels[c].empty())
            return fmd.labels[c];
        return "col_" + std::to_string(c);
    };

    // Build fields + chunked columns
    std::vector<std::shared_ptr<Field>> fields;
    fields.reserve(cols);
    std::vector<std::shared_ptr<ChunkedArray>> chunked_cols;
    chunked_cols.reserve(cols);

    const auto tag = tag_from_schema(fmd);

    auto add_numeric_column = [&](auto *mat, std::shared_ptr<DataType> dtype, auto builder_mk, auto getter) {
        const auto *vals = mat->getValues();
        const int64_t C = static_cast<int64_t>(mat->getNumCols());

        for (int64_t c = 0; c < cols; ++c) {
            std::vector<std::shared_ptr<Array>> chunks;
            chunks.reserve((rows + chunk_rows - 1) / chunk_rows);

            for (int64_t r0 = 0; r0 < rows; r0 += chunk_rows) {
                int64_t r1 = std::min(r0 + chunk_rows, rows);
                auto builder = builder_mk(); // e.g., Int32Builder(...)
                PARQUET_THROW_NOT_OK(builder->Reserve(r1 - r0));

                // Column-major build from row-major buffer
                for (int64_t r = r0; r < r1; ++r) {
                    builder->UnsafeAppend(getter(vals, r, c, C));
                }

                std::shared_ptr<Array> arr;
                PARQUET_THROW_NOT_OK(builder->Finish(&arr));
                chunks.push_back(std::move(arr));
            }

            fields.push_back(field(col_name(c), dtype));
            chunked_cols.push_back(std::make_shared<ChunkedArray>(std::move(chunks), dtype));
        }
    };

    switch(tag) {
        case VTTag::F64: {
            auto *mat = static_cast<const DenseMatrix<double>*>(data);
            add_numeric_column(
                mat, float64(),
                [](){ return std::make_unique<DoubleBuilder>(default_memory_pool()); },
                [](const double *vals, int64_t r, int64_t c, int64_t C){ return vals[r * C + c]; }
            );
            break;
        }
        case VTTag::F32: {
            auto *mat = static_cast<const DenseMatrix<float>*>(data);
            add_numeric_column(
                mat, float32(),
                [](){ return std::make_unique<FloatBuilder>(default_memory_pool()); },
                [](const float *vals, int64_t r, int64_t c, int64_t C){ return vals[r * C + c]; }
            );
            break;
        }
        case VTTag::SI64: {
            auto *mat = static_cast<const DenseMatrix<int64_t>*>(data);
            add_numeric_column(
                mat, int64(),
                [](){ return std::make_unique<Int64Builder>(default_memory_pool()); },
                [](const int64_t *vals, int64_t r, int64_t c, int64_t C){ return vals[r * C + c]; }
            );
            break;
        }
        case VTTag::SI32: {
            auto *mat = static_cast<const DenseMatrix<int32_t>*>(data);
            add_numeric_column(
                mat, int32(),
                [](){ return std::make_unique<Int32Builder>(default_memory_pool()); },
                [](const int32_t *vals, int64_t r, int64_t c, int64_t C){ return vals[r * C + c]; }
            );
            break;
        }
        case VTTag::SI8: {
            auto *mat = static_cast<const DenseMatrix<int8_t>*>(data);
            add_numeric_column(
                mat, int8(),
                [](){ return std::make_unique<Int8Builder>(default_memory_pool()); },
                [](const int8_t *vals, int64_t r, int64_t c, int64_t C){ return vals[r * C + c]; }
            );
            break;
        }
        case VTTag::UI64: {
            auto *mat = static_cast<const DenseMatrix<uint64_t>*>(data);
            add_numeric_column(
                mat, uint64(),
                [](){ return std::make_unique<UInt64Builder>(default_memory_pool()); },
                [](const uint64_t *vals, int64_t r, int64_t c, int64_t C){ return vals[r * C + c]; }
            );
            break;
        }
        case VTTag::UI32: {
            auto *mat = static_cast<const DenseMatrix<uint32_t>*>(data);
            add_numeric_column(
                mat, uint32(),
                [](){ return std::make_unique<UInt32Builder>(default_memory_pool()); },
                [](const uint32_t *vals, int64_t r, int64_t c, int64_t C){ return vals[r * C + c]; }
            );
            break;
        }
        case VTTag::UI8: {
            auto *mat = static_cast<const DenseMatrix<uint8_t>*>(data);
            add_numeric_column(
                mat, uint8(),
                [](){ return std::make_unique<UInt8Builder>(default_memory_pool()); },
                [](const uint8_t *vals, int64_t r, int64_t c, int64_t C){ return vals[r * C + c]; }
            );
            break;
        }
        case VTTag::STR: {
            // strings: choose utf8 or large_utf8
            auto *mat = static_cast<const DenseMatrix<std::string>*>(data);
            const std::string *vals = mat->getValues();

            for (int64_t c = 0; c < cols; ++c) {
                std::vector<std::shared_ptr<Array>> chunks;
                chunks.reserve((rows + chunk_rows - 1) / chunk_rows);

                if (!use_large_str) {
                    for (int64_t r0 = 0; r0 < rows; r0 += chunk_rows) {
                        int64_t r1 = std::min(r0 + chunk_rows, rows);
                        StringBuilder b(default_memory_pool());
                        for (int64_t r = r0; r < r1; ++r)
                            PARQUET_THROW_NOT_OK(b.Append(vals[r * cols + c]));
                        std::shared_ptr<Array> arr;
                        PARQUET_THROW_NOT_OK(b.Finish(&arr));
                        chunks.push_back(std::move(arr));
                    }
                    fields.push_back(field(col_name(c), utf8()));
                    chunked_cols.push_back(std::make_shared<ChunkedArray>(std::move(chunks), utf8()));
                }
                else {
                    for (int64_t r0 = 0; r0 < rows; r0 += chunk_rows) {
                        int64_t r1 = std::min(r0 + chunk_rows, rows);
                        LargeStringBuilder b(default_memory_pool()); // 64-bit offsets
                        for (int64_t r = r0; r < r1; ++r)
                            PARQUET_THROW_NOT_OK(b.Append(vals[r * cols + c]));
                        std::shared_ptr<Array> arr;
                        PARQUET_THROW_NOT_OK(b.Finish(&arr));
                        chunks.push_back(std::move(arr));
                    }
                    fields.push_back(field(col_name(c), large_utf8()));
                    chunked_cols.push_back(std::make_shared<ChunkedArray>(std::move(chunks), large_utf8()));
                }
            }
            break;
        }
    }

    auto sch = schema(fields);
    // Build table from chunked columns
    std::shared_ptr<arrow::Table> tbl = arrow::Table::Make(sch, chunked_cols, rows);
    auto out_res = arrow::io::FileOutputStream::Open(filename);
    if (!out_res.ok())
        throw std::runtime_error("parquet_write: cannot open file: " + out_res.status().ToString());
    std::shared_ptr<arrow::io::OutputStream> out = *out_res;

    // Write with Parquet
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*tbl, default_memory_pool(), out, static_cast<int64_t>(chunk_rows), props));
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