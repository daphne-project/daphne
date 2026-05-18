// gen_data.cpp
// Build: see Makefile below
// Usage: ./gen_data --outdir /path --target-size-mb 1500 --cols 8
//
// Outputs (in --outdir):
//   - gen.csv       (row-safe; no partial lines) + gen.csv.meta
//   - gen.mtx       (Matrix Market; line-safe; correct nnz) + gen.mtx.meta
//   - gen.parquet   (Arrow/Parquet C++; row groups; no truncation) + gen.parquet.meta
//
// The *.meta files have the shape:
// {
//   "numRows": <int>,
//   "numCols": <int>,
//   "valueType": "<str|f64|...>"
// }
// Here, all generated numeric data are double precision → "f64".
// For Matrix Market we use n×n (valueType "f64" via 'real' banner).

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

// ------------------------------------------------------------
static inline long long fsize_bytes(const std::string& p) {
    struct stat st{};
    if (stat(p.c_str(), &st) == 0) return static_cast<long long>(st.st_size);
    return 0LL;
}
static inline long long mb_to_bytes(long long mb) { return mb * 1024LL * 1024LL; }

// ------------------------------------------------------------
// Meta writer: writes <path>.meta with required JSON
static inline void write_meta(const std::string& data_path, long long numRows, long long numCols, const std::string& valueType) {
    std::ofstream meta(data_path + ".meta", std::ios::out | std::ios::trunc);
    if(!meta) throw std::runtime_error("Cannot open meta for write: " + data_path + ".meta");
    meta << "{\n"
         << "  \"numRows\": " << numRows << ",\n"
         << "  \"numCols\": " << numCols << ",\n"
         << "  \"valueType\": \"" << valueType << "\"\n"
         << "}\n";
    meta.close();
}

// ------------------------------------------------------------
// CSV generator (row-safe, portable: never writes a partial row)
// Returns number of data rows written (excludes header)
long long gen_csv_linesafe(const std::string& path, long long target_mb, int cols, uint32_t seed=42) {
    const long long target = mb_to_bytes(target_mb);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1e6);

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if(!out) throw std::runtime_error("Cannot open CSV for write: " + path);

    // header
    std::ostringstream hdr;
    for(int c=0;c<cols;c++){
        hdr << "c" << (c+1);
        if(c+1<cols) hdr << ",";
    }
    hdr << "\n";
    const std::string header = hdr.str();
    if (static_cast<long long>(header.size()) > target) {
        throw std::runtime_error("Target size smaller than header");
    }
    out << header;

    // rows
    long long rows_written = 0;
    while(true){
        std::ostringstream row;
        for(int c=0;c<cols;c++){
            row << std::fixed << std::setprecision(6) << dist(rng);
            if(c+1<cols) row << ",";
        }
        row << "\n";
        const std::string s = row.str();

        // check BEFORE writing (so we never need to truncate)
        auto pos = out.tellp();
        long long next = static_cast<long long>(pos) + static_cast<long long>(s.size());
        if(next > target) break;

        out << s;
        if(!out.good()) throw std::runtime_error("CSV write failed");
        ++rows_written;
    }
    out.close();
    return rows_written;
}


// ------------------------------------------------------------
// Matrix Market generator (line-safe, correct nnz)
// Returns n (matrix dimension; rows=cols=n)
// We write the entries to a temp data file first to avoid keeping 1–2 GB in RAM,
// then stitch the final file with a correct header.
int gen_mm_linesafe(const std::string& out_path, long long target_mb, int n=200000, uint32_t seed=42) {
    const long long target = mb_to_bytes(target_mb);
    const std::string tmp_data = out_path + ".data.tmp";

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni_idx(1, n);
    std::uniform_real_distribution<double> uni_val(-1.0, 1.0);

    std::ofstream data(tmp_data, std::ios::out | std::ios::trunc);
    if(!data) throw std::runtime_error("Cannot open temp MM data: " + tmp_data);

    long long bytes = 0;
    long long nnz = 0;
    while(true){
        int i = uni_idx(rng);
        int j = uni_idx(rng);
        double v = std::round(uni_val(rng)*1e6)/1e6;

        std::ostringstream line;
        line << i << " " << j << " " << std::fixed << std::setprecision(6) << v << "\n";
        auto s = line.str();
        if(bytes + static_cast<long long>(s.size()) > target) break;

        data << s;
        if(!data.good()) throw std::runtime_error("MM write failed");
        bytes += static_cast<long long>(s.size());
        ++nnz;
    }
    data.close();

    // Stitch final file with correct header
    std::ifstream in(tmp_data, std::ios::in | std::ios::binary);
    if(!in) throw std::runtime_error("Cannot reopen temp MM data");
    std::ofstream out(out_path, std::ios::out | std::ios::trunc | std::ios::binary);
    if(!out) throw std::runtime_error("Cannot open MM output: " + out_path);

    out << "%%MatrixMarket matrix coordinate real general\n%\n";
    out << n << " " << n << " " << nnz << "\n";
    out << in.rdbuf();
    out.close();
    in.close();
    std::remove(tmp_data.c_str());
    return n;
}

// ------------------------------------------------------------
// Parquet generator (Arrow C++; write row groups until size >= target)
// Returns total row count written
// NOTE: No appending. We keep the writer open and check sink->Tell() after each row group.
long long gen_parquet_arrow(const std::string& path, long long target_mb, int cols,
                            int64_t batch_rows = 250000, uint32_t seed = 42) {
    const int64_t target = mb_to_bytes(target_mb);

    // Schema: cols of float64
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(cols);
    for (int c = 0; c < cols; ++c) {
        fields.emplace_back(arrow::field("c" + std::to_string(c + 1), arrow::float64()));
    }
    auto schema = std::make_shared<arrow::Schema>(fields);

    // Output sink and writer (single open)
    PARQUET_ASSIGN_OR_THROW(auto sink, arrow::io::FileOutputStream::Open(path));

    parquet::WriterProperties::Builder wpb;
    wpb.compression(parquet::Compression::ZSTD);   // adjust if your toolchain lacks zstd
    auto writer_props = wpb.build();

    std::unique_ptr<parquet::arrow::FileWriter> writer;
    PARQUET_THROW_NOT_OK(parquet::arrow::FileWriter::Open(
        *schema, arrow::default_memory_pool(), sink, writer_props, &writer));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1e6);

    long long total_rows = 0;
    while (true) {
        // Build one row group
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        arrays.reserve(cols);
        for (int c = 0; c < cols; ++c) {
            arrow::DoubleBuilder b;
            PARQUET_THROW_NOT_OK(b.Reserve(batch_rows));
            for (int64_t r = 0; r < batch_rows; ++r) {
                double v = std::round(dist(rng) * 1e6) / 1e6;
                b.UnsafeAppend(v);
            }
            std::shared_ptr<arrow::Array> arr;
            PARQUET_THROW_NOT_OK(b.Finish(&arr));
            arrays.emplace_back(std::move(arr));
        }
        auto table = arrow::Table::Make(schema, arrays, batch_rows);

        // Write the row group
        PARQUET_THROW_NOT_OK(writer->WriteTable(*table, batch_rows));
        total_rows += batch_rows;

        // Check current file size without closing
        PARQUET_ASSIGN_OR_THROW(int64_t pos, sink->Tell());
        if (pos >= target) break;
    }

    // Clean close
    PARQUET_THROW_NOT_OK(writer->Close());
    PARQUET_THROW_NOT_OK(sink->Close());

    return total_rows;
}


// ------------------------------------------------------------
int main(int argc, char** argv){
    // Minimal CLI
    std::string outdir;
    long long target_mb = 1500;
    int cols = 8;

    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a=="--outdir" && i+1<argc) outdir = argv[++i];
        else if(a=="--target-size-mb" && i+1<argc) target_mb = std::stoll(argv[++i]);
        else if(a=="--cols" && i+1<argc) cols = std::stoi(argv[++i]);
        else if(a=="-h" || a=="--help") {
            std::cout << "Usage: " << argv[0] << " --outdir DIR [--target-size-mb N] [--cols M]\n";
            return 0;
        }
    }
    if(outdir.empty()){
        std::cerr << "[error] --outdir is required\n";
        return 1;
    }
    // ensure dir exists
    std::string mkdir_cmd = "mkdir -p \"" + outdir + "\"";
    if (std::system(mkdir_cmd.c_str()) != 0) {
        std::cerr << "[error] failed to create outdir: " << outdir << "\n";
        return 1;
    }

    const std::string csv_path = outdir + "/gen.csv";
    const std::string mm_path  = outdir + "/gen.mtx";
    const std::string pq_path  = outdir + "/gen.parquet";

    try {
        /*std::cout << "[gen] Parquet -> " << pq_path << " (~" << target_mb << " MB)\n";
        long long pq_rows = gen_parquet_arrow(pq_path, target_mb, cols);
        write_meta(pq_path, pq_rows, cols, "f64");*/

        std::cout << "[gen] CSV -> " << csv_path << " (~" << target_mb << " MB)\n";
        long long csv_rows = gen_csv_linesafe(csv_path, target_mb, cols);
        // rows exclude header (data rows only)
        write_meta(csv_path, csv_rows, cols, "f64");
        /*
        std::cout << "[gen] MM  -> " << mm_path  << " (~" << target_mb << " MB)\n";
        int mm_n = gen_mm_linesafe(mm_path, target_mb);
        write_meta(mm_path, mm_n, mm_n, "f64");*/
        

        std::cout << "[done]\n" << csv_path << "\n" << pq_path << "\n" << mm_path << "\n";
    } catch (const std::exception& ex){
        std::cerr << "[error] " << ex.what() << "\n";
        return 2;
    }
    return 0;
}
