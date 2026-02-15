// gen_meta.cpp
// Usage: ./gen_meta file1.csv file2.parquet file3.mtx ...
// Writes <path>.meta with:
// {
//   "numRows": N,
//   "numCols": M,
//   "valueType": "<f64|str|si32>"
// }

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/metadata.h>

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

static inline bool file_exists(const std::string& p) {
    struct stat st{};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}
static inline std::string to_lower(std::string s) {
    for (auto &ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}
static inline std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e-1]))) --e;
    return s.substr(b, e-b);
}
static bool is_numeric(const std::string& s_in) {
    std::string s = trim(s_in);
    if (s.empty()) return false;
    std::string sl = to_lower(s);
    if (sl == "nan" || sl == "+nan" || sl == "-nan" ||
        sl == "inf" || sl == "+inf" || sl == "-inf") return false;
    char* endp = nullptr;
    errno = 0;
    std::strtod(s.c_str(), &endp);
    if (endp == s.c_str() || errno == ERANGE) return false;
    while (endp && *endp) {
        if (!std::isspace(static_cast<unsigned char>(*endp))) return false;
        ++endp;
    }
    return true;
}

static long long count_lines_fast(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return 0;
    const size_t BUFSZ = 1 << 20;
    std::vector<char> buf(BUFSZ);
    long long cnt = 0;
    while (in) {
        in.read(buf.data(), buf.size());
        std::streamsize g = in.gcount();
        for (std::streamsize i = 0; i < g; ++i) if (buf[i] == '\n') ++cnt;
    }
    return cnt;
}

// ---------------- Parquet (Arrow) ----------------
static bool parquet_meta(const std::string& path, long long& rows, long long& cols, std::string& vtype) {
    try {
        auto infile_res = arrow::io::ReadableFile::Open(path);
        if (!infile_res.ok()) return false;
        std::shared_ptr<arrow::io::ReadableFile> infile = *infile_res;

        std::unique_ptr<parquet::ParquetFileReader> pq_reader = parquet::ParquetFileReader::Open(infile);
        std::shared_ptr<parquet::FileMetaData> md = pq_reader->metadata();
        rows = md->num_rows();
        cols = md->num_columns();

        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        auto st = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &arrow_reader);
        bool all_numeric = false;
        if (st.ok() && arrow_reader) {
            std::shared_ptr<arrow::Schema> schema;
            if (arrow_reader->GetSchema(&schema).ok() && schema) {
                all_numeric = !schema->fields().empty();
                for (const auto& f : schema->fields()) {
                    using T = arrow::Type::type;
                    auto id = f->type()->id();
                    bool is_num =
                        id == T::INT8 || id == T::INT16 || id == T::INT32 || id == T::INT64 ||
                        id == T::UINT8 || id == T::UINT16 || id == T::UINT32 || id == T::UINT64 ||
                        id == T::FLOAT || id == T::DOUBLE || id == T::HALF_FLOAT ||
                        id == T::DECIMAL128 || id == T::DECIMAL256;
                    if (!is_num) { all_numeric = false; break; }
                }
            }
        }
        vtype = (cols > 0 && rows >= 0 && all_numeric) ? "f64" : "str";
        return true;
    } catch (...) {
        return false;
    }
}

// -------------- CSV (quote-free, header present) --------------
static bool csv_meta(const std::string& path, long long& rows, long long& cols, std::string& vtype) {
    // Count physical lines
    long long lines = count_lines_fast(path);
    if (lines <= 0) { rows = 0; cols = 0; vtype = "str"; return true; }

    std::ifstream in(path);
    if (!in) return false;

    // Header (first line)
    std::string header;
    if (!std::getline(in, header)) { rows = 0; cols = 0; vtype = "str"; return true; }

    // Split header by comma (NO quoting expected)
    cols = 0;
    {
        std::stringstream ss(header);
        std::string tok;
        while (std::getline(ss, tok, ',')) ++cols;
    }
    if (cols == 0) { rows = 0; vtype = "str"; return true; }

    // Rows = total lines - 1 (header)
    rows = (lines > 0) ? (lines - 1) : 0;

    // Sample next up to N lines to infer numeric-ness; enforce width = cols
    const long long MAX_SAMPLE = 2000;
    bool all_numeric = true;
    std::string line;
    long long sampled = 0;

    while (sampled < MAX_SAMPLE && std::getline(in, line)) {
        if (line.empty()) { ++sampled; continue; }
        // split (quote-free)
        std::vector<std::string> fields;
        fields.reserve(static_cast<size_t>(cols));
        {
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ',')) fields.emplace_back(std::move(tok));
        }
        if (static_cast<long long>(fields.size()) != cols) { all_numeric = false; break; }
        for (const auto& f : fields) {
            if (!is_numeric(f)) { all_numeric = false; break; }
        }
        ++sampled;
        if (!all_numeric) break;
    }

    vtype = all_numeric ? "f64" : "str";
    return true;
}

// -------------- Matrix Market (.mtx) --------------
static bool mtx_meta(const std::string& path, long long& rows, long long& cols, std::string& vtype) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;

    if (!std::getline(in, line)) return false;
    std::string hl = to_lower(line);
    if (hl.find("matrixmarket") == std::string::npos) return false;

    if (hl.find(" real ") != std::string::npos) vtype = "f64";
    else if (hl.find(" integer ") != std::string::npos) vtype = "si32";
    else vtype = "str";

    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '%') break;
    }
    std::istringstream iss(line);
    long long r=0, c=0;
    iss >> r >> c;
    rows = r; cols = c;
    return true;
}

static void write_meta_file(const std::string& path, long long rows, long long cols, const std::string& vtype) {
    std::ofstream out(path + ".meta", std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "[meta][error] cannot open " << path << ".meta for write\n";
        return;
    }
    out << "{\n"
        << "  \"numRows\": " << rows << ",\n"
        << "  \"numCols\": " << cols << ",\n"
        << "  \"valueType\": \"" << vtype << "\"\n"
        << "}\n";
    out.close();
    std::cout << "[meta] wrote " << path << ".meta\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file1> [file2 ...]\n";
        return 1;
    }

    int rc_all = 0;
    for (int i = 1; i < argc; ++i) {
        std::string path = argv[i];
        if (!file_exists(path)) {
            std::cerr << "[meta][warn] skip missing: " << path << "\n";
            rc_all = 2; continue;
        }
        std::string lower = to_lower(path);
        long long rows = 0, cols = 0;
        std::string vtype = "str";
        bool ok = false;

        if (lower.size() >= 8 && lower.substr(lower.size() - 8) == ".parquet") {
            ok = parquet_meta(path, rows, cols, vtype);
        } else if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".csv") {
            ok = csv_meta(path, rows, cols, vtype);  // header-aware, quote-free
        } else if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".mtx") {
            ok = mtx_meta(path, rows, cols, vtype);
        } else {
            std::cerr << "[meta][warn] unknown extension, skipping: " << path << "\n";
            rc_all = 3; continue;
        }

        if (!ok) {
            std::cerr << "[meta][error] failed to derive meta for: " << path << "\n";
            rc_all = 4; continue;
        }
        write_meta_file(path, rows, cols, vtype);
    }
    return rc_all;
}
