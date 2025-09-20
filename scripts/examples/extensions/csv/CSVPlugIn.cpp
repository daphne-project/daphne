// CsvIO.cpp — fast CSV plugin with great single-thread performance and scalable multi-thread parsing.
// Assumptions: no quoted fields; fixed columns; '\n' line ends (tolerates '\r\n').
// Type rule: F64 -> double, UI64 -> uint64_t, else strings.
// Options (IOOptions.extra): hasHeader=true|false, delimiter=",", threads="N" (>=1).

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <system_error>
#include <charconv> // from_chars for uint64_t

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef DAPHNE_USE_FAST_FLOAT
  #include <fast_float/fast_float.h> // https://github.com/fastfloat/fast_float
#endif

#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/Structure.h"
#include "runtime/local/io/FileMetaData.h"
#include "runtime/local/io/FileIORegistry.h"
#include "runtime/local/datastructures/Frame.h"
#include "runtime/local/kernels/CreateFrame.h"
//======================= file mapping =======================

struct MappedFile {
    const char* data = nullptr;
    size_t size = 0;
    int fd = -1;
    void* map = MAP_FAILED;
    std::vector<char> fallback; // used if mmap fails

    static MappedFile open(const char* path) {
        MappedFile m;
        m.fd = ::open(path, O_RDONLY);
        if (m.fd < 0) {
            throw std::system_error(errno, std::generic_category(), std::string("open: ") + path);
        }
        struct stat st{};
        if (fstat(m.fd, &st) != 0) {
            int e = errno;
            ::close(m.fd);
            throw std::system_error(e, std::generic_category(), "fstat");
        }
        m.size = static_cast<size_t>(st.st_size);
        if (m.size > 0) {
            m.map = ::mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, m.fd, 0);
            if (m.map != MAP_FAILED) {
                m.data = static_cast<const char*>(m.map);
                // Hint the kernel we're going to scan forward once.
                #ifdef POSIX_MADV_SEQUENTIAL
                ::posix_madvise(m.map, m.size, POSIX_MADV_SEQUENTIAL);
                #endif
                return m;
            }
            // Fallback: read whole file
            m.fallback.resize(m.size);
            size_t off = 0;
            while (off < m.size) {
                ssize_t r = ::read(m.fd, m.fallback.data() + off, m.size - off);
                if (r < 0) {
                    int e = errno; ::close(m.fd);
                    throw std::system_error(e, std::generic_category(), "read");
                }
                if (r == 0) break;
                off += static_cast<size_t>(r);
            }
            m.data = m.fallback.data();
        }
        return m;
    }
    ~MappedFile() {
        if (map != MAP_FAILED) ::munmap(map, size);
        if (fd >= 0) ::close(fd);
    }
};

//======================= small helpers =======================

static inline int32_t parse_i32_token(const char* b, const char* e) {
    int32_t out = 0;
    auto res = std::from_chars(b, e, out, 10);
    if (res.ec == std::errc()) return out;
    char* ep = nullptr; return static_cast<int32_t>(std::strtol(b, &ep, 10));
}

static inline size_t header_end(const char* p, size_t n) {
    size_t i = 0;
    while (i < n && p[i] != '\n') ++i;
    if (i < n) ++i; // skip '\n'
    return std::min(i, n);
}

static inline void trim_token(const char*& b, const char*& e) {
    while (b < e && (*b == ' ' || *b == '\t')) ++b;
    while (e > b && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) --e;
}

static inline const char* find_next_sep(const char* p, const char* end, char delim, char& which) {
    // returns pointer to next delim ('d') or newline ('n'); sets which accordingly; end => which=0
    while (p < end) {
        char c = *p;
        if (c == delim) { which = 'd'; return p; }
        if (c == '\n')  { which = 'n'; return p; }
        ++p;
    }
    which = 0;
    return end;
}

// Build an index of line starts (byte offsets) for exactly `rows` rows, starting at `dataStart`.
// Produces vector of size rows+1 with sentinel=end-of-last-line (possibly file end).
static inline void build_line_index(const char* base, size_t n, size_t dataStart,
                                    size_t rows, std::vector<size_t>& lineStart) {
    lineStart.clear();
    lineStart.reserve(rows + 1);
    size_t i = dataStart;
    lineStart.push_back(i); // first row starts here
    while (lineStart.size() < rows && i < n) {
        if (base[i] == '\n') {
            size_t next = i + 1;
            if (next < n) lineStart.push_back(next);
        }
        ++i;
    }
    // For the final sentinel, walk to end-of-last-line (or EOF)
    if (i < n) {
        while (i < n && base[i] != '\n') ++i;
    }
    size_t sentinel = (i < n) ? (i) : n; // sentinel points to '\n' or EOF
    lineStart.push_back(sentinel);
    if (lineStart.size() != rows + 1)
        throw std::runtime_error("CSV: could not index requested number of rows");
}

//======================= numeric token parsing =======================

static inline uint64_t parse_u64_token(const char* b, const char* e) {
    uint64_t out = 0;
    auto res = std::from_chars(b, e, out, 10);
    if (res.ec != std::errc()) {
        // fallback (tolerate +spaces- trimming)
        char* ep = nullptr;
        out = static_cast<uint64_t>(std::strtoull(b, &ep, 10));
    }
    return out;
}

static inline double parse_f64_token(const char* b, const char* e) {
#ifdef DAPHNE_USE_FAST_FLOAT
    double v = 0.0;
    auto [ptr, ec] = fast_float::from_chars(b, e, v);
    if (ec == std::errc()) return v;
    // fallback:
    char* ep = nullptr;
    return std::strtod(b, &ep);
#else
    char* ep = nullptr;
    return std::strtod(b, &ep);
#endif
}

//======================= single-thread fast paths =======================

template<typename NumT>
static void parse_numeric_single(const char* p, const char* end,
                                 size_t rows, size_t cols, char delim,
                                 NumT* out)
{
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c + 1 < cols; ++c) {
            char which = 0;
            const char* sep = find_next_sep(p, end, delim, which);
            if (which != 'd')
                throw std::runtime_error("CSV: not enough columns (single, delim expected)");
            const char* b = p; const char* q = sep; trim_token(b, q);
            if constexpr (std::is_same_v<NumT, double>) out[r*cols + c] = parse_f64_token(b, q);
            else                                        out[r*cols + c] = static_cast<NumT>(parse_u64_token(b, q));
            p = sep + 1;
        }
        char which = 0;
        const char* sep = find_next_sep(p, end, delim, which);
        if (which == 'd')
            throw std::runtime_error("CSV: too many columns (single)");
        const char* b = p; const char* q = sep; trim_token(b, q);
        if constexpr (std::is_same_v<NumT, double>) out[r*cols + (cols-1)] = parse_f64_token(b, q);
        else                                        out[r*cols + (cols-1)] = static_cast<NumT>(parse_u64_token(b, q));
        p = (which == 'n' && sep < end) ? sep + 1 : sep;
    }
}

static void parse_string_single(const char* p, const char* end,
                                size_t rows, size_t cols, char delim,
                                std::string* out)
{
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c + 1 < cols; ++c) {
            char which = 0;
            const char* sep = find_next_sep(p, end, delim, which);
            if (which != 'd')
                throw std::runtime_error("CSV: not enough columns (str, delim expected)");
            const char* b = p; const char* q = sep; trim_token(b, q);
            out[r*cols + c].assign(b, q);
            p = sep + 1;
        }
        char which = 0;
        const char* sep = find_next_sep(p, end, delim, which);
        if (which == 'd')
            throw std::runtime_error("CSV: too many columns (str)");
        const char* b = p; const char* q = sep; trim_token(b, q);
        out[r*cols + (cols-1)].assign(b, q);
        p = (which == 'n' && sep < end) ? sep + 1 : sep;
    }
}

//======================= parallel by row ranges =======================

template<typename NumT>
static void parse_numeric_rows_parallel(const char* base, const std::vector<size_t>& rowStarts,
                                        size_t rows, size_t cols, char delim,
                                        NumT* out, size_t threads)
{
    threads = std::max<size_t>(1, std::min(threads, rows ? rows : 1));
    auto worker = [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const char* p   = base + rowStarts[r];
            const char* end = base + rowStarts[r + 1];
            for (size_t c = 0; c + 1 < cols; ++c) {
                char which = 0;
                const char* sep = find_next_sep(p, end, delim, which);
                if (which != 'd') throw std::runtime_error("CSV: not enough columns (parallel)");
                const char* b = p; const char* q = sep; trim_token(b, q);
                if constexpr (std::is_same_v<NumT, double>) out[r*cols + c] = parse_f64_token(b, q);
                else                                        out[r*cols + c] = static_cast<NumT>(parse_u64_token(b, q));
                p = sep + 1;
            }
            char which = 0;
            const char* sep = find_next_sep(p, end, delim, which);
            if (which == 'd') throw std::runtime_error("CSV: too many columns (parallel)");
            const char* b = p; const char* q = sep; trim_token(b, q);
            if constexpr (std::is_same_v<NumT, double>) out[r*cols + (cols-1)] = parse_f64_token(b, q);
            else                                        out[r*cols + (cols-1)] = static_cast<NumT>(parse_u64_token(b, q));
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(threads);
    const size_t chunk = (rows + threads - 1) / threads;
    size_t r0 = 0;
    for (size_t t = 0; t < threads && r0 < rows; ++t) {
        size_t r1 = std::min(rows, r0 + chunk);
        pool.emplace_back(worker, r0, r1);
        r0 = r1;
    }
    for (auto& th : pool) th.join();
}

//======================= plugin API =======================

extern "C" {

void csv_read_frame(
    Frame *&res,
    const FileMetaData &fmd,
    const char *filename,
    IOOptions &opts,
    DaphneContext *ctx
) {
    // 0) Preconditions
    const size_t rows = fmd.numRows;
    const size_t cols = fmd.numCols;
    if (rows == 0 || cols == 0)
        throw std::runtime_error("CSV(Frame): fmd.numRows/numCols must be provided");
    if (fmd.schema.size() < cols)
        throw std::runtime_error("CSV(Frame): fmd.schema must have at least numCols entries");
    if (fmd.labels.size() < cols)
        throw std::runtime_error("CSV(Frame): fmd.labels must have at least numCols entries");

    // 1) Options
    bool hasHeader = true;
    char delim = ',';
    if (auto it = opts.extra.find("hasHeader"); it != opts.extra.end())
        hasHeader = (it->second == "true" || it->second == "1");
    if (auto it = opts.extra.find("delimiter"); it != opts.extra.end()) {
        if (it->second.size() != 1) throw std::runtime_error("CSV(Frame): delimiter must be one character");
        delim = it->second[0];
    }

    // 2) mmap file + data region
    MappedFile mf = MappedFile::open(filename);
    if (mf.size == 0) throw std::runtime_error("CSV(Frame): empty file");
    const char* base = mf.data;
    const size_t n   = mf.size;
    const size_t dataStart = hasHeader ? header_end(base, n) : 0;
    if (dataStart >= n)
        throw std::runtime_error("CSV(Frame): no data region found");
    const char* p   = base + dataStart;
    const char* end = base + n;

    // 3) Labels (pointers must remain valid until createFrame copies them)
    std::vector<const char*> colLabels(cols);
    for (size_t c = 0; c < cols; ++c)
        colLabels[c] = fmd.labels[c].c_str();

    // 4) Allocate one rows×1 matrix per column based on fmd.schema[c]
    std::vector<Structure*> columns(cols, nullptr);
    std::vector<int32_t*>   colI32(cols, nullptr);
    std::vector<uint64_t*>  colU64(cols, nullptr);
    std::vector<double*>    colF64(cols, nullptr);
    std::vector<std::string*> colSTR(cols, nullptr);

    for (size_t c = 0; c < cols; ++c) {
        switch (fmd.schema[c]) {
            case ValueTypeCode::SI32: {
                auto* m = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, 1, false);
                columns[c] = m; colI32[c] = m->getValues(); break;
            }
            case ValueTypeCode::UI64: {
                auto* m = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, 1, false);
                columns[c] = m; colU64[c] = m->getValues(); break;
            }
            case ValueTypeCode::F64: {
                auto* m = DataObjectFactory::create<DenseMatrix<double>>(rows, 1, false);
                columns[c] = m; colF64[c] = m->getValues(); break;
            }
            default: {
                auto* m = DataObjectFactory::create<DenseMatrix<std::string>>(rows, 1, false);
                columns[c] = m; colSTR[c] = m->getValues(); break;
            }
        }
    }

    // 5) Single-pass parse: row by row, column by column
    for (size_t r = 0; r < rows; ++r) {
        // columns 0..cols-2 (must end at delimiter)
        for (size_t c = 0; c + 1 < cols; ++c) {
            char which = 0;
            const char* sep = find_next_sep(p, end, delim, which);
            if (which != 'd')
                throw std::runtime_error("CSV(Frame): not enough columns while expecting delimiter");
            const char* b = p; const char* q = sep; trim_token(b, q);
            switch (fmd.schema[c]) {
                case ValueTypeCode::SI32: colI32[c][r] = parse_i32_token(b, q); break;
                case ValueTypeCode::UI64: colU64[c][r] = parse_u64_token(b, q); break;
                case ValueTypeCode::F64:  colF64[c][r] = parse_f64_token(b, q);  break;
                default:                  colSTR[c][r].assign(b, q);             break;
            }
            p = sep + 1; // move after delimiter
        }
        // last column (must end at newline or EOF)
        char which = 0;
        const char* sep = find_next_sep(p, end, delim, which);
        if (which == 'd')
            throw std::runtime_error("CSV(Frame): too many columns on row");
        const char* b = p; const char* q = sep; trim_token(b, q);
        const size_t cLast = cols - 1;
        switch (fmd.schema[cLast]) {
            case ValueTypeCode::SI32: colI32[cLast][r] = parse_i32_token(b, q); break;
            case ValueTypeCode::UI64: colU64[cLast][r] = parse_u64_token(b, q); break;
            case ValueTypeCode::F64:  colF64[cLast][r] = parse_f64_token(b, q);  break;
            default:                  colSTR[cLast][r].assign(b, q);             break;
        }
        p = (which == 'n' && sep < end) ? sep + 1 : sep; // next row
    }

    // 6) Assemble Frame
    createFrame(
        res,
        columns.data(),
        cols,
        colLabels.data(),
        cols,
        ctx
    );
}

void csv_read(Structure*& res,
              const FileMetaData& fmd,
              const char* filename,
              IOOptions& opts,
              DaphneContext* /*ctx*/)
{
    // --- options ---
    bool hasHeader = true;
    char delim = ',';
    size_t threads = 1;

    if (auto it = opts.extra.find("hasHeader"); it != opts.extra.end())
        hasHeader = (it->second == "true" || it->second == "1");
    if (auto it = opts.extra.find("delimiter"); it != opts.extra.end()) {
        if (it->second.size() != 1) throw std::runtime_error("CSV: delimiter must be one character");
        delim = it->second[0];
    }
    if (auto it = opts.extra.find("threads"); it != opts.extra.end()) {
        threads = std::max<size_t>(1, std::stoul(it->second));
    }

    // --- type rule (built-in parity) ---
    ValueTypeCode vtc = ValueTypeCode::STR;
    if (fmd.isSingleValueType && !fmd.schema.empty()) {
        if (fmd.schema[0] == ValueTypeCode::F64)       vtc = ValueTypeCode::F64;
        else if (fmd.schema[0] == ValueTypeCode::UI64) vtc = ValueTypeCode::UI64;
        else if (fmd.schema[0] == ValueTypeCode::SI32) vtc = ValueTypeCode::SI32;  // <-- add this
        else                                           vtc = ValueTypeCode::STR;
    }

    // --- rows/cols are guaranteed by you ---
    const size_t rows = fmd.numRows;
    const size_t cols = fmd.numCols;
    if (rows == 0 || cols == 0)
        throw std::runtime_error("CSV: numRows/numCols must be provided");

    // --- map file once ---
    MappedFile mf = MappedFile::open(filename);
    if (mf.size == 0) throw std::runtime_error("CSV: empty file");
    const char* base = mf.data;
    const size_t n   = mf.size;

    // --- compute start of data region ---
    const size_t dataStart = hasHeader ? header_end(base, n) : 0;
    if (dataStart >= n) throw std::runtime_error("CSV: no data region found");

    // --- ultra-fast single-thread path (no extra indexing) ---
    if (threads == 1) {
        const char* p = base + dataStart;
        const char* end = base + n;
        if (vtc == ValueTypeCode::F64) {
            auto* mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
            parse_numeric_single<double>(p, end, rows, cols, delim, mat->getValues());
            res = mat; return;
        } else if (vtc == ValueTypeCode::UI64) {
            auto* mat = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false);
            parse_numeric_single<uint64_t>(p, end, rows, cols, delim, mat->getValues());
            res = mat; return;
        } else if (vtc == ValueTypeCode::SI32) {
            auto* mat = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);
            parse_numeric_single<int32_t>(p, end, rows, cols, delim, mat->getValues());
            res = mat; return;
        } else {
            auto* mat = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
            parse_string_single(p, end, rows, cols, delim, mat->getValues());
            res = mat; return;
        }
    }

    // --- multi-thread: build a row-start index ONCE, then split by rows ---
    std::vector<size_t> rowStarts;
    build_line_index(base, n, dataStart, rows, rowStarts);

    if (vtc == ValueTypeCode::F64) {
        auto* mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
        parse_numeric_rows_parallel<double>(base, rowStarts, rows, cols, delim, mat->getValues(), threads);
        res = mat; return;
    } else if (vtc == ValueTypeCode::UI64) {
        auto* mat = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false);
        parse_numeric_rows_parallel<uint64_t>(base, rowStarts, rows, cols, delim, mat->getValues(), threads);
        res = mat; return;
    }  else if (vtc == ValueTypeCode::SI32) {
        auto* mat = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);
        parse_numeric_rows_parallel<int32_t>(base, rowStarts, rows, cols, delim, mat->getValues(), threads);
        res = mat; return;
    } else {
        // strings: parallelization often regresses due to allocations;
        // if you still want it, implement a rows-parallel variant like numeric.
        const char* p = base + dataStart;
        const char* end = base + n;
        auto* mat = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
        parse_string_single(p, end, rows, cols, delim, mat->getValues());
        res = mat; return;
    }
}

void csv_write(const Structure* matrix,
               const FileMetaData& /*fmd*/,
               const char* filename,
               IOOptions& /*opts*/,
               DaphneContext* /*ctx*/)
{
    FILE* f = std::fopen(filename, "wb");
    if (!f) throw std::runtime_error(std::string("Failed to open for writing: ") + filename);

    auto write_rows = [&](auto* m) {
        using T = std::decay_t<decltype(*m->getValues())>;
        const size_t R = m->getNumRows(), C = m->getNumCols();
        const T* v = m->getValues();
        for (size_t i = 0; i < R; ++i) {
            for (size_t j = 0; j < C; ++j) {
                if constexpr (std::is_same_v<T, std::string>) {
                    std::fwrite(v[i*C + j].data(), 1, v[i*C + j].size(), f);
                } else {
                    char buf[64];
                    if constexpr (std::is_same_v<T, double>) {
                        int n = std::snprintf(buf, sizeof(buf), "%.17g", v[i*C + j]);
                        std::fwrite(buf, 1, n, f);
                    } else if constexpr (std::is_same_v<T, uint64_t>) {
                        int n = std::snprintf(buf, sizeof(buf), "%llu",
                                              static_cast<unsigned long long>(v[i*C + j]));
                        std::fwrite(buf, 1, n, f);
                    } else if constexpr (std::is_same_v<T, int32_t>) {
                        int n = std::snprintf(buf, sizeof(buf), "%d",
                                              static_cast<int>(v[i*C + j]));
                        std::fwrite(buf, 1, n, f);
                    }
                }
                if (j + 1 < C) std::fputc(',', f);
            }
            std::fputc('\n', f);
        }
    };

    if      (auto *im = dynamic_cast<const DenseMatrix<int32_t>*>(matrix))          write_rows(im);
    else if (auto *dm = dynamic_cast<const DenseMatrix<double>*>(matrix))           write_rows(dm);
    else if (auto *um = dynamic_cast<const DenseMatrix<uint64_t>*>(matrix))         write_rows(um);
    else if (auto *sm = dynamic_cast<const DenseMatrix<std::string>*>(matrix))      write_rows(sm);
    else {
        std::fclose(f);
        throw std::runtime_error("csv_write: unsupported matrix type");
    }

    std::fclose(f);
}

} // extern "C"
