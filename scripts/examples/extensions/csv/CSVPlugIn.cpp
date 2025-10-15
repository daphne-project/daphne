// CsvIO.cpp — fast CSV plugin with improved scan & lower indexing memory.
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
#include <charconv> // from_chars, to_chars
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/Structure.h"
#include "runtime/local/io/FileMetaData.h"
#include "runtime/local/io/FileIORegistry.h"
#include "runtime/local/datastructures/Frame.h"
#include "runtime/local/kernels/CreateFrame.h"

#if defined(__has_builtin)
#  if __has_builtin(__builtin_expect)
#    define LIKELY(x)   __builtin_expect(!!(x), 1)
#    define UNLIKELY(x) __builtin_expect(!!(x), 0)
#  else
#    define LIKELY(x)   (x)
#    define UNLIKELY(x) (x)
#  endif
#else
#  define LIKELY(x)   (x)
#  define UNLIKELY(x) (x)
#endif

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
#if defined(POSIX_FADV_SEQUENTIAL)
        ::posix_fadvise(m.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
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

static inline int64_t parse_i64_token(const char* b, const char* e) {
    int64_t out = 0;
    auto res = std::from_chars(b, e, out, 10);
    if (res.ec == std::errc()) return out;
    char* ep = nullptr; return static_cast<int64_t>(std::strtoll(b, &ep, 10));
}

static inline int32_t parse_i32_token(const char* b, const char* e) {
    int32_t out = 0;
    auto res = std::from_chars(b, e, out, 10);
    if (res.ec == std::errc()) return out;
    char* ep = nullptr; return static_cast<int32_t>(std::strtol(b, &ep, 10));
}

static inline size_t header_end(const char* p, size_t n) {
    const void* nl = memchr(p, '\n', n);
    if (!nl) return n;
    size_t i = static_cast<const char*>(nl) - p;
    if (i + 1 <= n) ++i; // skip '\n'
    return std::min(i, n);
}

static inline void trim_token(const char*& b, const char*& e) {
    while (b < e && (*b == ' ' || *b == '\t')) ++b;
    while (e > b && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) --e;
}

static inline const char* find_next_sep_memchr(const char* p, const char* end, char delim, char& which) {
    const size_t len = static_cast<size_t>(end - p);
    const char* pd = static_cast<const char*>(memchr(p, delim, len));
    const char* pn = static_cast<const char*>(memchr(p, '\n',  len));
    if (pd && pn) {
        if (pd < pn) { which = 'd'; return pd; }
        else         { which = 'n'; return pn; }
    } else if (pd)  { which = 'd'; return pd; }
    else if (pn)    { which = 'n'; return pn; }
    which = 0; return end;
}

// Heuristic: check first few rows for leading/trailing blanks. If none, skip trim.
static inline bool detect_likely_no_trim(const char* p, const char* end, char delim, size_t cols, size_t rowsToSample = 8) {
    size_t seenRows = 0;
    const char* cur = p;
    while (seenRows < rowsToSample && cur < end) {
        for (size_t c = 0; c < cols; ++c) {
            char which = 0;
            const char* sep = find_next_sep_memchr(cur, end, delim, which);
            const char* b = cur; const char* e = sep;
            // Check whitespace without actually trimming
            bool leftWS = (b < e) && (*b == ' ' || *b == '\t');
            bool rightWS = (e > b) && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r');
            if (leftWS || rightWS) return false;
            if (which == 'd') cur = sep + 1;
            else { // 'n' or 0
                cur = (which == 'n' && sep < end) ? sep + 1 : sep;
                break;
            }
        }
        ++seenRows;
    }
    return true;
}

//======================= numeric token parsing =======================

static inline uint64_t parse_u64_token(const char* b, const char* e) {
    uint64_t out = 0;
    auto res = std::from_chars(b, e, out, 10);
    if (res.ec != std::errc()) {
        char* ep = nullptr;
        out = static_cast<uint64_t>(std::strtoull(b, &ep, 10));
    }
    return out;
}

static inline double parse_f64_token(const char* b, const char* e) {
    char* ep = nullptr;
    return std::strtod(b, &ep);
}

//======================= single-thread fast paths =======================

template<typename NumT>
static void parse_numeric_single(const char* p, const char* end,
                                 size_t rows, size_t cols, char delim,
                                 NumT* out, bool likelyNoTrim)
{
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c + 1 < cols; ++c) {
            char which = 0;
            const char* sep = find_next_sep_memchr(p, end, delim, which);
            if (UNLIKELY(which != 'd'))
                throw std::runtime_error("CSV: not enough columns (single, delim expected)");
            const char* b = p; const char* q = sep;
            if (!likelyNoTrim) trim_token(b, q);

            if constexpr (std::is_floating_point_v<NumT>) {
                out[r*cols + c] = static_cast<NumT>(parse_f64_token(b, q));
            } else if constexpr (std::is_signed_v<NumT>) {
                out[r*cols + c] = static_cast<NumT>(parse_i64_token(b, q));
            } else {
                out[r*cols + c] = static_cast<NumT>(parse_u64_token(b, q));
            }
            p = sep + 1;
        }
        char which = 0;
        const char* sep = find_next_sep_memchr(p, end, delim, which);
        if (UNLIKELY(which == 'd'))
            throw std::runtime_error("CSV: too many columns (single)");
        const char* b = p; const char* q = sep;
        if (!likelyNoTrim) trim_token(b, q);

        if constexpr (std::is_floating_point_v<NumT>) {
            out[r*cols + (cols-1)] = static_cast<NumT>(parse_f64_token(b, q));
        } else if constexpr (std::is_signed_v<NumT>) {
            out[r*cols + (cols-1)] = static_cast<NumT>(parse_i64_token(b, q));
        } else {
            out[r*cols + (cols-1)] = static_cast<NumT>(parse_u64_token(b, q));
        }
        p = (which == 'n' && sep < end) ? sep + 1 : sep;
    }
}

static void parse_string_single(const char* p, const char* end,
                                size_t rows, size_t cols, char delim,
                                std::string* out, bool likelyNoTrim)
{
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c + 1 < cols; ++c) {
            char which = 0;
            const char* sep = find_next_sep_memchr(p, end, delim, which);
            if (UNLIKELY(which != 'd'))
                throw std::runtime_error("CSV: not enough columns (str, delim expected)");
            const char* b = p; const char* q = sep;
            if (!likelyNoTrim) trim_token(b, q);
            out[r*cols + c].assign(b, q);
            p = sep + 1;
        }
        char which = 0;
        const char* sep = find_next_sep_memchr(p, end, delim, which);
        if (UNLIKELY(which == 'd'))
            throw std::runtime_error("CSV: too many columns (str)");
        const char* b = p; const char* q = sep;
        if (!likelyNoTrim) trim_token(b, q);
        out[r*cols + (cols-1)].assign(b, q);
        p = (which == 'n' && sep < end) ? sep + 1 : sep;
    }
}

//======================= parallel by row ranges =======================

// Build an index of line starts (byte offsets) for exactly `rows` rows.
template<class IndexT>
static inline void build_line_index_t(const char* base, size_t n, size_t dataStart,
                                      size_t rows, std::vector<IndexT>& lineStart) {
    lineStart.clear();
    lineStart.reserve(rows + 1);
    size_t i = dataStart;
    if (dataStart > std::numeric_limits<IndexT>::max())
        throw std::runtime_error("CSV: line index requires 64-bit offsets");
    lineStart.push_back(static_cast<IndexT>(i)); // first row starts here
    while (lineStart.size() < rows && i < n) {
        const void* nl = memchr(base + i, '\n', n - i);
        if (!nl) break;
        size_t pos = static_cast<const char*>(nl) - base;
        size_t next = pos + 1;
        if (next < n) {
            if (next > std::numeric_limits<IndexT>::max())
                throw std::runtime_error("CSV: line index requires 64-bit offsets");
            lineStart.push_back(static_cast<IndexT>(next));
        }
        i = next;
    }
    // Final sentinel: end-of-last-line (or EOF)
    if (i < n) {
        const void* nl = memchr(base + i, '\n', n - i);
        size_t sentinel = nl ? (static_cast<const char*>(nl) - base) : n;
        if (sentinel > std::numeric_limits<IndexT>::max())
            throw std::runtime_error("CSV: line index requires 64-bit offsets");
        lineStart.push_back(static_cast<IndexT>(sentinel));
    } else {
        lineStart.push_back(static_cast<IndexT>(n));
    }
    if (lineStart.size() != rows + 1)
        throw std::runtime_error("CSV: could not index requested number of rows");
}

template<typename NumT, class IndexT>
static void parse_numeric_rows_parallel(const char* base, const std::vector<IndexT>& rowStarts,
                                        size_t rows, size_t cols, char delim,
                                        NumT* out, size_t threads, bool likelyNoTrim)
{
    threads = std::max<size_t>(1, std::min(threads, rows ? rows : 1));
    auto worker = [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const char* p   = base + rowStarts[r];
            const char* end = base + rowStarts[r + 1];
            for (size_t c = 0; c + 1 < cols; ++c) {
                char which = 0;
                const char* sep = find_next_sep_memchr(p, end, delim, which);
                if (UNLIKELY(which != 'd')) throw std::runtime_error("CSV: not enough columns (parallel)");
                const char* b = p; const char* q = sep;
                if (!likelyNoTrim) trim_token(b, q);

                if constexpr (std::is_floating_point_v<NumT>) {
                    out[r*cols + c] = static_cast<NumT>(parse_f64_token(b, q));
                } else if constexpr (std::is_signed_v<NumT>) {
                    out[r*cols + c] = static_cast<NumT>(parse_i64_token(b, q));
                } else {
                    out[r*cols + c] = static_cast<NumT>(parse_u64_token(b, q));
                }
                p = sep + 1;
            }
            char which = 0;
            const char* sep = find_next_sep_memchr(p, end, delim, which);
            if (UNLIKELY(which == 'd')) throw std::runtime_error("CSV: too many columns (parallel)");
            const char* b = p; const char* q = sep;
            if (!likelyNoTrim) trim_token(b, q);

            if constexpr (std::is_floating_point_v<NumT>) {
                out[r*cols + (cols-1)] = static_cast<NumT>(parse_f64_token(b, q));
            } else if constexpr (std::is_signed_v<NumT>) {
                out[r*cols + (cols-1)] = static_cast<NumT>(parse_i64_token(b, q));
            } else {
                out[r*cols + (cols-1)] = static_cast<NumT>(parse_u64_token(b, q));
            }
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

// ---------- FRAME READER (branchless per-cell via prebound column parsers) ----------

struct ColParse {
    void (*parse)(const char* b, const char* e, void* colBase, size_t r);
    void* base = nullptr;
};

static inline ColParse make_col_parse(ValueTypeCode t, void* colBase) {
    switch (t) {
        case ValueTypeCode::F64:  return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<double*>(base)[r] = parse_f64_token(b,e);
        }, colBase};
        case ValueTypeCode::F32:  return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<float*>(base)[r] = static_cast<float>(parse_f64_token(b,e));
        }, colBase};
        case ValueTypeCode::SI64: return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<int64_t*>(base)[r] = parse_i64_token(b,e);
        }, colBase};
        case ValueTypeCode::SI32: return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<int32_t*>(base)[r] = parse_i32_token(b,e);
        }, colBase};
        case ValueTypeCode::SI8:  return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<int8_t*>(base)[r] = static_cast<int8_t>(parse_i64_token(b,e));
        }, colBase};
        case ValueTypeCode::UI64: return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<uint64_t*>(base)[r] = parse_u64_token(b,e);
        }, colBase};
        case ValueTypeCode::UI32: return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<uint32_t*>(base)[r] = static_cast<uint32_t>(parse_u64_token(b,e));
        }, colBase};
        case ValueTypeCode::UI8:  return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<uint8_t*>(base)[r] = static_cast<uint8_t>(parse_u64_token(b,e));
        }, colBase};
        default:                  return {+[](const char* b,const char* e,void* base,size_t r){
            static_cast<std::string*>(base)[r].assign(b,e);
        }, colBase};
    }
}

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
    std::vector<Structure*>     columns(cols, nullptr);
    std::vector<void*>          colBases(cols, nullptr);

    for (size_t c = 0; c < cols; ++c) {
        switch (fmd.schema[c]) {
            case ValueTypeCode::F64: {
                auto* m = DataObjectFactory::create<DenseMatrix<double>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
            case ValueTypeCode::F32: {
                auto* m = DataObjectFactory::create<DenseMatrix<float>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
            case ValueTypeCode::SI64: {
                auto* m = DataObjectFactory::create<DenseMatrix<int64_t>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
            case ValueTypeCode::SI32: {
                auto* m = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
            case ValueTypeCode::SI8: {
                auto* m = DataObjectFactory::create<DenseMatrix<int8_t>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
            case ValueTypeCode::UI64: {
                auto* m = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
            case ValueTypeCode::UI32: {
                auto* m = DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
            case ValueTypeCode::UI8: {
                auto* m = DataObjectFactory::create<DenseMatrix<uint8_t>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
            default: {
                auto* m = DataObjectFactory::create<DenseMatrix<std::string>>(rows, 1, true);
                columns[c] = m; colBases[c] = m->getValues(); break;
            }
        }
    }

    // 4b) Prebind parsers (removes a hot switch per cell)
    std::vector<ColParse> colFns(cols);
    for (size_t c = 0; c < cols; ++c)
        colFns[c] = make_col_parse(fmd.schema[c], colBases[c]);

    // 5) Single-pass parse: row by row, column by column (branchless per cell)
    const bool likelyNoTrim = detect_likely_no_trim(p, end, delim, cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c + 1 < cols; ++c) {
            char which = 0;
            const char* sep = find_next_sep_memchr(p, end, delim, which);
            if (UNLIKELY(which != 'd'))
                throw std::runtime_error("CSV(Frame): not enough columns while expecting delimiter");
            const char* b = p; const char* q = sep;
            if (!likelyNoTrim) trim_token(b, q);
            colFns[c].parse(b, q, colFns[c].base, r);
            p = sep + 1; // move after delimiter
        }
        char which = 0;
        const char* sep = find_next_sep_memchr(p, end, delim, which);
        if (UNLIKELY(which == 'd'))
            throw std::runtime_error("CSV(Frame): too many columns on row");
        const char* b = p; const char* q = sep;
        if (!likelyNoTrim) trim_token(b, q);
        const size_t cLast = cols - 1;
        colFns[cLast].parse(b, q, colFns[cLast].base, r);
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

// ---------- MATRIX READER (single-/multi-thread like before, faster scan, lighter index) ----------

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
        switch (fmd.schema[0]) {
            case ValueTypeCode::F64:  vtc = ValueTypeCode::F64;  break;
            case ValueTypeCode::F32:  vtc = ValueTypeCode::F32;  break;
            case ValueTypeCode::SI64: vtc = ValueTypeCode::SI64; break;
            case ValueTypeCode::SI32: vtc = ValueTypeCode::SI32; break;
            case ValueTypeCode::SI8:  vtc = ValueTypeCode::SI8;  break;
            case ValueTypeCode::UI64: vtc = ValueTypeCode::UI64; break;
            case ValueTypeCode::UI32: vtc = ValueTypeCode::UI32; break;
            case ValueTypeCode::UI8:  vtc = ValueTypeCode::UI8;  break;
            default:                  vtc = ValueTypeCode::STR;  break;
        }
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

    // --- detect trim behavior once ---
    const char* p0 = base + dataStart;
    const char* e0 = base + n;
    const bool likelyNoTrim = detect_likely_no_trim(p0, e0, delim, cols);

    // --- ultra-fast single-thread path (no extra indexing) ---
    if (threads == 1) {
        const char* p = p0;
        const char* end = e0;

        switch (vtc) {
            case ValueTypeCode::F64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
                parse_numeric_single<double>(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::F32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<float>>(rows, cols, false);
                parse_numeric_single<float>(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int64_t>>(rows, cols, false);
                parse_numeric_single<int64_t>(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);
                parse_numeric_single<int32_t>(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI8: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int8_t>>(rows, cols, false);
                parse_numeric_single<int8_t>(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false);
                parse_numeric_single<uint64_t>(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, cols, false);
                parse_numeric_single<uint32_t>(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI8: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint8_t>>(rows, cols, false);
                parse_numeric_single<uint8_t>(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
            default: {
                auto* mat = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
                parse_string_single(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
        }
    }

    // --- multi-thread: build a row-start index ONCE, then split by rows ---
    // Optimize memory: use 32-bit offsets if addressable.
    const bool canUse32 = (n <= std::numeric_limits<uint32_t>::max());
    if (canUse32) {
        std::vector<uint32_t> rowStarts32;
        build_line_index_t(base, n, dataStart, rows, rowStarts32);
        switch (vtc) {
            case ValueTypeCode::F64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
                parse_numeric_rows_parallel<double>(base, rowStarts32, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::F32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<float>>(rows, cols, false);
                parse_numeric_rows_parallel<float>(base, rowStarts32, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int64_t>>(rows, cols, false);
                parse_numeric_rows_parallel<int64_t>(base, rowStarts32, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);
                parse_numeric_rows_parallel<int32_t>(base, rowStarts32, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI8: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int8_t>>(rows, cols, false);
                parse_numeric_rows_parallel<int8_t>(base, rowStarts32, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false);
                parse_numeric_rows_parallel<uint64_t>(base, rowStarts32, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, cols, false);
                parse_numeric_rows_parallel<uint32_t>(base, rowStarts32, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI8: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint8_t>>(rows, cols, false);
                parse_numeric_rows_parallel<uint8_t>(base, rowStarts32, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            default: {
                // strings keep single-thread (typically bound by allocations)
                const char* p = base + dataStart;
                const char* end = base + n;
                auto* mat = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
                parse_string_single(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
        }
    } else {
        std::vector<size_t> rowStarts;
        build_line_index_t(base, n, dataStart, rows, rowStarts);
        switch (vtc) {
            case ValueTypeCode::F64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
                parse_numeric_rows_parallel<double>(base, rowStarts, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::F32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<float>>(rows, cols, false);
                parse_numeric_rows_parallel<float>(base, rowStarts, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int64_t>>(rows, cols, false);
                parse_numeric_rows_parallel<int64_t>(base, rowStarts, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);
                parse_numeric_rows_parallel<int32_t>(base, rowStarts, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::SI8: {
                auto* mat = DataObjectFactory::create<DenseMatrix<int8_t>>(rows, cols, false);
                parse_numeric_rows_parallel<int8_t>(base, rowStarts, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI64: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false);
                parse_numeric_rows_parallel<uint64_t>(base, rowStarts, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI32: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, cols, false);
                parse_numeric_rows_parallel<uint32_t>(base, rowStarts, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            case ValueTypeCode::UI8: {
                auto* mat = DataObjectFactory::create<DenseMatrix<uint8_t>>(rows, cols, false);
                parse_numeric_rows_parallel<uint8_t>(base, rowStarts, rows, cols, delim, mat->getValues(), threads, likelyNoTrim);
                res = mat; return;
            }
            default: {
                const char* p = base + dataStart;
                const char* end = base + n;
                auto* mat = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);
                parse_string_single(p, end, rows, cols, delim, mat->getValues(), likelyNoTrim);
                res = mat; return;
            }
        }
    }
}

#include <cinttypes> // PRId64, PRIu64

void csv_write(const Structure* matrix,
               const FileMetaData& /*fmd*/,
               const char* filename,
               IOOptions& opts,
               DaphneContext* /*ctx*/)
{
    // 1) Resolve delimiter from opts (default: ',')
    char delim = ',';
    if (auto it = opts.extra.find("delimiter"); it != opts.extra.end()) {
        const std::string &d = it->second;
        if (d.size() != 1)
            throw std::runtime_error("csv_write: 'delimiter' must be a single character");
        delim = d[0];
    }

    FILE* f = std::fopen(filename, "wb");
    if (!f) throw std::runtime_error(std::string("Failed to open for writing: ") + filename);
    // Big stdio buffer to reduce syscalls
    setvbuf(f, nullptr, _IOFBF, 1 << 20);

    auto write_rows = [&](auto* m) {
        using T = std::decay_t<decltype(*m->getValues())>;
        const size_t R = m->getNumRows(), C = m->getNumCols();
        const T* v = m->getValues();

        char numbuf[128];

        for (size_t i = 0; i < R; ++i) {
            for (size_t j = 0; j < C; ++j) {
                if constexpr (std::is_same_v<T, std::string>) {
                    const auto &s = v[i*C + j];
                    std::fwrite(s.data(), 1, s.size(), f);
                } else {
                    // Prefer to_chars for ints; snprintf for floats
                    size_t n = 0;
                    if constexpr (std::is_floating_point_v<T>) {
                        if constexpr (std::is_same_v<T, double>) {
                            n = static_cast<size_t>(std::snprintf(numbuf, sizeof(numbuf), "%.17g", v[i*C + j]));
                        } else {
                            n = static_cast<size_t>(std::snprintf(numbuf, sizeof(numbuf), "%.9g", v[i*C + j]));
                        }
                    } else if constexpr (std::is_integral_v<T>) {
                        auto [ptr, ec] = std::to_chars(numbuf, numbuf + sizeof(numbuf), v[i*C + j]);
                        if (ec != std::errc()) throw std::runtime_error("csv_write: to_chars failed");
                        n = static_cast<size_t>(ptr - numbuf);
                    } else {
                        static_assert(!sizeof(T*), "csv_write: unsupported element type");
                    }
                    std::fwrite(numbuf, 1, n, f);
                }
                if (j + 1 < C) std::fputc(delim, f);
            }
            std::fputc('\n', f);
        }
    };

    if      (auto *m = dynamic_cast<const DenseMatrix<double>*>(matrix))           write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<float>*>(matrix))            write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<int64_t>*>(matrix))          write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<int32_t>*>(matrix))          write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<int8_t>*>(matrix))           write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<uint64_t>*>(matrix))         write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<uint32_t>*>(matrix))         write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<uint8_t>*>(matrix))          write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<std::string>*>(matrix))      write_rows(m);
    else {
        std::fclose(f);
        throw std::runtime_error("csv_write: unsupported matrix type");
    }

    std::fclose(f);
}

} // extern "C"
