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
#include <charconv>
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/Structure.h"
#include "runtime/local/datastructures/Frame.h"
#include "runtime/local/kernels/CreateFrame.h"
#include "runtime/local/io/FileMetaData.h"
#include "runtime/local/io/FileIORegistry.h"


//======================= file mapping =======================
// ------------------------ small mmap helper ------------------------
struct MappedFile {
    const char* data = nullptr; size_t size = 0; int fd = -1; void* map = MAP_FAILED;
    static MappedFile open(const char* path) {
        MappedFile m;
        m.fd = ::open(path, O_RDONLY);
        if (m.fd < 0) throw std::runtime_error(std::string("open: ") + path);
        struct stat st{}; if (fstat(m.fd, &st) != 0) { int e=errno; ::close(m.fd); throw std::system_error(e, std::generic_category(), "fstat"); }
        m.size = static_cast<size_t>(st.st_size);
        if (m.size) {
            m.map = ::mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, m.fd, 0);
            if (m.map == MAP_FAILED) { int e=errno; ::close(m.fd); throw std::system_error(e, std::generic_category(), "mmap"); }
            m.data = static_cast<const char*>(m.map);
        }
        return m;
    }
    ~MappedFile(){ if(map!=MAP_FAILED) ::munmap(map,size); if(fd>=0) ::close(fd); }
};

static inline size_t skip_header(const char* base, size_t n) {
    const void* nl = memchr(base, '\n', n);
    return nl ? (static_cast<const char*>(nl) - base) + 1 : n;
}

// ultra-cheap parsers ---------------------------------------------------
static inline double  p_f64(const char* b, const char* e){ char* ep=nullptr; return std::strtod(b, &ep); }
static inline int64_t p_i64(const char* b, const char* e){ int64_t v=0; auto r=std::from_chars(b,e,v,10); if(r.ec==std::errc()) return v; char* ep=nullptr; return std::strtoll(b,&ep,10); }
static inline int32_t p_i32(const char* b, const char* e){ int32_t v=0; auto r=std::from_chars(b,e,v,10); if(r.ec==std::errc()) return v; char* ep=nullptr; return std::strtol(b,&ep,10); }
static inline uint64_t p_u64(const char* b, const char* e){ uint64_t v=0; auto r=std::from_chars(b,e,v,10); if(r.ec==std::errc()) return v; char* ep=nullptr; return std::strtoull(b,&ep,10); }

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

static inline size_t header_end(const char* p, size_t n) {
    const void* nl = memchr(p, '\n', n);
    if (!nl) return n;
    size_t i = static_cast<const char*>(nl) - p;
    return std::min(i + 1, n); // skip '\n'
}
static inline void trim_token(const char*& b, const char*& e) {
    while (b < e && (*b == ' ' || *b == '\t')) ++b;
    while (e > b && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) --e;
}

static inline const char* find_next_sep_memchr(const char* p, const char* end, char delim, char& which) {
    const size_t len = static_cast<size_t>(end - p);
    const char* pd = static_cast<const char*>(memchr(p, delim, len));
    const char* pn = static_cast<const char*>(memchr(p, '\n',  len));
    if (pd && pn) { which = (pd < pn) ? 'd' : 'n'; return (pd < pn) ? pd : pn; }
    if (pd) { which = 'd'; return pd; }
    if (pn) { which = 'n'; return pn; }
    which = 0; return end;
}

static inline bool detect_likely_no_trim(const char* p, const char* end, char delim, size_t cols, size_t rowsToSample = 8) {
    size_t seen = 0; const char* cur = p;
    while (seen < rowsToSample && cur < end) {
        for (size_t c = 0; c < cols; ++c) {
            char w=0; const char* sep = find_next_sep_memchr(cur, end, delim, w);
            const char* b=cur; const char* e=sep;
            bool leftWS = (b < e) && (*b == ' ' || *b == '\t');
            bool rightWS= (e > b) && (e[-1]==' '||e[-1]=='\t'||e[-1]=='\r');
            if (leftWS || rightWS) return false;
            cur = (w=='d') ? sep+1 : ((w=='n' && sep<end) ? sep+1 : sep);
            if (w!='d') break;
        }
        ++seen;
    }
    return true;
}
//======================= single-thread parse =======================

template<typename NumT>
static void parse_numeric_single(const char* p, const char* end,
                                 size_t rows, size_t cols, char delim,
                                 NumT* out, bool noTrim)
{
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c + 1 < cols; ++c) {
            char w=0; const char* sep = find_next_sep_memchr(p, end, delim, w);
            if (w != 'd') throw std::runtime_error("CSV: not enough columns");
            const char* b=p; const char* q=sep; if (!noTrim) trim_token(b,q);
            if constexpr (std::is_floating_point_v<NumT>) out[r*cols+c] = static_cast<NumT>(parse_f64_token(b,q));
            else if constexpr (std::is_signed_v<NumT>)    out[r*cols+c] = static_cast<NumT>(parse_i64_token(b,q));
            else                                          out[r*cols+c] = static_cast<NumT>(parse_u64_token(b,q));
            p = sep + 1;
        }
        char w=0; const char* sep = find_next_sep_memchr(p, end, delim, w);
        if (w=='d') throw std::runtime_error("CSV: too many columns");
        const char* b=p; const char* q=sep; if (!noTrim) trim_token(b,q);
        if constexpr (std::is_floating_point_v<NumT>) out[r*cols+cols-1] = static_cast<NumT>(parse_f64_token(b,q));
        else if constexpr (std::is_signed_v<NumT>)    out[r*cols+cols-1] = static_cast<NumT>(parse_i64_token(b,q));
        else                                          out[r*cols+cols-1] = static_cast<NumT>(parse_u64_token(b,q));
        p = (w=='n' && sep<end) ? sep+1 : sep;
    }
}

static void parse_string_single(const char* p, const char* end,
                                size_t rows, size_t cols, char delim,
                                std::string* out, bool noTrim)
{
    for (size_t r=0;r<rows;++r) {
        for (size_t c=0;c+1<cols;++c) {
            char w=0; const char* sep = find_next_sep_memchr(p,end,delim,w);
            if (w!='d') throw std::runtime_error("CSV: not enough columns (str)");
            const char* b=p; const char* q=sep; if (!noTrim) trim_token(b,q);
            out[r*cols+c].assign(b,q);
            p = sep+1;
        }
        char w=0; const char* sep = find_next_sep_memchr(p,end,delim,w);
        if (w=='d') throw std::runtime_error("CSV: too many columns (str)");
        const char* b=p; const char* q=sep; if (!noTrim) trim_token(b,q);
        out[r*cols+cols-1].assign(b,q);
        p = (w=='n' && sep<end) ? sep+1 : sep;
    }
}

//======================= row index (parallel) =======================

template<class IndexT>
static inline void build_line_index_t(const char* base, size_t n, size_t dataStart,
                                      size_t rows, std::vector<IndexT>& lineStart) {
    lineStart.clear(); lineStart.reserve(rows+1);
    size_t i = dataStart;
    if (dataStart > std::numeric_limits<IndexT>::max())
        throw std::runtime_error("CSV: need 64-bit offsets");
    lineStart.push_back(static_cast<IndexT>(i));
    while (lineStart.size() < rows && i < n) {
        const void* nl = memchr(base+i, '\n', n-i);
        if (!nl) break;
        size_t pos = static_cast<const char*>(nl) - base;
        size_t next = pos + 1;
        if (next > std::numeric_limits<IndexT>::max())
            throw std::runtime_error("CSV: need 64-bit offsets");
        lineStart.push_back(static_cast<IndexT>(next));
        i = next;
    }
    size_t sentinel;
    if (i < n) {
        const void* nl = memchr(base+i, '\n', n-i);
        sentinel = nl ? (static_cast<const char*>(nl) - base) : n;
    } else sentinel = n;
    if (sentinel > std::numeric_limits<IndexT>::max())
        throw std::runtime_error("CSV: need 64-bit offsets");
    lineStart.push_back(static_cast<IndexT>(sentinel));
    if (lineStart.size() != rows + 1)
        throw std::runtime_error("CSV: could not index requested rows");
}

template<typename NumT, class IndexT>
static void parse_numeric_rows_parallel(const char* base, const std::vector<IndexT>& rowStarts,
                                        size_t rows, size_t cols, char delim,
                                        NumT* out, size_t threads, bool noTrim)
{
    threads = std::max<size_t>(1, std::min(threads, rows ? rows : 1));
    auto worker = [&](size_t r0, size_t r1){
        for (size_t r = r0; r < r1; ++r) {
            const char* p   = base + rowStarts[r];
            const char* end = base + rowStarts[r+1];
            for (size_t c=0;c+1<cols;++c){
                char w=0; const char* sep = find_next_sep_memchr(p,end,delim,w);
                if (w!='d') throw std::runtime_error("CSV: not enough columns");
                const char* b=p; const char* q=sep; if (!noTrim) trim_token(b,q);
                if constexpr (std::is_floating_point_v<NumT>) out[r*cols+c] = static_cast<NumT>(parse_f64_token(b,q));
                else if constexpr (std::is_signed_v<NumT>)    out[r*cols+c] = static_cast<NumT>(parse_i64_token(b,q));
                else                                          out[r*cols+c] = static_cast<NumT>(parse_u64_token(b,q));
                p = sep+1;
            }
            char w=0; const char* sep = find_next_sep_memchr(p,end,delim,w);
            if (w=='d') throw std::runtime_error("CSV: too many columns");
            const char* b=p; const char* q=sep; if (!noTrim) trim_token(b,q);
            if constexpr (std::is_floating_point_v<NumT>) out[r*cols+cols-1] = static_cast<NumT>(parse_f64_token(b,q));
            else if constexpr (std::is_signed_v<NumT>)    out[r*cols+cols-1] = static_cast<NumT>(parse_i64_token(b,q));
            else                                          out[r*cols+cols-1] = static_cast<NumT>(parse_u64_token(b,q));
        }
    };

    std::vector<std::thread> pool;
    const size_t chunk = (rows + threads - 1) / threads;
    size_t r0 = 0;
    for (size_t t=0; t<threads && r0<rows; ++t) {
        size_t r1 = std::min(rows, r0 + chunk);
        pool.emplace_back(worker, r0, r1);
        r0 = r1;
    }
    for (auto& th : pool) th.join();
}

//======================= Fast buffered writer =======================

struct FastCSVOut {
    FILE* f;
    std::vector<char> buf;
    size_t pos = 0;

    explicit FastCSVOut(FILE* f_, size_t cap = 32u << 20) : f(f_), buf(cap) {
        setvbuf(f, nullptr, _IOFBF, 1 << 20);
    }
    ~FastCSVOut() { flush(); }

    inline void flush() {
        if (pos) { std::fwrite(buf.data(), 1, pos, f); pos = 0; }
    }
    inline void ensure(size_t n) {
        if (pos + n > buf.size()) flush();
        if (n > buf.size()) buf.resize(std::max(buf.size()*2, n));
    }
    inline void write(const char* p, size_t n) {
        if (n > buf.size()) { flush(); std::fwrite(p, 1, n, f); return; }
        ensure(n); memcpy(buf.data()+pos, p, n); pos += n;
    }
    inline void putc(char c) {
        if (pos == buf.size()) flush();
        buf[pos++] = c;
    }

    template<class Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
    inline void append_num(Int v) {
        char tmp[32];
        auto [p, ec] = std::to_chars(tmp, tmp + sizeof(tmp), v);
        if (ec != std::errc()) throw std::runtime_error("to_chars(int) failed");
        write(tmp, static_cast<size_t>(p - tmp));
    }

#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    template<class FP, std::enable_if_t<std::is_floating_point_v<FP>, int> = 0>
    inline void append_num(FP v) {
        char tmp[64];
        auto [p, ec] = std::to_chars(tmp, tmp + sizeof(tmp), v, std::chars_format::general);
        if (ec != std::errc()) {
            int n = std::snprintf(tmp, sizeof(tmp), std::is_same_v<FP,double> ? "%.17g" : "%.9g", v);
            write(tmp, static_cast<size_t>(n));
        } else write(tmp, static_cast<size_t>(p - tmp));
    }
#else
    template<class FP, std::enable_if_t<std::is_floating_point_v<FP>, int> = 0>
    inline void append_num(FP v) {
        char tmp[64];
        int n = std::snprintf(tmp, sizeof(tmp), std::is_same_v<FP,double> ? "%.17g" : "%.9g", v);
        write(tmp, static_cast<size_t>(n));
    }
#endif
};

//======================= plugin API =======================
namespace {
// ------------------------ NEW FRAME READER ------------------------
static void csv_read_frame_impl(Frame*& outFrame,
                    const FileMetaData& fmd,
                    const char* filename,
                    IOOptions& opts,
                    DaphneContext* ctx)
{
    const size_t R = fmd.numRows, C = fmd.numCols;
    if (!R || !C) throw std::runtime_error("CSV(Frame): numRows/numCols required");
    if (fmd.schema.size() < C) throw std::runtime_error("CSV(Frame): schema missing");
    if (fmd.labels.size() < C) throw std::runtime_error("CSV(Frame): labels missing");

    // options
    bool hasHeader=false; char delim=',';
    size_t threads = std::thread::hardware_concurrency(); if(!threads) threads=4;
    if (auto it=opts.extra.find("hasHeader"); it!=opts.extra.end()) hasHeader = (it->second=="true"||it->second=="1");
    if (auto it=opts.extra.find("delimiter"); it!=opts.extra.end()) { if (it->second.size()!=1) throw std::runtime_error("CSV(Frame): delimiter must be 1 char"); delim = it->second[0]; }
    if (auto it=opts.extra.find("threads");   it!=opts.extra.end()) threads = std::max<size_t>(1, std::stoul(it->second));

    // projection: useCols
    std::vector<size_t> wanted; wanted.reserve(C);
    std::vector<const char*> labels; labels.reserve(C);
    if (auto it=opts.extra.find("useCols"); it!=opts.extra.end()) {
        std::unordered_map<std::string,size_t> L2I; L2I.reserve(C*2);
        for(size_t c=0;c<C;++c) L2I.emplace(fmd.labels[c], c);
        std::string s = it->second; size_t p0=0;
        while (p0 <= s.size()) {
            size_t p1 = s.find(',', p0);
            std::string key = s.substr(p0, p1==std::string::npos ? std::string::npos : p1-p0);
            // trim
            size_t a=0,b=key.size(); while(a<b && (key[a]==' '||key[a]=='\t')) ++a; while(b>a && (key[b-1]==' '||key[b-1]=='\t')) --b;
            if (b>a) {
                auto f = L2I.find(key.substr(a,b-a));
                if (f==L2I.end()) throw std::runtime_error("CSV(Frame): unknown label in useCols: "+key);
                wanted.push_back(f->second);
                labels.push_back(fmd.labels[f->second].c_str());
            }
            if (p1==std::string::npos) break; p0 = p1+1;
        }
    }
    if (wanted.empty()) {
        wanted.resize(C); std::iota(wanted.begin(), wanted.end(), 0);
        labels.resize(C); for(size_t c=0;c<C;++c) labels[c]=fmd.labels[c].c_str();
    }
    const size_t W = wanted.size();
    std::vector<uint8_t> isWanted(C,0); std::vector<size_t> posOf(C,0);
    for (size_t k=0;k<W;++k){ isWanted[wanted[k]]=1; posOf[wanted[k]]=k; }

    // mmap + header skip
    MappedFile mf = MappedFile::open(filename);
    if (!mf.size) throw std::runtime_error("CSV(Frame): empty file");
    const char* base = mf.data; const size_t n = mf.size;
    size_t dataStart = hasHeader ? skip_header(base, n) : 0;
    if (dataStart >= n) throw std::runtime_error("CSV(Frame): no data region");

    // decide blocks by rows; find T cut points with a single cheap scan
    threads = std::min(threads, R);
    const size_t rowsPerBlk = (R + threads - 1) / threads;
    std::vector<size_t> rCuts; rCuts.reserve(threads+1);
    std::vector<size_t> bCuts; bCuts.reserve(threads+1);
    rCuts.push_back(0); bCuts.push_back(dataStart);

    size_t rCount = 0, nextCut = rowsPerBlk;
    const char* p = base + dataStart;
    const char* end = base + n;
    while (p < end && rCuts.size() < threads) {
        const void* nl = memchr(p, '\n', static_cast<size_t>(end - p));
        if (!nl) break;
        p = static_cast<const char*>(nl) + 1;
        if (++rCount == nextCut) {
            rCuts.push_back(rCount);
            bCuts.push_back(static_cast<size_t>(p - base));
            nextCut += rowsPerBlk;
        }
    }
    rCuts.push_back(R);
    bCuts.push_back(n);
    const size_t B = rCuts.size() - 1;

    // allocate only wanted columns
    std::vector<Structure*> colsOut(W,nullptr);
    std::vector<void*>      baseOut(W,nullptr);
    for (size_t k=0;k<W;++k) {
        size_t c = wanted[k];
        switch (fmd.schema[c]) {
            case ValueTypeCode::F64:  { auto* m=DataObjectFactory::create<DenseMatrix<double>>(R,1,true);  colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            case ValueTypeCode::F32:  { auto* m=DataObjectFactory::create<DenseMatrix<float>>(R,1,true);   colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            case ValueTypeCode::SI64: { auto* m=DataObjectFactory::create<DenseMatrix<int64_t>>(R,1,true); colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            case ValueTypeCode::SI32: { auto* m=DataObjectFactory::create<DenseMatrix<int32_t>>(R,1,true); colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            case ValueTypeCode::UI64: { auto* m=DataObjectFactory::create<DenseMatrix<uint64_t>>(R,1,true);colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            case ValueTypeCode::UI32: { auto* m=DataObjectFactory::create<DenseMatrix<uint32_t>>(R,1,true);colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            case ValueTypeCode::SI8:  { auto* m=DataObjectFactory::create<DenseMatrix<int8_t>>(R,1,true);  colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            case ValueTypeCode::UI8:  { auto* m=DataObjectFactory::create<DenseMatrix<uint8_t>>(R,1,true); colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            case ValueTypeCode::FIXEDSTR16: { auto*m=DataObjectFactory::create<DenseMatrix<FixedStr16>>(R,1,true); colsOut[k]=m; baseOut[k]=m->getValues(); break; }
            default:                  { auto* m=DataObjectFactory::create<DenseMatrix<std::string>>(R,1,true); colsOut[k]=m; baseOut[k]=m->getValues(); break; }
        }
    }

    // worker: parse rows [r0, r1) from byte span [b0, b1)
    auto worker = [&](size_t bid){
        size_t r0 = rCuts[bid], r1 = rCuts[bid+1];
        const char* q   = base + bCuts[bid];
        const char* qend= base + bCuts[bid+1];

        for (size_t r = r0; r < r1; ++r) {
            // For each column: walk to delim or '\n'; write if wanted.
            for (size_t c = 0; c < C; ++c) {
                const char* tok_b = q;
                // tight scan: no memchr, no trim
                while (q < qend && *q != delim && *q != '\n') ++q;
                const char* tok_e = q;

                if (isWanted[c]) {
                    size_t k = posOf[c];
                    switch (fmd.schema[c]) {
                        case ValueTypeCode::F64:  static_cast<double*   >(baseOut[k])[r] = p_f64(tok_b, tok_e); break;
                        case ValueTypeCode::F32:  static_cast<float*    >(baseOut[k])[r] = static_cast<float>(p_f64(tok_b, tok_e)); break;
                        case ValueTypeCode::SI64: static_cast<int64_t*  >(baseOut[k])[r] = p_i64(tok_b, tok_e); break;
                        case ValueTypeCode::SI32: static_cast<int32_t*  >(baseOut[k])[r] = p_i32(tok_b, tok_e); break;
                        case ValueTypeCode::UI64: static_cast<uint64_t* >(baseOut[k])[r] = p_u64(tok_b, tok_e); break;
                        case ValueTypeCode::UI32: static_cast<uint32_t* >(baseOut[k])[r] = static_cast<uint32_t>(p_u64(tok_b, tok_e)); break;
                        case ValueTypeCode::SI8:  static_cast<int8_t*   >(baseOut[k])[r] = static_cast<int8_t >(p_i64(tok_b, tok_e)); break;
                        case ValueTypeCode::UI8:  static_cast<uint8_t*  >(baseOut[k])[r] = static_cast<uint8_t>(p_u64(tok_b, tok_e)); break;
                        case ValueTypeCode::FIXEDSTR16: {
                            // Trim trailing '\r' just in case (Windows line endings)
                            const char* endTok = tok_e;
                            if (endTok > tok_b && endTok[-1] == '\r')
                                --endTok;

                            // Construct a std::string first, because FixedStr16 only takes that
                            std::string tmp(tok_b, endTok);
                            reinterpret_cast<FixedStr16*>(baseOut[k])[r] = FixedStr16(tmp);
                            break;
                        }

                        default:                  static_cast<std::string*>(baseOut[k])[r].assign(tok_b, tok_e); break;
                    }
                }

                // advance over separator if present
                if (q < qend) {
                    char ch = *q++;
                    if (ch == '\n') break; // end of row
                    // else ch==delim, continue to next column
                }
            }
            
        }
    };

    // run workers
    std::vector<std::thread> pool; pool.reserve(B);
    for (size_t b=0; b<B; ++b) pool.emplace_back(worker, b);
    for (auto& th : pool) th.join();

    createFrame(outFrame, colsOut.data(), W, labels.data(), W, ctx);
}

// ---------- MATRIX READER ----------

static void csv_read_impl(Structure*& res,
              const FileMetaData& fmd,
              const char* filename,
              IOOptions& opts,
              DaphneContext* /*ctx*/)
{
    // options
    bool hasHeader = false; // default: SSB style (no header)
    char delim = ',';
    size_t threads = 1;

    if (auto it = opts.extra.find("hasHeader"); it != opts.extra.end())
        hasHeader = (it->second == "true" || it->second == "1");
    if (auto it = opts.extra.find("delimiter"); it != opts.extra.end()) {
        if (it->second.size() != 1) throw std::runtime_error("CSV: delimiter must be 1 char");
        delim = it->second[0];
    }
    if (auto it = opts.extra.find("threads"); it != opts.extra.end())
        threads = std::max<size_t>(1, std::stoul(it->second));

    // type rule for homogeneous matrix
    ValueTypeCode vtc = ValueTypeCode::STR;
    if (fmd.isSingleValueType && !fmd.schema.empty())
        vtc = fmd.schema[0];

    const size_t rows = fmd.numRows, cols = fmd.numCols;
    if (rows == 0 || cols == 0) throw std::runtime_error("CSV: numRows/numCols required");

    MappedFile mf = MappedFile::open(filename);
    if (mf.size == 0) throw std::runtime_error("CSV: empty file");
    const char* base = mf.data; const size_t n = mf.size;
    const size_t dataStart = hasHeader ? header_end(base, n) : 0;
    if (dataStart >= n) throw std::runtime_error("CSV: no data region");

    const char* p0 = base + dataStart; const char* e0 = base + n;
    const bool noTrim = detect_likely_no_trim(p0, e0, delim, cols);

    if (threads == 1) {
        const char* p = p0; const char* end = e0;
        switch (vtc) {
            case ValueTypeCode::F64:  { auto* m = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);    parse_numeric_single<double>(p,end,rows,cols,delim,m->getValues(),noTrim); res = m; return; }
            case ValueTypeCode::F32:  { auto* m = DataObjectFactory::create<DenseMatrix<float>>(rows, cols, false);     parse_numeric_single<float>(p,end,rows,cols,delim,m->getValues(),noTrim);  res = m; return; }
            case ValueTypeCode::SI64: { auto* m = DataObjectFactory::create<DenseMatrix<int64_t>>(rows, cols, false);   parse_numeric_single<int64_t>(p,end,rows,cols,delim,m->getValues(),noTrim);res = m; return; }
            case ValueTypeCode::SI32: { auto* m = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);   parse_numeric_single<int32_t>(p,end,rows,cols,delim,m->getValues(),noTrim);res = m; return; }
            case ValueTypeCode::SI8:  { auto* m = DataObjectFactory::create<DenseMatrix<int8_t>>(rows, cols, false);    parse_numeric_single<int8_t>(p,end,rows,cols,delim,m->getValues(),noTrim); res = m; return; }
            case ValueTypeCode::UI64: { auto* m = DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false);  parse_numeric_single<uint64_t>(p,end,rows,cols,delim,m->getValues(),noTrim);res = m; return; }
            case ValueTypeCode::UI32: { auto* m = DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, cols, false);  parse_numeric_single<uint32_t>(p,end,rows,cols,delim,m->getValues(),noTrim);res = m; return; }
            case ValueTypeCode::UI8:  { auto* m = DataObjectFactory::create<DenseMatrix<uint8_t>>(rows, cols, false);   parse_numeric_single<uint8_t>(p,end,rows,cols,delim,m->getValues(),noTrim); res = m; return; }
            default:                  { auto* m = DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false);parse_string_single(p,end,rows,cols,delim,m->getValues(),noTrim);          res = m; return; }
        }
    }

    const bool canUse32 = (n <= std::numeric_limits<uint32_t>::max());
    if (canUse32) {
        std::vector<uint32_t> rowStarts32; build_line_index_t(base, n, dataStart, rows, rowStarts32);
        switch (vtc) {
            case ValueTypeCode::F64:  { auto* m=DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);   parse_numeric_rows_parallel<double>(base,rowStarts32,rows,cols,delim,m->getValues(),threads,noTrim); res=m; return; }
            case ValueTypeCode::F32:  { auto* m=DataObjectFactory::create<DenseMatrix<float>>(rows, cols, false);    parse_numeric_rows_parallel<float>(base,rowStarts32,rows,cols,delim,m->getValues(),threads,noTrim);  res=m; return; }
            case ValueTypeCode::SI64: { auto* m=DataObjectFactory::create<DenseMatrix<int64_t>>(rows, cols, false);  parse_numeric_rows_parallel<int64_t>(base,rowStarts32,rows,cols,delim,m->getValues(),threads,noTrim);res=m; return; }
            case ValueTypeCode::SI32: { auto* m=DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);  parse_numeric_rows_parallel<int32_t>(base,rowStarts32,rows,cols,delim,m->getValues(),threads,noTrim);res=m; return; }
            case ValueTypeCode::SI8:  { auto* m=DataObjectFactory::create<DenseMatrix<int8_t>>(rows, cols, false);   parse_numeric_rows_parallel<int8_t>(base,rowStarts32,rows,cols,delim,m->getValues(),threads,noTrim); res=m; return; }
            case ValueTypeCode::UI64: { auto* m=DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false); parse_numeric_rows_parallel<uint64_t>(base,rowStarts32,rows,cols,delim,m->getValues(),threads,noTrim);res=m; return; }
            case ValueTypeCode::UI32: { auto* m=DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, cols, false); parse_numeric_rows_parallel<uint32_t>(base,rowStarts32,rows,cols,delim,m->getValues(),threads,noTrim);res=m; return; }
            case ValueTypeCode::UI8:  { auto* m=DataObjectFactory::create<DenseMatrix<uint8_t>>(rows, cols, false);  parse_numeric_rows_parallel<uint8_t>(base,rowStarts32,rows,cols,delim,m->getValues(),threads,noTrim); res=m; return; }
            default:                  { const char* p=base+dataStart; const char* e=base+n; auto* m=DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false); parse_string_single(p,e,rows,cols,delim,m->getValues(),noTrim); res=m; return; }
        }
    } else {
        std::vector<size_t> rowStarts; build_line_index_t(base, n, dataStart, rows, rowStarts);
        switch (vtc) {
            case ValueTypeCode::F64:  { auto* m=DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);   parse_numeric_rows_parallel<double>(base,rowStarts,rows,cols,delim,m->getValues(),threads,noTrim); res=m; return; }
            case ValueTypeCode::F32:  { auto* m=DataObjectFactory::create<DenseMatrix<float>>(rows, cols, false);    parse_numeric_rows_parallel<float>(base,rowStarts,rows,cols,delim,m->getValues(),threads,noTrim);  res=m; return; }
            case ValueTypeCode::SI64: { auto* m=DataObjectFactory::create<DenseMatrix<int64_t>>(rows, cols, false);  parse_numeric_rows_parallel<int64_t>(base,rowStarts,rows,cols,delim,m->getValues(),threads,noTrim);res=m; return; }
            case ValueTypeCode::SI32: { auto* m=DataObjectFactory::create<DenseMatrix<int32_t>>(rows, cols, false);  parse_numeric_rows_parallel<int32_t>(base,rowStarts,rows,cols,delim,m->getValues(),threads,noTrim);res=m; return; }
            case ValueTypeCode::SI8:  { auto* m=DataObjectFactory::create<DenseMatrix<int8_t>>(rows, cols, false);   parse_numeric_rows_parallel<int8_t>(base,rowStarts,rows,cols,delim,m->getValues(),threads,noTrim); res=m; return; }
            case ValueTypeCode::UI64: { auto* m=DataObjectFactory::create<DenseMatrix<uint64_t>>(rows, cols, false); parse_numeric_rows_parallel<uint64_t>(base,rowStarts,rows,cols,delim,m->getValues(),threads,noTrim);res=m; return; }
            case ValueTypeCode::UI32: { auto* m=DataObjectFactory::create<DenseMatrix<uint32_t>>(rows, cols, false); parse_numeric_rows_parallel<uint32_t>(base,rowStarts,rows,cols,delim,m->getValues(),threads,noTrim);res=m; return; }
            case ValueTypeCode::UI8:  { auto* m=DataObjectFactory::create<DenseMatrix<uint8_t>>(rows, cols, false);  parse_numeric_rows_parallel<uint8_t>(base,rowStarts,rows,cols,delim,m->getValues(),threads,noTrim); res=m; return; }
            default:                  { const char* p=base+dataStart; const char* e=base+n; auto* m=DataObjectFactory::create<DenseMatrix<std::string>>(rows, cols, false); parse_string_single(p,e,rows,cols,delim,m->getValues(),noTrim); res=m; return; }
        }
    }
}

// ---------- MATRIX WRITER ----------

static void csv_write_impl(const Structure* matrix, const FileMetaData& /*fmd*/, const char* filename, IOOptions& opts, DaphneContext* /*ctx*/)
{
    char delim = ',';
    if (auto it = opts.extra.find("delimiter"); it != opts.extra.end()) {
        const std::string &d = it->second;
        if (d.size() != 1) throw std::runtime_error("csv_write: 'delimiter' must be a single character");
        delim = d[0];
    }

    FILE* f = std::fopen(filename, "wb");
    if (!f) throw std::runtime_error(std::string("Failed to open for writing: ") + filename);
    FastCSVOut out(f);

    auto write_rows = [&](auto* m) {
        using T = std::decay_t<decltype(*m->getValues())>;
        const size_t R = m->getNumRows(), C = m->getNumCols();
        const T* __restrict v = m->getValues();
        const T* p = v;
        for (size_t i=0;i<R;++i) {
            for (size_t j=0;j<C;++j,++p) {
                if constexpr (std::is_same_v<T, std::string>) {
                    out.write(p->data(), p->size());
                } else if constexpr (std::is_integral_v<T>) {
                    out.append_num(*p);
                } else if constexpr (std::is_floating_point_v<T>) {
                    out.append_num(*p);
                } else {
                    static_assert(!sizeof(T*), "csv_write: unsupported element type");
                }
                if (j + 1 < C) out.putc(delim);
            }
            out.putc('\n');
        }
    };

    if (auto *m = dynamic_cast<const DenseMatrix<double>*>(matrix))      write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<float>*>(matrix))       write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<int64_t>*>(matrix))     write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<int32_t>*>(matrix))     write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<int8_t>*>(matrix))      write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<uint64_t>*>(matrix))    write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<uint32_t>*>(matrix))    write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<uint8_t>*>(matrix))     write_rows(m);
    else if (auto *m = dynamic_cast<const DenseMatrix<std::string>*>(matrix)) write_rows(m);
    else { std::fclose(f); throw std::runtime_error("csv_write: unsupported matrix type"); }

    out.flush(); std::fclose(f);
}

// ---------- FRAME WRITER ----------

static void csv_write_frame_impl(const Frame* fr,
                     const FileMetaData& fmd,
                     const char* filename,
                     IOOptions& opts,
                     DaphneContext* /*ctx*/)
{
    if (!fr) throw std::runtime_error("csv_write_frame: null frame");
    const size_t R = fr->getNumRows(), C = fr->getNumCols();
    if (C == 0) throw std::runtime_error("csv_write_frame: zero columns");

    bool hasHeader = false;
    char delim = ',';
    if (auto it = opts.extra.find("hasHeader"); it != opts.extra.end())
        hasHeader = (it->second == "true" || it->second == "1");
    if (auto it = opts.extra.find("delimiter"); it != opts.extra.end()) {
        const std::string& d = it->second;
        if (d.size() != 1) throw std::runtime_error("csv_write_frame: 'delimiter' must be a single character");
        delim = d[0];
    }

    FILE* f = std::fopen(filename, "wb");
    if (!f) throw std::runtime_error(std::string("csv_write_frame: failed to open: ") + filename);
    FastCSVOut out(f);

    if (hasHeader) {
        if (fmd.labels.size() < C)
            throw std::runtime_error("csv_write_frame: hasHeader=true but labels < numCols");
        for (size_t c=0;c<C;++c) {
            out.write(fmd.labels[c].data(), fmd.labels[c].size());
            if (c + 1 < C) out.putc(delim);
        }
        out.putc('\n');
    }

    std::vector<const void*> col(C, nullptr);
    std::vector<ValueTypeCode> typ(C);
    for (size_t c=0;c<C;++c) {
        col[c] = fr->getColumnRaw(c);
        typ[c] = (fmd.schema.size() > c) ? fmd.schema[c] : fr->getColumnType(c);
    }

    for (size_t r=0;r<R;++r) {
        for (size_t c=0;c<C;++c) {
            const void* base = col[c];
            switch (typ[c]) {
                case ValueTypeCode::SI8:  out.append_num(static_cast<int>(reinterpret_cast<const int8_t *>(base)[r])); break;
                case ValueTypeCode::SI32: out.append_num(reinterpret_cast<const int32_t*>(base)[r]); break;
                case ValueTypeCode::SI64: out.append_num(reinterpret_cast<const int64_t*>(base)[r]); break;
                case ValueTypeCode::UI8:  out.append_num(static_cast<unsigned int>(reinterpret_cast<const uint8_t *>(base)[r])); break;
                case ValueTypeCode::UI32: out.append_num(reinterpret_cast<const uint32_t*>(base)[r]); break;
                case ValueTypeCode::UI64: out.append_num(reinterpret_cast<const uint64_t*>(base)[r]); break;
                case ValueTypeCode::F32:  out.append_num(reinterpret_cast<const float  *>(base)[r]); break;
                case ValueTypeCode::F64:  out.append_num(reinterpret_cast<const double *>(base)[r]); break;
                case ValueTypeCode::STR: {
                    const auto& s = reinterpret_cast<const std::string*>(base)[r];
                    out.write(s.data(), s.size());
                    break;
                }
                default:
                    std::fclose(f);
                    throw std::runtime_error("csv_write_frame: unknown value type");
            }
            if (c + 1 < C) out.putc(delim);
        }
        out.putc('\n');
    }

    out.flush(); std::fclose(f);
}
}



// export macro
#if defined(__GNUC__) || defined(__clang__)
#  define DAPHNE_PLUGIN_API __attribute__((visibility("default")))
#else
#  define DAPHNE_PLUGIN_API
#endif

// Only the 4 entry points use extern "C"
extern "C" DAPHNE_PLUGIN_API
void csv_read(Structure*& res,
              const FileMetaData& fmd,
              const char* filename,
              IOOptions& opts,
              DaphneContext* ctx) {
  csv_read_impl(res, fmd, filename, opts, ctx);  // 3) thin wrapper
}

extern "C" DAPHNE_PLUGIN_API
void csv_read_frame(Frame*& outFrame,
                    const FileMetaData& fmd,
                    const char* filename,
                    IOOptions& opts,
                    DaphneContext* ctx) {
  csv_read_frame_impl(outFrame, fmd, filename, opts, ctx);
}

extern "C" DAPHNE_PLUGIN_API
void csv_write(const Structure* matrix,
               const FileMetaData& fmd,
               const char* filename,
               IOOptions& opts,
               DaphneContext* ctx) {
  csv_write_impl(matrix, fmd, filename, opts, ctx);
}

extern "C" DAPHNE_PLUGIN_API
void csv_write_frame(const Frame* fr,
                     const FileMetaData& fmd,
                     const char* filename,
                     IOOptions& opts,
                     DaphneContext* ctx) {
  csv_write_frame_impl(fr, fmd, filename, opts, ctx);
}