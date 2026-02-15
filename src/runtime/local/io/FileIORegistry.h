#pragma once
#include <cstddef>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

struct FileMetaData;
class DaphneContext;

enum IODataType {
    FRAME,
    DENSEMATRIX,
    CSRMATRIX
};

// Flexible options passed to readers/writers, stored externally
struct IOOptions {
    std::map<std::string, std::string> extra; // plugin-specific flags
};

// Lazy spec describing how to load a symbol from a shared lib
struct LazySpec {
    std::string libPath;
    std::string readerSymbol; // "" if none
    std::string writerSymbol; // "" if none
    IOOptions   opts;
};

// Generic reader signature including options
using GenericReader = std::function<
    void(void*              res,
         const FileMetaData &fmd,
         const char*         filename,
         const IOOptions&    opts,
         DaphneContext*      ctx)>;

// Generic writer signature including options
using GenericWriter = std::function<
    void(const void*        data,
         const FileMetaData &fmd,
         const char*         filename,
         const IOOptions&    opts,
         DaphneContext*      ctx)>;

class FileIORegistry {
public:
    FileIORegistry() = default;

    static FileIORegistry &instance() {
        static FileIORegistry inst;
        return inst;
    }

    // ---------- Registration (engine + priority explicit) ----------
    void registerReader(const std::string& ext, IODataType dt,
                    const std::string& engine, int priority,
                    const IOOptions& opts, GenericReader fn) {
        std::lock_guard<std::mutex> lk(mtx);
        Key4 k{ext, engine, (size_t)dt, priority};
        if (readers.count(k) ||
            (lazySpecs.count(k) && !lazySpecs.at(k).readerSymbol.empty()))
            throw std::runtime_error("Duplicate reader for ext=" + ext +
                " engine=" + engine + " dt=" + std::to_string(dt) +
                " priority=" + std::to_string(priority));
        readers[k] = std::move(fn);
        optionsMap[k] = opts;
    }


    void registerWriter(const std::string &ext, IODataType dt,
                        const std::string &engine, int priority,
                        const IOOptions &opts, GenericWriter fn) {
        std::lock_guard<std::mutex> lk(mtx);
        Key4 k{ext, engine, (size_t)dt, priority};
        if (writers.count(k) || (lazySpecs.count(k) && !lazySpecs.at(k).writerSymbol.empty()))
            throw std::runtime_error("Duplicate writer for ext=" + ext +
                " engine=" + engine + " dt=" + std::to_string(dt) +
                " priority=" + std::to_string(priority));
        writers[k] = std::move(fn);
        optionsMap[k] = opts;
    }

    // Back-compat: single impl (engine="default", priority=0)
    void registerReader(const std::string &ext,
                        IODataType dt,
                        const IOOptions &opts,
                        GenericReader fn) {
        registerReader(ext, dt, "default", 0, opts, std::move(fn));
    }

    void registerWriter(const std::string &ext,
                        IODataType dt,
                        const IOOptions &opts,
                        GenericWriter fn) {
        registerWriter(ext, dt, "default", 0, opts, std::move(fn));
    }

    void registerLazy(const std::string& ext,
                    IODataType dt,
                    const std::string& libPath,
                    const std::string& readerSymbol,
                    const std::string& writerSymbol,
                    const IOOptions&  opts,
                    const std::string& engine,
                    int priority) {
        std::lock_guard<std::mutex> lk(mtx);
        Key4 k{ext, engine, (size_t)dt, priority};

        // Reject if anything already exists for this full key
        if (lazySpecs.find(k) != lazySpecs.end() ||
            readers.find(k)    != readers.end()   ||
            writers.find(k)    != writers.end()) {
            throw std::runtime_error(
                "registerLazy: duplicate registration for (ext=" + ext +
                ", engine=" + engine + ", dt=" + std::to_string(dt) +
                ", prio=" + std::to_string(priority) + ")");
        }

        lazySpecs[k]  = LazySpec{libPath, readerSymbol, writerSymbol, opts};
        optionsMap[k] = opts; // defaults visible pre-load
    }


    // ---------- Lookup (engine optional; highest priority wins) ----------
    GenericReader getReader(const std::string &ext, IODataType dt,
                            const std::string &engine /* may be "" */) {
        std::lock_guard<std::mutex> lk(mtx);

        const Key4* best = findBestKey(readers, ext, (size_t)dt, engine);
        if (!best) best   = findBestKey(lazySpecs, ext, (size_t)dt, engine);
        if (!best) throw std::out_of_range("No suitable reader found in the registry");

        return ensureReaderLoaded(*best);
    }

    GenericWriter getWriter(const std::string &ext, IODataType dt,
                            const std::string &engine /* may be "" */) {
        std::lock_guard<std::mutex> lk(mtx);

        const Key4* best = findBestKey(writers, ext, (size_t)dt, engine);
        if (!best) best   = findBestKey(lazySpecs, ext, (size_t)dt, engine);
        if (!best) throw std::out_of_range("No suitable writer found in the registry");

        return ensureWriterLoaded(*best);
    }

    // Back-compat overloads (default engine selection)
    GenericReader getReader(const std::string &ext, IODataType dt) {
        return getReader(ext, dt, "" /* default selection */);
    }
    GenericWriter getWriter(const std::string &ext, IODataType dt) {
        return getWriter(ext, dt, "" /* default selection */);
    }

    // ---------- Options helpers ----------
    // Returns the IOOptions bound to the selected impl (best by engine/priority)
    const IOOptions &getOptions(const std::string &ext, IODataType dt,
                                const std::string &engine /* may be "" */) {
        std::lock_guard<std::mutex> lk(mtx);

        const Key4* best = findBestKey(optionsMap, ext, (size_t)dt, engine);
        if (!best) throw std::out_of_range("No suitable options found in the registry");
        return optionsMap.at(*best);
    }

    // Optional: expose all (debugging)
    std::map<std::tuple<std::string,std::string,size_t,int>, IOOptions> getAllOptions() const {
        std::lock_guard<std::mutex> lk(mtx);
        return optionsMap;
    }
    std::map<std::tuple<std::string,std::string,size_t,int>, GenericReader> getAllReaders() const {
        std::lock_guard<std::mutex> lk(mtx);
        return readers;
    }
    std::map<std::tuple<std::string,std::string,size_t,int>, GenericWriter> getAllWriters() const {
        std::lock_guard<std::mutex> lk(mtx);
        return writers;
    }

    // ---------- Copy/assign & clear ----------
    FileIORegistry(const FileIORegistry& other) {
        std::lock_guard<std::mutex> lk(other.mtx);
        readers    = other.readers;
        writers    = other.writers;
        optionsMap = other.optionsMap;
        lazySpecs  = other.lazySpecs;
        libHandles = other.libHandles; // shallow copy of handles is fine (process-global)
    }

    FileIORegistry& operator=(const FileIORegistry& other) {
        if (this != &other) {
            std::unique_lock<std::mutex> lk1(mtx, std::defer_lock);
            std::unique_lock<std::mutex> lk2(other.mtx, std::defer_lock);
            std::lock(lk1, lk2);
            readers    = other.readers;
            writers    = other.writers;
            optionsMap = other.optionsMap;
            lazySpecs  = other.lazySpecs;
            libHandles = other.libHandles;
        }
        return *this;
    }

    void dumpReaders(std::ostream& os = std::cerr) const {
        std::lock_guard<std::mutex> lk(mtx);

        auto ioTypeName = [](IODataType t) {
            switch (t) {
                case FRAME:       return "Frame";
                case DENSEMATRIX: return "DenseMatrix";
                case CSRMATRIX:   return "CSRMatrix";
            }
            return "?";
        };

        os << "=== FileIORegistry: Readers ===\n";

        // 1) Loaded readers
        for (const auto& kv : readers) {
            const auto& k = kv.first;
            const std::string& ext    = std::get<0>(k);
            const std::string& engine = std::get<1>(k);
            IODataType dt             = static_cast<IODataType>(std::get<2>(k));
            int prio                  = std::get<3>(k);
            os << "  [loaded] ext='" << ext
            << "'  dt=" << ioTypeName(dt)
            << "  engine='" << engine
            << "'  priority=" << prio
            << "\n";
        }

        // 2) Lazy reader facets that aren't loaded yet
        for (const auto& kv : lazySpecs) {
            const auto& k   = kv.first;
            const auto& spec = kv.second;
            if (spec.readerSymbol.empty()) continue;                // writer-only; skip
            if (readers.find(k) != readers.end()) continue;         // already listed as loaded

            const std::string& ext    = std::get<0>(k);
            const std::string& engine = std::get<1>(k);
            IODataType dt             = static_cast<IODataType>(std::get<2>(k));
            int prio                  = std::get<3>(k);
            os << "  [lazy  ] ext='" << ext
            << "'  dt=" << ioTypeName(dt)
            << "  engine='" << engine
            << "'  priority=" << prio
            << "  symbol='" << spec.readerSymbol << "'"
            << "  lib='" << spec.libPath << "'"
            << "\n";
        }
        os << std::flush;
    }

    void dumpWriters(std::ostream& os = std::cerr) const {
        std::lock_guard<std::mutex> lk(mtx);

        auto ioTypeName = [](IODataType t) {
            switch (t) {
                case FRAME:       return "Frame";
                case DENSEMATRIX: return "DenseMatrix";
                case CSRMATRIX:   return "CSRMatrix";
            }
            return "?";
        };

        os << "=== FileIORegistry: Writers ===\n";

        // 1) Loaded writers
        for (const auto& kv : writers) {
            const auto& k = kv.first;
            const std::string& ext    = std::get<0>(k);
            const std::string& engine = std::get<1>(k);
            IODataType dt             = static_cast<IODataType>(std::get<2>(k));
            int prio                  = std::get<3>(k);
            os << "  [loaded] ext='" << ext
            << "'  dt=" << ioTypeName(dt)
            << "  engine='" << engine
            << "'  priority=" << prio
            << "\n";
        }

        // 2) Lazy writer facets that aren't loaded yet
        for (const auto& kv : lazySpecs) {
            const auto& k   = kv.first;
            const auto& spec = kv.second;
            if (spec.writerSymbol.empty()) continue;                // reader-only; skip
            if (writers.find(k) != writers.end()) continue;         // already listed as loaded

            const std::string& ext    = std::get<0>(k);
            const std::string& engine = std::get<1>(k);
            IODataType dt             = static_cast<IODataType>(std::get<2>(k));
            int prio                  = std::get<3>(k);
            os << "  [lazy  ] ext='" << ext
            << "'  dt=" << ioTypeName(dt)
            << "  engine='" << engine
            << "'  priority=" << prio
            << "  symbol='" << spec.writerSymbol << "'"
            << "  lib='" << spec.libPath << "'"
            << "\n";
        }
        os << std::flush;
    }


    // Call this once right after you load BuiltIns.json
    void captureBaseline() {
        std::lock_guard<std::mutex> lk(mtx);
        baseline_readers = readers;
        baseline_writers = writers;
        baseline_options = optionsMap;
        baseline_lazy    = lazySpecs;
        baselineCaptured = true;
    }

    // Reset registry back to the captured baseline (keeps built-ins, drops recent)
    void resetToBaseline() {
        std::lock_guard<std::mutex> lk(mtx);
        if (!baselineCaptured) { readers.clear(); writers.clear(); optionsMap.clear(); lazySpecs.clear(); return; }
        readers    = baseline_readers;
        writers    = baseline_writers;
        optionsMap = baseline_options;
        lazySpecs  = baseline_lazy;
    }

    void clear(){
        readers.clear(); 
        writers.clear(); 
        optionsMap.clear(); 
        lazySpecs.clear();
    }


private:
    using Key4 = std::tuple<std::string /*ext*/,
                            std::string /*engine*/,
                            size_t      /*dt*/,
                            int         /*priority*/>;

    template<class MapT>
    static const Key4* findBestKey(const MapT &m, const std::string &ext, size_t dt, const std::string &engine /* "" = any */) {
        const Key4* best = nullptr;
        for (auto &kv : m) {
            const auto &k = kv.first;
            const auto &kExt    = std::get<0>(k);
            const auto &kEngine = std::get<1>(k);
            const auto  kDt     = std::get<2>(k);
            const auto  kPrio   = std::get<3>(k);
            if (kExt == ext && kDt == dt && (engine.empty() || kEngine == engine)) {
                if (!best || std::get<3>(*best) < kPrio) best = &k;
            }
        }
        return best;
    }

    GenericReader ensureReaderLoaded(const Key4 &k) {
        if (auto it = readers.find(k); it != readers.end())
            return it->second;

        // lazy resolve via spec
        auto itL = lazySpecs.find(k);
        if (itL == lazySpecs.end())
            throw std::out_of_range("No reader (and no lazy spec) for requested key");

        const auto &spec = itL->second;
        if (spec.readerSymbol.empty())
            throw std::runtime_error("No reader symbol specified");

        void* &h = libHandles[spec.libPath];
        if (!h) {
            h = dlopen(spec.libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (!h) throw std::runtime_error("dlopen failed for " + spec.libPath + ": " + std::string(dlerror()));
        }
        using ReaderFn = void(*)(void*, const FileMetaData&, const char*, const IOOptions&, DaphneContext*);
        void* sym = dlsym(h, spec.readerSymbol.c_str());
        if (!sym)
            throw std::runtime_error("dlsym failed for " + spec.readerSymbol + " in " + spec.libPath + ": " + std::string(dlerror()));

        readers[k] = GenericReader(reinterpret_cast<ReaderFn>(sym));
        return readers[k];
    }

    GenericWriter ensureWriterLoaded(const Key4 &k) {
        if (auto it = writers.find(k); it != writers.end())
            return it->second;

        auto itL = lazySpecs.find(k);
        if (itL == lazySpecs.end())
            throw std::out_of_range("No writer (and no lazy spec) for requested key");

        const auto &spec = itL->second;
        if (spec.writerSymbol.empty())
            throw std::runtime_error("No writer symbol specified");

        void* &h = libHandles[spec.libPath];
        if (!h) {
            h = dlopen(spec.libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (!h) throw std::runtime_error("dlopen failed for " + spec.libPath + ": " + std::string(dlerror()));
        }
        using WriterFn = void(*)(const void*, const FileMetaData&, const char*, const IOOptions&, DaphneContext*);
        void* sym = dlsym(h, spec.writerSymbol.c_str());
        if (!sym)
            throw std::runtime_error("dlsym failed for " + spec.writerSymbol + " in " + spec.libPath + ": " + std::string(dlerror()));

        writers[k] = GenericWriter(reinterpret_cast<WriterFn>(sym));
        return writers[k];
    }

private:
    mutable std::mutex mtx;

    // Core maps now keyed by (ext, engine, dt, priority)
    std::map<Key4, GenericReader> readers;
    std::map<Key4, GenericWriter> writers;
    std::map<Key4, IOOptions>     optionsMap;
    std::map<Key4, LazySpec>      lazySpecs;

    std::map<Key4, GenericReader> baseline_readers;
    std::map<Key4, GenericWriter> baseline_writers;
    std::map<Key4, IOOptions>     baseline_options;
    std::map<Key4, LazySpec>      baseline_lazy;
    bool baselineCaptured = false;

    std::map<std::string, void*>  libHandles;
};
