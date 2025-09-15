#pragma once
#include <cstddef>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

struct FileMetaData;
class DaphneContext;

// Supported I/O data object categories
enum IODataType {
    FRAME,
    DENSEMATRIX,
    CSRMATRIX
};

// Flexible options passed to readers/writers, stored externally
struct IOOptions {
    std::map<std::string, std::string> extra; // plugin-specific flags
};

struct LazySpec {
    std::string libPath;
    std::string readerSymbol; // "" if none
    std::string writerSymbol; // "" if none
    IOOptions   opts;
};

// Generic reader signature (void* -> DTRes*) including options in the callback
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

    // Register a reader callback for extension+dataType
    // The callback is expected to capture any IOOptions in its lambda closure
    void registerReader(const std::string &ext,
                        IODataType  dt,
                        const IOOptions  &opts,
                        GenericReader  fn) {
        std::lock_guard<std::mutex> lk(mtx);
        readers[{ext, (size_t)dt}] = std::move(fn);
        optionsMap[{ext,(size_t)dt}] = opts;
    }

    GenericReader getReader(const std::string &ext, IODataType dt) {
        std::lock_guard<std::mutex> lk(mtx);
        auto k = std::make_pair(ext, (size_t)dt);

        // Fast path: already resolved
        if (auto it = readers.find(k); it != readers.end())
            return it->second;

        // Lazy resolve
        if (auto it = lazySpecs.find(k); it != lazySpecs.end()) {
            const auto& spec = it->second;
            if (spec.readerSymbol.empty())
                throw std::runtime_error("No reader symbol specified for " + ext);

            // open (or reuse) the library
            void* &h = libHandles[spec.libPath];
            if (!h) {
                h = dlopen(spec.libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
                if (!h)
                    throw std::runtime_error("dlopen failed for " + spec.libPath + ": " + std::string(dlerror()));
            }

            using ReaderFn = void(*)(void*, const FileMetaData&, const char*, const IOOptions&, DaphneContext*);
            void* sym = dlsym(h, spec.readerSymbol.c_str());
            if (!sym)
                throw std::runtime_error("dlsym failed for " + spec.readerSymbol + " in " + spec.libPath + ": " + std::string(dlerror()));

            readers[k] = GenericReader(reinterpret_cast<ReaderFn>(sym)); // cache
            return readers[k];
        }

        throw std::out_of_range("No reader registered for ext=" + ext);
    }

    GenericWriter getWriter(const std::string &ext, IODataType dt) {
        std::lock_guard<std::mutex> lk(mtx);
        auto k = std::make_pair(ext, (size_t)dt);

        if (auto it = writers.find(k); it != writers.end())
            return it->second;

        if (auto it = lazySpecs.find(k); it != lazySpecs.end()) {
            const auto& spec = it->second;
            if (spec.writerSymbol.empty())
                throw std::runtime_error("No writer symbol specified for " + ext);

            void* &h = libHandles[spec.libPath];
            if (!h) {
                h = dlopen(spec.libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
                if (!h)
                    throw std::runtime_error("dlopen failed for " + spec.libPath + ": " + std::string(dlerror()));
            }

            using WriterFn = void(*)(const void*, const FileMetaData&, const char*, const IOOptions&, DaphneContext*);
            void* sym = dlsym(h, spec.writerSymbol.c_str());
            if (!sym)
                throw std::runtime_error("dlsym failed for " + spec.writerSymbol + " in " + spec.libPath + ": " + std::string(dlerror()));

            writers[k] = GenericWriter(reinterpret_cast<WriterFn>(sym));
            return writers[k];
        }

        throw std::out_of_range("No writer registered for ext=" + ext);
    }


    void registerLazy(const std::string& ext,
                  IODataType dt,
                  const std::string& libPath,
                  const std::string& readerSymbol,
                  const std::string& writerSymbol,
                  const IOOptions&  opts) {
        std::lock_guard<std::mutex> lk(mtx);
        auto k = std::make_pair(ext, (size_t)dt);
        lazySpecs[k] = LazySpec{libPath, readerSymbol, writerSymbol, opts};
        optionsMap[k] = opts; // defaults visible even before load  
    }

    // Register a writer callback for extension+dataType
    void registerWriter(const std::string &ext,
                        IODataType           dt,
                        const IOOptions  &opts,
                        GenericWriter        fn) {
        std::lock_guard<std::mutex> lk(mtx);
        writers[{ext, (size_t)dt}] = std::move(fn);
        optionsMap[{ext,(size_t)dt}] = opts;
    }

     const IOOptions &getOptions(const std::string &ext, IODataType dt) {
        std::lock_guard<std::mutex> lk(mtx);
        return optionsMap.at({ext, (size_t)dt});
    }

    // Optional: expose all options for debugging
    std::map<std::pair<std::string, size_t>, IOOptions> getAllOptions()  {
        std::lock_guard<std::mutex> lk(mtx);
        return optionsMap;
    }

    std::map<std::pair<std::string, size_t>, GenericReader> getAllReaders() const {
        std::lock_guard<std::mutex> lk(mtx);
        return readers;
    }

        
    std::map<std::pair<std::string, size_t>, GenericWriter> getAllWriters() const {
        std::lock_guard<std::mutex> lk(mtx);
        return writers;
    }

    FileIORegistry( const FileIORegistry& other) {
        std::lock_guard<std::mutex> lk(other.mtx);
        readers = other.readers;
        writers = other.writers;
        optionsMap = other.optionsMap;
    }

    FileIORegistry& operator=( const FileIORegistry& other) {
        if (this != &other) {
            std::unique_lock<std::mutex> lk1(mtx, std::defer_lock);
            std::unique_lock<std::mutex> lk2(other.mtx, std::defer_lock);
            std::lock(lk1, lk2);  // Lock both without deadlock

            readers = other.readers;
            writers = other.writers;
            optionsMap = other.optionsMap;
        }
        return *this;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx);
        readers.clear();
        writers.clear();
        optionsMap.clear();
        lazySpecs.clear();
        libHandles.clear();
    }


private:

    mutable std::mutex mtx;
    std::map<std::pair<std::string, size_t>, GenericReader> readers;
    std::map<std::pair<std::string, size_t>, GenericWriter> writers;
    std::map<std::pair<std::string, size_t>, IOOptions> optionsMap;

    std::map<std::pair<std::string, size_t>, LazySpec> lazySpecs;
    std::map<std::string, void*> libHandles; // keep handles alive
};