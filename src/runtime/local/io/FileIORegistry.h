#pragma once
#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
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

    // Lookup reader (throws if missing)
    GenericReader getReader(const std::string &ext, IODataType dt) {
        std::lock_guard<std::mutex> lk(mtx);
        return readers.at({ext, (size_t)dt});
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

    // Lookup writer (throws if missing)
    GenericWriter getWriter(const std::string &ext, IODataType dt) {
        std::lock_guard<std::mutex> lk(mtx);
        return writers.at({ext, (size_t)dt});
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
    }


private:

    mutable std::mutex mtx;
    std::map<std::pair<std::string, size_t>, GenericReader> readers;
    std::map<std::pair<std::string, size_t>, GenericWriter> writers;
    std::map<std::pair<std::string, size_t>, IOOptions> optionsMap;

};