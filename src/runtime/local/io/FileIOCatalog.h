/*
 * Copyright 2026 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <runtime/local/datastructures/Frame.h>

#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

#include <cstddef>
#include <dlfcn.h>

struct FileMetaData;
class DaphneContext;

enum class PhyDataType { FRAME, DENSEMATRIX, CSRMATRIX };

// Flexible options passed to readers/writers, stored externally
using IOOptions = std::map<std::string, std::string>;

// Describes an IO extension in terms of its shared lib, reader/writer function names, and options.
struct IOExtensionSpec {
    std::string libPath;
    std::string readerSymbol; // "" if none
    std::string writerSymbol; // "" if none
    IOOptions opts;
};

// Generic reader signature.
using GenericReader = std::function<void(void *res, const FileMetaData &fmd, const char *filename,
                                         const IOOptions &opts, DaphneContext *ctx)>;

// Generic writer signature.
using GenericWriter = std::function<void(const void *data, const FileMetaData &fmd, const char *filename,
                                         const IOOptions &opts, DaphneContext *ctx)>;

class FileIOCatalog {
  public:
    FileIOCatalog() = default;

    static FileIOCatalog &instance() {
        static FileIOCatalog inst;
        return inst;
    }

    void registerIOExtension(const std::string &ext, PhyDataType dt, const std::string &engine, int priority,
                             const IOOptions &opts, const std::string &libPath, const std::string &readerSymbol,
                             const std::string &writerSymbol) {
        Key4 k{ext, engine, (size_t)dt, priority};

        // Reject if anything already exists for this full key
        if (extSpecs.find(k) != extSpecs.end() || readers.find(k) != readers.end() || writers.find(k) != writers.end())
            // TODO Print the human-readable name of the data type, not the numeric code.
            throw std::runtime_error("duplicate plug-in registration for filename extension `" + ext + "`, engine `" +
                                     engine + "`, data type `" + std::to_string(static_cast<int>(dt)) +
                                     "` and priority " + std::to_string(priority));

        extSpecs[k] = IOExtensionSpec{libPath, readerSymbol, writerSymbol, opts};
        optionsMap[k] = opts; // defaults visible pre-load
    }

    // ---------- Lookup (engine optional; highest priority wins) ----------
    GenericReader getReader(const std::string &ext, PhyDataType dt, const std::string &engine /* may be "" */) {
        const Key4 *bestLoaded = findBestKey(readers, ext, (size_t)dt, engine);
        const Key4 *bestNonLoaded = findBestKey(extSpecs, ext, (size_t)dt, engine);
        const Key4 *best = nullptr;

        // Error, if neither a loaded nor a non-loaded reader exists.
        if (!bestLoaded && !bestNonLoaded)
            throw std::runtime_error("no suitable reader found in the catalog");

        // Use the best non-loaded reader, if no loaded reader exist or if the priority of the best loaded reader is
        // lower than that of the best non-loaded reader.
        if (!bestLoaded || std::get<3>(*bestLoaded) < std::get<3>(*bestNonLoaded))
            best = bestNonLoaded;
        // Use the best loaded reader, otherwise.
        else
            best = bestLoaded;

        return ensureReaderLoaded(*best);
    }

    GenericWriter getWriter(const std::string &ext, PhyDataType dt, const std::string &engine /* may be "" */) {
        const Key4 *bestLoaded = findBestKey(writers, ext, (size_t)dt, engine);
        const Key4 *bestNonLoaded = findBestKey(extSpecs, ext, (size_t)dt, engine);
        const Key4 *best = nullptr;

        // Error, if neither a loaded nor a non-loaded writer exists.
        if (!bestLoaded && !bestNonLoaded)
            throw std::runtime_error("no suitable writer found in the catalog");

        // Use the best non-loaded writer, if no loaded writer exist or if the priority of the best loaded writer is
        // lower than that of the best non-loaded writer.
        if (!bestLoaded || std::get<3>(*bestLoaded) < std::get<3>(*bestNonLoaded))
            best = bestNonLoaded;
        // Use the best loaded writer, otherwise.
        else
            best = bestLoaded;

        return ensureWriterLoaded(*best);
    }

    // ---------- Options helpers ----------
    // Returns the IOOptions bound to the selected impl (best by engine/priority)
    const IOOptions &getOptions(const std::string &ext, PhyDataType dt, const std::string &engine /* may be "" */) {
        const Key4 *best = findBestKey(optionsMap, ext, (size_t)dt, engine);
        if (!best)
            throw std::runtime_error("no suitable options found in the catalog");
        return optionsMap.at(*best);
    }

    // Optional: expose all (debugging)
    std::map<std::tuple<std::string, std::string, size_t, int>, IOOptions> getAllOptions() const { return optionsMap; }
    std::map<std::tuple<std::string, std::string, size_t, int>, GenericReader> getAllReaders() const { return readers; }
    std::map<std::tuple<std::string, std::string, size_t, int>, GenericWriter> getAllWriters() const { return writers; }

    // ---------- Copy/assign & clear ----------
    FileIOCatalog(const FileIOCatalog &other) {
        readers = other.readers;
        writers = other.writers;
        optionsMap = other.optionsMap;
        extSpecs = other.extSpecs;
        libHandles = other.libHandles; // shallow copy of handles is fine (process-global)
    }

    FileIOCatalog &operator=(const FileIOCatalog &other) {
        if (this != &other) {
            readers = other.readers;
            writers = other.writers;
            optionsMap = other.optionsMap;
            extSpecs = other.extSpecs;
            libHandles = other.libHandles;
        }
        return *this;
    }

    void dumpReaders(std::ostream &os = std::cerr) const {
        auto ioTypeName = [](PhyDataType t) {
            switch (t) {
            case PhyDataType::FRAME:
                return "Frame";
            case PhyDataType::DENSEMATRIX:
                return "DenseMatrix";
            case PhyDataType::CSRMATRIX:
                return "CSRMatrix";
            }
            return "?";
        };

        os << "=== FileIOCatalog: Readers ===\n";

        // 1) Loaded readers
        for (const auto &kv : readers) {
            const auto &k = kv.first;
            const std::string &ext = std::get<0>(k);
            const std::string &engine = std::get<1>(k);
            PhyDataType dt = static_cast<PhyDataType>(std::get<2>(k));
            int prio = std::get<3>(k);
            os << "  [    loaded] ext='" << ext << "'  dt=" << ioTypeName(dt) << "  engine='" << engine
               << "'  priority=" << prio << "\n";
        }

        // 2) Non-loaded reader facets that aren't loaded yet
        for (const auto &kv : extSpecs) {
            const auto &k = kv.first;
            const auto &spec = kv.second;
            if (spec.readerSymbol.empty())
                continue; // writer-only; skip
            if (readers.find(k) != readers.end())
                continue; // already listed as loaded

            const std::string &ext = std::get<0>(k);
            const std::string &engine = std::get<1>(k);
            PhyDataType dt = static_cast<PhyDataType>(std::get<2>(k));
            int prio = std::get<3>(k);
            os << "  [non-loaded] ext='" << ext << "'  dt=" << ioTypeName(dt) << "  engine='" << engine
               << "'  priority=" << prio << "  symbol='" << spec.readerSymbol << "'"
               << "  lib='" << spec.libPath << "'"
               << "\n";
        }
        os << std::flush;
    }

    void dumpWriters(std::ostream &os = std::cerr) const {
        auto ioTypeName = [](PhyDataType t) {
            switch (t) {
            case PhyDataType::FRAME:
                return "Frame";
            case PhyDataType::DENSEMATRIX:
                return "DenseMatrix";
            case PhyDataType::CSRMATRIX:
                return "CSRMatrix";
            }
            return "?";
        };

        os << "=== FileIOCatalog: Writers ===\n";

        // 1) Loaded writers
        for (const auto &kv : writers) {
            const auto &k = kv.first;
            const std::string &ext = std::get<0>(k);
            const std::string &engine = std::get<1>(k);
            PhyDataType dt = static_cast<PhyDataType>(std::get<2>(k));
            int prio = std::get<3>(k);
            os << "  [    loaded] ext='" << ext << "'  dt=" << ioTypeName(dt) << "  engine='" << engine
               << "'  priority=" << prio << "\n";
        }

        // 2) Non-loaded writer facets that aren't loaded yet
        for (const auto &kv : extSpecs) {
            const auto &k = kv.first;
            const auto &spec = kv.second;
            if (spec.writerSymbol.empty())
                continue; // reader-only; skip
            if (writers.find(k) != writers.end())
                continue; // already listed as loaded

            const std::string &ext = std::get<0>(k);
            const std::string &engine = std::get<1>(k);
            PhyDataType dt = static_cast<PhyDataType>(std::get<2>(k));
            int prio = std::get<3>(k);
            os << "  [non-loaded] ext='" << ext << "'  dt=" << ioTypeName(dt) << "  engine='" << engine
               << "'  priority=" << prio << "  symbol='" << spec.writerSymbol << "'"
               << "  lib='" << spec.libPath << "'"
               << "\n";
        }
        os << std::flush;
    }

  private:
    using Key4 = std::tuple<std::string /*ext*/, std::string /*engine*/, size_t /*dt*/, int /*priority*/>;

    template <class MapT>
    static const Key4 *findBestKey(const MapT &m, const std::string &ext, size_t dt,
                                   const std::string &engine /* "" = any */) {
        const Key4 *best = nullptr;
        for (auto &kv : m) {
            const auto &k = kv.first;
            const auto &kExt = std::get<0>(k);
            const auto &kEngine = std::get<1>(k);
            const auto kDt = std::get<2>(k);
            const auto kPrio = std::get<3>(k);
            if (kExt == ext && kDt == dt && (engine.empty() || kEngine == engine)) {
                if (!best || std::get<3>(*best) < kPrio)
                    best = &k;
            }
        }
        return best;
    }

    GenericReader ensureReaderLoaded(const Key4 &k) {
        if (auto it = readers.find(k); it != readers.end())
            return it->second;

        auto itL = extSpecs.find(k);
        if (itL == extSpecs.end())
            throw std::runtime_error("no reader for requested key");

        const auto &spec = itL->second;
        if (spec.readerSymbol.empty())
            throw std::runtime_error("no reader symbol specified");

        void *&h = libHandles[spec.libPath];
        if (!h) {
            h = dlopen(spec.libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (!h)
                throw std::runtime_error("dlopen() failed for `" + spec.libPath + "`: " + std::string(dlerror()));
        }
        using ReaderFn = void (*)(void *, const FileMetaData &, const char *, const IOOptions &, DaphneContext *);
        void *sym = dlsym(h, spec.readerSymbol.c_str());
        if (!sym)
            throw std::runtime_error("dlsym() failed for ''" + spec.readerSymbol + "` in `" + spec.libPath +
                                     "`: " + std::string(dlerror()));

        readers[k] = GenericReader(reinterpret_cast<ReaderFn>(sym));
        return readers[k];
    }

    GenericWriter ensureWriterLoaded(const Key4 &k) {
        if (auto it = writers.find(k); it != writers.end())
            return it->second;

        auto itL = extSpecs.find(k);
        if (itL == extSpecs.end())
            throw std::runtime_error("no writer for requested key");

        const auto &spec = itL->second;
        if (spec.writerSymbol.empty())
            throw std::runtime_error("no writer symbol specified");

        void *&h = libHandles[spec.libPath];
        if (!h) {
            h = dlopen(spec.libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (!h)
                throw std::runtime_error("dlopen() failed for `" + spec.libPath + "`: " + std::string(dlerror()));
        }
        using WriterFn = void (*)(const void *, const FileMetaData &, const char *, const IOOptions &, DaphneContext *);
        void *sym = dlsym(h, spec.writerSymbol.c_str());
        if (!sym)
            throw std::runtime_error("dlsym() failed for `" + spec.writerSymbol + "` in `" + spec.libPath +
                                     "`: " + std::string(dlerror()));

        writers[k] = GenericWriter(reinterpret_cast<WriterFn>(sym));
        return writers[k];
    }

    // Core maps now keyed by (ext, engine, dt, priority)
    std::map<Key4, GenericReader> readers;
    std::map<Key4, GenericWriter> writers;
    std::map<Key4, IOOptions> optionsMap;
    std::map<Key4, IOExtensionSpec> extSpecs;

    std::map<std::string, void *> libHandles;
};

// Helper: Merge a Frame* of column-label → single-row-value into IOOptions
IOOptions mergeOptions(const std::string &ext, PhyDataType dt, const std::string &engine, const Frame *opts,
                       FileIOCatalog &cat);

// Extract "engine" (and ignore "priority") from the options Frame if present.
// Returns "" if not provided (so catalog picks highest-priority default).
std::string extractEngine(const Frame *opts);