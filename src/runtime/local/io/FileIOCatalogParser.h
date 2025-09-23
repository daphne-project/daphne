#pragma once
#include "runtime/local/io/FileIORegistry.h"
#include <nlohmannjson/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <dlfcn.h>
#include <runtime/local/datastructures/Frame.h>
#include <string>

/**
 * @brief Parses a JSON catalog of I/O plugins, dynamically loads each shared
 * library, discovers its reader/writer functions by name, and registers them
 * directly in FileIORegistry (no plugin-side registration required).
 * JSON format:
 * [
 *   {
 *     "extension": ".csv",
 *     "readerFuncName": "csv_read",
 *     "writerFuncName": "csv_write",
 *     "libPath": "libcsvio.so"
 *   },
 *   ...
 * ]
 */
class FileIOCatalogParser {
public:
    FileIOCatalogParser() = default;

    /**
     * Parses the given JSON file and registers each plugin's reader & writer.
     * @param filePath Path to the catalog JSON
     */
    void parseFileIOCatalog(const std::string &filePath, FileIORegistry &registry) const;
};

inline void FileIOCatalogParser::parseFileIOCatalog(
    const std::string &filePath, FileIORegistry &registry) const
{
    namespace fs = std::filesystem;
    fs::path dir = fs::path(filePath).parent_path();

    std::ifstream in(filePath);
    if(!in.good())
        throw std::runtime_error("Could not open I/O catalog: " + filePath);

    // Parse JSON array of plugin entries
    nlohmann::json jsonData = nlohmann::json::parse(in);
    for(const auto &entry : jsonData) {
        // Read metadata
        const std::string ext      = entry.at("extension").get<std::string>();
        const std::string rdrName  = entry.at("readerFuncName").get<std::string>();
        const std::string wtrName  = entry.at("writerFuncName").get<std::string>();
        const std::string libRel   = entry.at("libPath").get<std::string>();
        const std::string libPath  = (dir / libRel).string();
        const std::string engine = entry.value("engine", "default");
        const int priority       = entry.value("priority", 0);
        const std::string typeName  = entry.value("type", "Frame");

        // Map typeName string to actual type_info
        IODataType typeHash;
        if(typeName == "Frame") {
            typeHash = FRAME;
        } else if(typeName == "DenseMatrix") {
            typeHash = DENSEMATRIX;
        } else if(typeName == "CSRMatrix") {
            typeHash = CSRMATRIX;
        } else {
            throw std::runtime_error("Unknown type in I/O catalog: " + typeName);
        }

        IOOptions opts;
        if(auto it = entry.find("options"); it != entry.end()) {
            for(auto jt = it->begin(); jt != it->end(); ++jt) {
                // Each key/value in JSON becomes a string→string pair
                // e.g. "delimiter":"", "hasHeader":"", etc.
                opts.extra[jt.key()] = jt.value().get<std::string>();
            }
        }

        registry.registerLazy(ext, typeHash, libPath, rdrName, wtrName, opts, engine, priority);


        // Note: keep 'handle' loaded for process lifetime to preserve symbols
    }
}
