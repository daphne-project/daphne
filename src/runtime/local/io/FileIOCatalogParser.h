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
#include <runtime/local/io/FileIOCatalog.h>

#include <nlohmannjson/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <dlfcn.h>

/**
 * @brief Parses a JSON catalog of I/O plugins, discovers its reader/writer functions by name, and registers them as
 * lazy
 */
class FileIOCatalogParser {
  public:
    FileIOCatalogParser() = default;

    /**
     * Parses the given JSON file and registers each plugin's reader & writer.
     * @param filePath Path to the catalog JSON
     */
    void parseFileIOCatalog(const std::string &filePath, FileIOCatalog &catalog, int64_t priority) const;
};

inline void FileIOCatalogParser::parseFileIOCatalog(const std::string &filePath, FileIOCatalog &catalog,
                                                    int64_t priority) const {
    std::filesystem::path dir = std::filesystem::path(filePath).parent_path();
    try {
        std::ifstream in(filePath);
        if (!in.good())
            throw std::runtime_error("could not open file for reading");

        // Parse JSON array of plugin entries
        nlohmann::json jsonData = nlohmann::json::parse(in);
        for (const auto &entry : jsonData) {
            // Read metadata
            const std::string ext = entry.at("extension").get<std::string>();
            const std::string rdrName = entry.at("readerFuncName").get<std::string>();
            const std::string wtrName = entry.at("writerFuncName").get<std::string>();
            const std::string libRel = entry.at("libPath").get<std::string>();
            const std::string libPath = (dir / libRel).string();
            const std::string engine = entry.value("engine", "default");

            const std::string typeName = entry.value("type", "Frame");

            // Map typeName string to actual type_info
            PhyDataType dt;
            if (typeName == "Frame")
                dt = PhyDataType::FRAME;
            else if (typeName == "DenseMatrix")
                dt = PhyDataType::DENSEMATRIX;
            else if (typeName == "CSRMatrix")
                dt = PhyDataType::CSRMATRIX;
            else
                throw std::runtime_error("unknown data type in I/O catalog: `" + typeName + "`");

            IOOptions opts;
            if (auto it = entry.find("options"); it != entry.end())
                for (auto jt = it->begin(); jt != it->end(); ++jt) {
                    // The name "engine" is not allowed for extension-specific options because this key is used to
                    // optionally select a particular extension.
                    if (jt.key() == "engine")
                        throw std::runtime_error("an option of a file IO extension must not be called 'engine'");
                    opts[jt.key()] = jt.value().get<std::string>();
                }

            catalog.registerLazy(ext, dt, engine, priority, opts, libPath, rdrName, wtrName);
        }
    } catch (std::exception &e) {
        throw std::runtime_error("error while parsing I/O catalog file `" + filePath + "`: " + e.what());
    }
}
