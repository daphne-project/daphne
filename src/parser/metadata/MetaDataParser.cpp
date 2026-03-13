/*
 * Copyright 2021 The DAPHNE Consortium
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

#include <parser/metadata/JsonKeys.h>
#include <parser/metadata/MetaDataParser.h>
#include <runtime/local/datastructures/MncSketch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

static std::vector<std::uint32_t> readU32VecChecked(
    const nlohmann::json &j, const std::string &key,
    std::size_t expectedSize, const std::string &ctx
) {
    if (j.find(key) == j.end())
        throw std::invalid_argument(ctx + " missing key \"" + key + "\"");

    auto v = j.at(key).get<std::vector<std::uint32_t>>();
    if (v.size() != expectedSize) {
        throw std::invalid_argument(
            ctx + " key \"" + key + "\" has size " + std::to_string(v.size()) +
            " but expected " + std::to_string(expectedSize)
        );
    }
    return v;
}

static void recomputeMncSummary(MncSketch &s) {
    // recompute summary stats from hr/hc (useful if meta file omits them or they got stale)
    s.maxHr = 0; s.maxHc = 0;
    s.nnzRows = 0; s.nnzCols = 0;
    s.rowsEq1 = 0; s.colsEq1 = 0;
    s.rowsGtHalf = 0; s.colsGtHalf = 0;

    if (s.hr) {
        for (std::size_t i = 0; i < s.m; i++) {
            auto x = (*s.hr)[i];
            s.maxHr = std::max(s.maxHr, x);
            if (x > 0) s.nnzRows++;
            if (x == 1) s.rowsEq1++;
            if (s.n > 0 && x > s.n / 2) s.rowsGtHalf++;
        }
    }
    if (s.hc) {
        for (std::size_t j = 0; j < s.n; j++) {
            auto x = (*s.hc)[j];
            s.maxHc = std::max(s.maxHc, x);
            if (x > 0) s.nnzCols++;
            if (x == 1) s.colsEq1++;
            if (s.m > 0 && x > s.m / 2) s.colsGtHalf++;
        }
    }
}

static MncSketch readMncSketchFromJson(const nlohmann::json &jf, std::size_t numRows, std::size_t numCols) {
    const std::string ctx = "mncSketch";
    const auto &jm = jf.at(JsonKeys::MNC_SKETCH);
    if (!jm.is_object())
        throw std::invalid_argument(ctx + " must be a JSON object");

    MncSketch s;

    // Use numRows/numCols from metadata as source of truth
    s.m = numRows;
    s.n = numCols;

    // Optional sanity: if m/n exist inside sketch object, verify
    if (jm.find(JsonKeys::MNCKeys::M) != jm.end() && jm.at(JsonKeys::MNCKeys::M).get<std::size_t>() != numRows)
        throw std::invalid_argument(ctx + " dimension mismatch: m");
    if (jm.find(JsonKeys::MNCKeys::N) != jm.end() && jm.at(JsonKeys::MNCKeys::N).get<std::size_t>() != numCols)
        throw std::invalid_argument(ctx + " dimension mismatch: n");

    // Core vectors
    auto hrVec = readU32VecChecked(jm, JsonKeys::MNCKeys::HR, s.m, ctx);
    auto hcVec = readU32VecChecked(jm, JsonKeys::MNCKeys::HC, s.n, ctx);
    s.hr = std::make_shared<std::vector<std::uint32_t>>(std::move(hrVec));
    s.hc = std::make_shared<std::vector<std::uint32_t>>(std::move(hcVec));

    // Extended vectors: must appear together
    const bool hasHer = (jm.find(JsonKeys::MNCKeys::HER) != jm.end());
    const bool hasHec = (jm.find(JsonKeys::MNCKeys::HEC) != jm.end());
    if (hasHer != hasHec)
        throw std::invalid_argument(ctx + " must contain both \"her\" and \"hec\" or neither");

    if (hasHer) {
        auto herVec = readU32VecChecked(jm, JsonKeys::MNCKeys::HER, s.m, ctx);
        auto hecVec = readU32VecChecked(jm, JsonKeys::MNCKeys::HEC, s.n, ctx);
        s.her = std::make_shared<std::vector<std::uint32_t>>(std::move(herVec));
        s.hec = std::make_shared<std::vector<std::uint32_t>>(std::move(hecVec));
    }

    // Optional flags + summary stats
    s.isDiagonal = (jm.find(JsonKeys::MNCKeys::IS_DIAGONAL) != jm.end())
        ? jm.at(JsonKeys::MNCKeys::IS_DIAGONAL).get<bool>()
        : false;

    auto getU32 = [&](const std::string &k, std::uint32_t &out) {
        if (jm.find(k) != jm.end()) out = jm.at(k).get<std::uint32_t>();
    };

    getU32(JsonKeys::MNCKeys::MAX_HR, s.maxHr);
    getU32(JsonKeys::MNCKeys::MAX_HC, s.maxHc);
    getU32(JsonKeys::MNCKeys::NNZ_ROWS, s.nnzRows);
    getU32(JsonKeys::MNCKeys::NNZ_COLS, s.nnzCols);
    getU32(JsonKeys::MNCKeys::ROWS_EQ1, s.rowsEq1);
    getU32(JsonKeys::MNCKeys::COLS_EQ1, s.colsEq1);
    getU32(JsonKeys::MNCKeys::ROWS_GT_HALF, s.rowsGtHalf);
    getU32(JsonKeys::MNCKeys::COLS_GT_HALF, s.colsGtHalf);

    // If some summary fields are missing, recompute them (safe + consistent)
    // (You can always recompute unconditionally if you want.)
    if (s.maxHr == 0 && s.maxHc == 0 && (numRows > 0 || numCols > 0)) {
        recomputeMncSummary(s);
    }

    // Strong sanity check: total nnz must match
    std::uint64_t sumHr = 0, sumHc = 0;
    for (auto x : *s.hr) sumHr += x;
    for (auto x : *s.hc) sumHc += x;
    if (sumHr != sumHc)
        throw std::invalid_argument(ctx + " inconsistent: sum(hr) != sum(hc)");

    return s;
}

static void writeMncSketchToJson(nlohmann::json &json, const MncSketch &s) {
    nlohmann::json jm;
    jm[JsonKeys::MNCKeys::M] = s.m;
    jm[JsonKeys::MNCKeys::N] = s.n;

    jm[JsonKeys::MNCKeys::HR] = (s.hr ? *s.hr : std::vector<std::uint32_t>{});
    jm[JsonKeys::MNCKeys::HC] = (s.hc ? *s.hc : std::vector<std::uint32_t>{});
    
    if (s.her && s.hec) {
        jm[JsonKeys::MNCKeys::HER] = *s.her;
        jm[JsonKeys::MNCKeys::HEC] = *s.hec;
    }

    jm[JsonKeys::MNCKeys::MAX_HR] = s.maxHr;
    jm[JsonKeys::MNCKeys::MAX_HC] = s.maxHc;
    jm[JsonKeys::MNCKeys::NNZ_ROWS] = s.nnzRows;
    jm[JsonKeys::MNCKeys::NNZ_COLS] = s.nnzCols;
    jm[JsonKeys::MNCKeys::ROWS_EQ1] = s.rowsEq1;
    jm[JsonKeys::MNCKeys::COLS_EQ1] = s.colsEq1;
    jm[JsonKeys::MNCKeys::ROWS_GT_HALF] = s.rowsGtHalf;
    jm[JsonKeys::MNCKeys::COLS_GT_HALF] = s.colsGtHalf;

    jm[JsonKeys::MNCKeys::IS_DIAGONAL] = s.isDiagonal;

    json[JsonKeys::MNC_SKETCH] = std::move(jm);
}

FileMetaData MetaDataParser::readMetaData(const std::string &filename_) {
    std::string metaFilename = filename_ + ".meta";
    std::ifstream ifs(metaFilename, std::ios::in);
    if (!ifs.good())
        throw std::runtime_error("Could not open file '" + metaFilename + "' for reading meta data.");
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return MetaDataParser::readMetaDataFromString(buffer.str());
}
FileMetaData MetaDataParser::readMetaDataFromString(const std::string &str) {
    nlohmann::json jf = nlohmann::json::parse(str);

    if (!keyExists(jf, JsonKeys::NUM_ROWS) || !keyExists(jf, JsonKeys::NUM_COLS)) {
        throw std::invalid_argument("A meta data JSON file should always contain \"" + JsonKeys::NUM_ROWS +
                                    "\" and \"" + JsonKeys::NUM_COLS + "\" keys.");
    }

    const size_t numRows = jf.at(JsonKeys::NUM_ROWS).get<size_t>();
    const size_t numCols = jf.at(JsonKeys::NUM_COLS).get<size_t>();
    const bool isHDFS = (keyExists(jf, JsonKeys::HDFS));
    const bool isSingleValueType = !(keyExists(jf, JsonKeys::SCHEMA));
    const ssize_t numNonZeros =
        (keyExists(jf, JsonKeys::NUM_NON_ZEROS)) ? jf.at(JsonKeys::NUM_NON_ZEROS).get<ssize_t>() : -1;
    std::optional<MncSketch> mnc;
    if (keyExists(jf, JsonKeys::MNC_SKETCH)) {
        mnc = readMncSketchFromJson(jf, numRows, numCols);
    }
    HDFSMetaData hdfs;
    if (isHDFS) {
        // TODO check if key exist and throw errors if not
        hdfs.isHDFS = jf.at(JsonKeys::HDFS)["isHDFS"];
        ;
        hdfs.HDFSFilename = jf.at(JsonKeys::HDFS)["HDFSFilename"];
    }
    if (isSingleValueType) {
        if (keyExists(jf, JsonKeys::VALUE_TYPE)) {
            ValueTypeCode vtc = jf.at(JsonKeys::VALUE_TYPE).get<ValueTypeCode>();
            FileMetaData md(numRows, numCols, isSingleValueType, vtc, numNonZeros, hdfs);
            if (mnc.has_value())
                md.mncSketch = std::move(mnc);
            return md;
        } else {
            throw std::invalid_argument("A (matrix) meta data JSON file should contain the \"" + JsonKeys::VALUE_TYPE +
                                        "\" key.");
        }
    } else {
        if (keyExists(jf, JsonKeys::SCHEMA)) {
            ValueTypeCode default_vtc = ValueTypeCode::INVALID;
            if (keyExists(jf, JsonKeys::VALUE_TYPE)) {
                default_vtc = jf.at(JsonKeys::VALUE_TYPE).get<ValueTypeCode>();
            }
            std::vector<ValueTypeCode> schema;
            std::vector<std::string> labels;
            auto schemaColumn = jf.at(JsonKeys::SCHEMA).get<std::vector<SchemaColumn>>();
            for (const auto &column : schemaColumn) {
                auto vtc = column.getValueType();
                if (vtc == ValueTypeCode::INVALID) {
                    vtc = default_vtc;
                    if (default_vtc == ValueTypeCode::INVALID)
                        throw std::invalid_argument("While reading a frame's meta data, a column "
                                                    "without value type was "
                                                    "found while not providing a default value type.");
                }
                schema.emplace_back(vtc);
                labels.emplace_back(column.getLabel());
            }
            FileMetaData md(numRows, numCols, isSingleValueType, schema, labels, numNonZeros, hdfs);
            if (mnc.has_value())
                md.mncSketch = std::move(mnc);
            return md;
        } else {
            throw std::invalid_argument("A (frame) meta data JSON file should contain the \"" + JsonKeys::SCHEMA +
                                        "\" key.");
        }
    }
}

std::string MetaDataParser::writeMetaDataToString(const FileMetaData &metaData) {
    nlohmann::json json;
    // Optionally include mnc sketch in meta data
    bool isWriteMncSketch = true;

    json[JsonKeys::NUM_ROWS] = metaData.numRows;
    json[JsonKeys::NUM_COLS] = metaData.numCols;

    if (metaData.isSingleValueType) {
        if (metaData.schema.size() != 1)
            throw std::runtime_error("inappropriate meta data tried to be written to file");
        json[JsonKeys::VALUE_TYPE] = metaData.schema[0];
    } else {
        std::vector<SchemaColumn> schemaColumns;
        // assume that the schema and labels are the same lengths
        for (unsigned int i = 0; i < metaData.schema.size(); i++) {
            SchemaColumn schemaColumn;
            schemaColumn.setLabel(metaData.labels[i]);
            schemaColumn.setValueType(metaData.schema[i]);
            schemaColumns.emplace_back(schemaColumn);
        }
        json[JsonKeys::SCHEMA] = schemaColumns;
    }

    if (metaData.numNonZeros != -1)
        json[JsonKeys::NUM_NON_ZEROS] = metaData.numNonZeros;

    // HDFS
    if (metaData.hdfs.isHDFS) {
        json[JsonKeys::HDFS][JsonKeys::HDFSKeys::isHDFS] = metaData.hdfs.isHDFS;
        std::filesystem::path filePath(metaData.hdfs.HDFSFilename);
        auto baseFileName = filePath.filename().string();

        json[JsonKeys::HDFS][JsonKeys::HDFSKeys::HDFSFilename] = "/" + baseFileName;
    }
    if (isWriteMncSketch && metaData.mncSketch.has_value()) {
        writeMncSketchToJson(json, *metaData.mncSketch);
    }

    return json.dump();
}
void MetaDataParser::writeMetaData(const std::string &filename_, const FileMetaData &metaData) {
    std::string metaFilename = filename_ + ".meta";
    std::ofstream ofs(metaFilename, std::ios::out);
    if (!ofs.good())
        throw std::runtime_error("could not open file '" + metaFilename + "' for writing meta data");

    if (ofs.is_open()) {
        ofs << MetaDataParser::writeMetaDataToString(metaData);
    } else
        throw std::runtime_error("could not open file '" + metaFilename + "' for writing meta data");
}

bool MetaDataParser::keyExists(const nlohmann::json &j, const std::string &key) { return j.find(key) != j.end(); }
