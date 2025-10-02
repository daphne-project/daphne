/*
 * Copyright 2025 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <api/cli/StatusCode.h>
#include <api/cli/Utils.h>


#include <tags.h>
#include <catch.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Paths
// -----------------------------------------------------------------------------
static const std::string dirPath = "test/api/cli/extensibility/IO/";
static const std::string outDir  = dirPath + "out/";
const std::string dirCheckPath = "test/api/cli/io/";

// Extra catalog flag (passed to every DAPHNE invocation)
static const char* kExtFlag = "--FileIO-ext";
static const char* kExtPath = "scripts/examples/extensions/csv/myIO.json";

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cctype>
#include <algorithm>

static inline std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) --b;
    return s.substr(a, b - a);
}

static std::vector<std::string> splitCsvLine(const std::string &line, char delim) {
    std::vector<std::string> out;
    std::string cur;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (inQuotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i+1] == '"') { cur.push_back('"'); ++i; }
                else inQuotes = false;
            } else cur.push_back(c);
        } else {
            if (c == '"') inQuotes = true;
            else if (c == delim) { out.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

static bool parseFloat(const std::string& s, double& out) {
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end && *end == '\0';
}

bool compareCsvFiles(const std::string& aPath,
                     const std::string& bPath,
                     char delim = ',',
                     bool hasHeader = false,
                     double tol = 0.0,
                     bool nanSafe = false)
{
    std::ifstream fa(aPath), fb(bPath);
    if (!fa.good() || !fb.good()) return false;

    std::string la, lb;
    size_t row = 0;

    if (hasHeader) { std::getline(fa, la); std::getline(fb, lb); if (la != lb) return false; }

    while (true) {
        bool ea = !std::getline(fa, la);
        bool eb = !std::getline(fb, lb);
        if (ea || eb) return ea && eb; // both ended?
        ++row;

        auto ca = splitCsvLine(la, delim);
        auto cb = splitCsvLine(lb, delim);
        if (ca.size() != cb.size()) return false;

        for (size_t c = 0; c < ca.size(); ++c) {
            std::string sa = trim(ca[c]);
            std::string sb = trim(cb[c]);

            // try numeric compare if tol > 0
            if (tol > 0.0) {
                double da, db;
                bool na = parseFloat(sa, da);
                bool nb = parseFloat(sb, db);
                if (na && nb) {
                    if (nanSafe && std::isnan(da) && std::isnan(db)) continue;
                    if (std::isnan(da) || std::isnan(db)) return false;
                    if (std::fabs(da - db) > tol) return false;
                    continue;
                }
            }
            if (sa != sb) return false;
        }
    }
}


// -----------------------------------------------------------------------------
// Macros (same style as your externalSQL tests), but with the extra flag
// -----------------------------------------------------------------------------

// Compare stdout with .txt reference
#define MAKE_READ_TEST_CASE(name, count)                                                      \
    TEST_CASE(name "_success", TAG_IO) {                                                                    \
        for (unsigned i = 1; i <= count; ++i) {                                                 \
            DYNAMIC_SECTION(name "_success_" << i << ".daphne") {                             \
                compareDaphneToRefSimple(dirPath, name "_success", i, kExtFlag, kExtPath);                   \
            }                                                                                   \
        }                                                                                       \
    }

// For write tests: ensure SUCCESS, stdout OK, and files created in out/
#define MAKE_CSV_WRITE_TEST_CASE(NAME_LIT, COUNT, DT_LIT)                                             \
   TEST_CASE(NAME_LIT "_success", TAG_IO) {                                                              \
      const std::string name = NAME_LIT;                                                                \
      const std::string dt   = DT_LIT;                                                                  \
      std::filesystem::create_directories(dirPath + "out");                                             \
                                                                                                         \
      for (unsigned i = 1; i <= (COUNT); i++) {                                                         \
         DYNAMIC_SECTION(name << "_success_" << i) {                                                   \
               std::string scriptPathWrt =                                                         \
                  dirPath + name + "_success_" + std::to_string(i) + ".daphne";                         \
               std::string scriptPathCmp =                                                         \
                  dirCheckPath + "do_check_" + dt + ".daphne";                                          \
               std::string outPath =                                                               \
                  dirPath + "out/" + name + "_success_" + std::to_string(i) + ".csv";                   \
               std::string refPath =                                                               \
                  dirPath + "ref/" + name + "_success_" + std::to_string(i) + "_ref.csv";               \
                                                                                                         \
               std::error_code ec;                                                                       \
               std::filesystem::remove(outPath, ec);                                                     \
                                                                                                         \
               checkDaphneStatusCode(StatusCode::SUCCESS,                                        \
                     scriptPathWrt.c_str(), "--args",                                                  \
                     (std::string("outPath=\"") + outPath + "\"").c_str(),                             \
                     kExtFlag, kExtPath);                                                             \
                                                                                                         \
               std::string nanSafe = "false";                                                      \
               CHECK(compareCsvFiles(outPath, refPath, ',', /*hasHeader*/false, /*tol*/1e-9, /*nanSafe*/true));                       \
         }                                                                                             \
      }                                                                                                 \
   }



// Expect EXECUTION_ERROR for failure scripts
#define MAKE_FAILURE_TEST_CASE(base, count)                                                     \
    TEST_CASE(base ", IO failure", TAG_IO) {                                                    \
        for (unsigned i = 1; i <= count; ++i) {                                                 \
            DYNAMIC_SECTION(base "_failure_" << i << ".daphne") {                               \
                checkDaphneStatusCodeSimple(StatusCode::EXECUTION_ERROR,                        \
                                            dirPath, base "_failure", i,                        \
                                            kExtFlag, kExtPath);                                \
            }                                                                                   \
        }                                                                                       \
    }

// -----------------------------------------------------------------------------
// Instantiate tests for your current files
// -----------------------------------------------------------------------------

// Read success
MAKE_READ_TEST_CASE("IO_read", 2)

// Write success
MAKE_CSV_WRITE_TEST_CASE("IO_write", 2, "matrix")

// Failures
MAKE_FAILURE_TEST_CASE("IO_read", 2)