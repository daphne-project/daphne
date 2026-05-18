// io_benchmarks.cpp
// Drop-in file with your exact benchmarks, gated by an env var.
//
// Run them only when you want:
//   DAPHNE_BENCH=1 ./io_tests "[.bench]"
//
// If you don't use tags, they'll still be skipped unless DAPHNE_BENCH is set.

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch.hpp>

#include <string>
#include <cstdlib> // std::getenv

#include <tags.h>

#include "runtime/local/io/FileIOCatalogParser.h"
#include "runtime/local/io/FileIORegistry.h"
#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/Frame.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/DataObjectFactory.h"
#include <runtime/local/kernels/Read.h>
#include <runtime/local/kernels/CreateFrame.h>

// ---------------------------
// Small helper: enable switch
// ---------------------------
static inline bool daphne_bench_enabled() {
#if defined(CATCH_CONFIG_ENABLE_BENCHMARKING)
    const char* e = std::getenv("DAPHNE_BENCH");
    return e && *e; // any non-empty value enables running the benches
#else
    return false;
#endif
}

// ---------------------------
// Minimal globals to match your tests
// ---------------------------
static DaphneContext* ctx = nullptr;
static Frame* emptyFrame = DataObjectFactory::create<Frame>(0, 0, nullptr, nullptr, false);

//#############################################
//                BENCHMARKS
//#############################################

// 1) DenseMatrix<double> parquet benchmark (1 thread vs 16 threads)
TEMPLATE_PRODUCT_TEST_CASE(
    "FileIO Benchmark double parquet into a densematrix",
    TAG_IO "[.bench]",
    (DenseMatrix),
    (double)
) {
    if (!daphne_bench_enabled()) {
        SUCCEED("Benchmarks disabled. Set DAPHNE_BENCH=1 to run.");
        return;
    }

    using DT = TestType;
    FileIORegistry::instance().resetToBaseline();
    FileIORegistry &registry = FileIORegistry::instance();
    DT *m1 = nullptr;
    DT *m2 = nullptr;

    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json",registry));

    std::vector<Structure*> columns(1);
    auto* keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto* val1 = keyCol->getValues();
    val1[0] = "Daphne";
    columns[0] = keyCol;
    const char* labels[1] = {"engine"};
    Frame* optsFrame = nullptr;
    createFrame(optsFrame, columns.data(), 1, labels, 1, ctx);

    BENCHMARK("read double parquet into a DenseMatrix with 1 thread") {
        read(m1, "scripts/examples/extensions/parquetReader/random_doubles2.parquet", optsFrame, ctx);
        REQUIRE(m1->getNumRows() == 3000000);
        REQUIRE(m1->getNumCols() == 16);
        return 0;
    };
    REQUIRE_NOTHROW(m1->get(1,1));

    std::vector<Structure*> columns2(2);
    auto* threadsCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    threadsCol->getValues()[0] = "16";

    auto* engineCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    engineCol->getValues()[0] = "Daphne";

    columns2[0] = threadsCol;
    columns2[1] = engineCol;

    // Two labels matching the two columns
    const char* labels2[2] = {"threads", "engine"};

    Frame* optsFrame2 = nullptr;
    createFrame(optsFrame2, columns2.data(), /*numCols=*/2, labels2, /*numLabels=*/2, ctx);

    BENCHMARK("read double parquet into a DenseMatrix with 16 thread") {
        read(m2, "scripts/examples/extensions/parquetReader/random_doubles2.parquet", optsFrame2, ctx);
        REQUIRE(m2->getNumRows() == 3000000);
        REQUIRE(m2->getNumCols() == 16);
        return 0;
    };

    REQUIRE(*m1 == *m2);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);

    FileIORegistry::instance().resetToBaseline();
}

// 2) Frame parquet benchmark (1 thread vs 16 threads)
TEMPLATE_PRODUCT_TEST_CASE(
    "FileIO Benchmark double parquet into a Frame",
    TAG_IO "[.bench]",
    (DenseMatrix),
    (double)
) {
    if (!daphne_bench_enabled()) {
        SUCCEED("Benchmarks disabled. Set DAPHNE_BENCH=1 to run.");
        return;
    }

    FileIORegistry::instance().resetToBaseline();
    FileIORegistry &registry = FileIORegistry::instance();
    Frame *m1 = nullptr;
    Frame *m2 = nullptr;

    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json",registry));

    std::vector<Structure*> columns(1);
    auto* keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto* val1 = keyCol->getValues();
    val1[0] = "Daphne";
    columns[0] = keyCol;
    const char* labels[1] = {"engine"};
    Frame* optsFrame = nullptr;
    createFrame(optsFrame, columns.data(), 1, labels, 1, ctx);

    BENCHMARK("read double parquet into a Frame with 1 thread") {
        read(m1, "scripts/examples/extensions/parquetReader/random_doubles.parquet", optsFrame, ctx);
        REQUIRE(m1->getNumRows() == 3000000);
        REQUIRE(m1->getNumCols() == 16);
        return 0;
    };

    std::vector<Structure*> columns2(2);
    auto* threadsCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    threadsCol->getValues()[0] = "16";

    auto* engineCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    engineCol->getValues()[0] = "Daphne";

    columns2[0] = threadsCol;
    columns2[1] = engineCol;

    // Two labels matching the two columns
    const char* labels2[2] = {"threads", "engine"};

    Frame* optsFrame2 = nullptr;
    createFrame(optsFrame2, columns2.data(), /*numCols=*/2, labels2, /*numLabels=*/2, ctx);

    BENCHMARK("read double parquet into a Frame with 16 thread") {
        read(m2, "scripts/examples/extensions/parquetReader/random_doubles.parquet", optsFrame2, ctx);
        REQUIRE(m2->getNumRows() == 3000000);
        REQUIRE(m2->getNumCols() == 16);
        return 0;
    };

    REQUIRE(*m1 == *m2);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);

    FileIORegistry::instance().resetToBaseline();
}