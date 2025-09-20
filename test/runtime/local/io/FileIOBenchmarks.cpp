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
    FileIORegistry::instance().clear();
    FileIORegistry &registry = FileIORegistry::instance();
    DT *m1 = nullptr;
    DT *m2 = nullptr;

    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json",registry));

    BENCHMARK("read double parquet into a DenseMatrix with 1 thread") {
        read(m1, "scripts/examples/extensions/parquetReader/random_doubles2.parquet", emptyFrame, ctx);
        REQUIRE(m1->getNumRows() == 3000000);
        REQUIRE(m1->getNumCols() == 16);
        return 0;
    };
    REQUIRE_NOTHROW(m1->get(1,1));

    std::vector<Structure*> columns(1);
    auto* keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto* val1 = keyCol->getValues();
    val1[0] = "16";
    columns[0] = keyCol;
    const char* labels[1] = {"threads"};
    Frame* optsFrame = nullptr;
    createFrame(optsFrame, columns.data(), 1, labels, 1, ctx);

    BENCHMARK("read double parquet into a DenseMatrix with 16 thread") {
        read(m2, "scripts/examples/extensions/parquetReader/random_doubles2.parquet", optsFrame, ctx);
        REQUIRE(m2->getNumRows() == 3000000);
        REQUIRE(m2->getNumCols() == 16);
        return 0;
    };

    REQUIRE(*m1 == *m2);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);

    FileIORegistry::instance().clear();
}

// 2) Frame parquet benchmark (1 thread vs 16 threads)
TEMPLATE_PRODUCT_TEST_CASE(
    "FileIO3 Benchmark double parquet into a Frame",
    TAG_IO "[.bench]",
    (DenseMatrix),
    (double)
) {
    if (!daphne_bench_enabled()) {
        SUCCEED("Benchmarks disabled. Set DAPHNE_BENCH=1 to run.");
        return;
    }

    FileIORegistry::instance().clear();
    FileIORegistry &registry = FileIORegistry::instance();
    Frame *m1 = nullptr;
    Frame *m2 = nullptr;

    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json",registry));

    BENCHMARK("read double parquet into a Frame with 1 thread") {
        read(m1, "scripts/examples/extensions/parquetReader/random_doubles.parquet", emptyFrame, ctx);
        REQUIRE(m1->getNumRows() == 3000000);
        REQUIRE(m1->getNumCols() == 16);
        return 0;
    };

    std::vector<Structure*> columns(1);
    auto* keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto* val1 = keyCol->getValues();
    val1[0] = "4";
    columns[0] = keyCol;
    const char* labels[1] = {"threads"};
    Frame* optsFrame = nullptr;
    createFrame(optsFrame, columns.data(), 1, labels, 1, ctx);

    BENCHMARK("read double parquet into a Frame with 16 thread") {
        read(m2, "scripts/examples/extensions/parquetReader/random_doubles.parquet", optsFrame, ctx);
        REQUIRE(m2->getNumRows() == 3000000);
        REQUIRE(m2->getNumCols() == 16);
        return 0;
    };

    REQUIRE(*m1 == *m2);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);

    FileIORegistry::instance().clear();
}

// 3) CSV -> DenseMatrix overhead: built-in vs plug-in defaults
TEMPLATE_PRODUCT_TEST_CASE(
    "FileIOBenchmark CSV Reader into matrix overhead: default vs options-frame",
    TAG_IO "[.bench]",
    (DenseMatrix),
    (std::string)
) {
    if (!daphne_bench_enabled()) {
        SUCCEED("Benchmarks disabled. Set DAPHNE_BENCH=1 to run.");
        return;
    }

    using DT = TestType;
    FileIORegistry::instance().clear();
    FileIORegistry &registry = FileIORegistry::instance();
    DT *nm = nullptr;
    DT *pm = nullptr;
    DT *pm2 = nullptr;
    FileIOCatalogParser parser;

    BENCHMARK("read csv with built-in into a DenseMatrix") {
        read(nm, "scripts/examples/extensions/builtInIO/random_data2.csv", emptyFrame, ctx);
        REQUIRE(nm->getNumRows() == 1000000);
        REQUIRE(nm->getNumCols() == 6);
        return 0;
    };

    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/builtInIO/benchmark-defaults.json",registry));

    BENCHMARK("read csv with built-in as plug-in with default arguements into DenseMatrix") {
        read(pm,"scripts/examples/extensions/builtInIO/random_data2.csv", emptyFrame, ctx);
        REQUIRE(pm->getNumRows() == 1000000);
        REQUIRE(pm->getNumCols() == 6);
        return 0;
    };

    REQUIRE(*nm == *pm);

    FileIORegistry::instance().clear();

   REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/csv/myIO.json",registry));

    BENCHMARK("read csv with plug-in with default arguements into DenseMatrix") {
        read(pm2,"scripts/examples/extensions/builtInIO/random_data2.csv", emptyFrame, ctx);
        REQUIRE(pm2->getNumRows() == 1000000);
        REQUIRE(pm2->getNumCols() == 6);
        return 0;
    };

    FileIORegistry::instance().clear();
    REQUIRE(*nm == *pm2);

    DataObjectFactory::destroy(nm);
    DataObjectFactory::destroy(pm);
    DataObjectFactory::destroy(pm2);
}

// 4) CSV -> Frame overhead: built-in vs plug-in defaults
TEMPLATE_PRODUCT_TEST_CASE(
    "FileIOBenchmark CSV Reader into Frame overhead: default vs options-frame",
    TAG_IO "[.bench]",
    (DenseMatrix),
    (std::string)
) {
    if (!daphne_bench_enabled()) {
        SUCCEED("Benchmarks disabled. Set DAPHNE_BENCH=1 to run.");
        return;
    }

    FileIORegistry::instance().clear();
    FileIORegistry &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;
    Frame *nf = nullptr;
    Frame *pf = nullptr;
    Frame *pf2 = nullptr;

    BENCHMARK("read csv with built-in into a Frame") {
        read(nf, "scripts/examples/extensions/builtInIO/random_data.csv", emptyFrame, ctx);
        REQUIRE(nf->getNumRows() == 1000000);
        REQUIRE(nf->getNumCols() == 6);
        return 0;
    };

    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/builtInIO/benchmark-defaults.json",registry));

    BENCHMARK("read csv with built-in as plug-in with default arguements into Frame") {
        read(pf,"scripts/examples/extensions/builtInIO/random_data.csv", emptyFrame, ctx);
        REQUIRE(pf->getNumRows() == 1000000);
        REQUIRE(pf->getNumCols() == 6);
        return 0;
    };

    REQUIRE(*nf == *pf);
    FileIORegistry::instance().clear();

    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/csv/myIO.json",registry));


    BENCHMARK("read csv with plug-in with default arguements into Frame") {
        read(pf2,"scripts/examples/extensions/builtInIO/random_data.csv", emptyFrame, ctx);
        REQUIRE(pf2->getNumRows() == 1000000);
        REQUIRE(pf2->getNumCols() == 6);
        return 0;
    };


    DataObjectFactory::destroy(nf);
    DataObjectFactory::destroy(pf);
    DataObjectFactory::destroy(pf2);
    
}
