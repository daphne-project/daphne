#include <catch.hpp>
#include <iostream>

#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <tags.h>

#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/Frame.h"
#include "runtime/local/datastructures/ValueTypeCode.h"
#include "runtime/local/io/FileIOCatalog.h"
#include "runtime/local/io/FileIOCatalogParser.h"
#include "runtime/local/io/FileMetaData.h"
#include <runtime/local/kernels/CreateFrame.h>
#include <runtime/local/kernels/Read.h>
#include <runtime/local/kernels/Write.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

using namespace std;

// Hard-coded path to the example JSON catalog
static const string JSON_PATH = "scripts/examples/extensions/csv/csv.json";

static const std::string CSV_FILE = "scripts/examples/extensions/csv/data.csv";
static const std::string specialCSV = "scripts/examples/extensions/csv/specialCSV.csv";

DaphneContext *ctx = nullptr;
static Frame *emptyFrame = DataObjectFactory::create<Frame>(0, 0, nullptr, nullptr, false);
FileIOCatalogParser parser;
auto &catalog = FileIOCatalog::instance();

static void loadBuiltInIOPlugins() {
    // TODO don't hardcode this path here again, rather use a script-level test (much easier and more useful)
    parser.parseFileIOCatalog("scripts/examples/extensions/builtInIO/BuiltIns.json", catalog, 0);
    catalog.captureBaseline();
}

TEST_CASE("FileIOCatalogParser registers CSV plugin via catalog", "[io][catalog]") {
    // Ensure catalog is clean (if supported); otherwise run in fresh process
    loadBuiltInIOPlugins();

    // Should parse without throwing
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH, catalog, 0));

    // Now catalog should have reader and writer for .csv and Frame type

    REQUIRE_NOTHROW(catalog.getReader(".csv", PhyDataType::DENSEMATRIX, ""));
    REQUIRE_NOTHROW(catalog.getWriter(".csv", PhyDataType::DENSEMATRIX, ""));
    catalog.resetToBaseline();
}

TEST_CASE("FileIOCatalog registerReader and getReader", "[io][catalog]") {

    IOOptions opts;

    // Register a dummy reader for Frame type
    catalog.registerReader(
        ".test", PhyDataType::FRAME, "default", 0, opts,
        [](void *res, const FileMetaData &fmd, const char *filename, IOOptions opts, DaphneContext *ctx) {
            // no-op
        });

    SECTION("Lookup existing reader succeeds") { REQUIRE_NOTHROW(catalog.getReader(".test", PhyDataType::FRAME, "")); }

    SECTION("Lookup non-registered reader throws") {
        REQUIRE_THROWS_AS(catalog.getReader(".unknown", PhyDataType::FRAME, ""), std::runtime_error);
    }
    catalog.resetToBaseline();
}

TEST_CASE("FileIO Plugin dynamic load and registration validity", "[io][plugin]") {
    // The parser already loads and registers the plugin

    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH, catalog, 0));

    // Verify catalog lookups work
    auto reader = catalog.getReader(".csv", PhyDataType::DENSEMATRIX, "");
    auto writer = catalog.getWriter(".csv", PhyDataType::DENSEMATRIX, "");

    REQUIRE(reader != nullptr);
    REQUIRE(writer != nullptr);

    // Optionally, test that the .so exists
    filesystem::path pluginDir = filesystem::path(JSON_PATH).parent_path();
    filesystem::path soPath = pluginDir / "libCsvIO.so";
    INFO("Expecting plugin file at: " << soPath.string());
    REQUIRE(filesystem::exists(soPath));
    catalog.resetToBaseline();
}

TEST_CASE("FileIO csv_read loads numeric CSV into DenseMatrix<int32_t>", "[csv][read]") {
    // Prepare a temporary CSV file
    auto tempDir = std::filesystem::temp_directory_path();
    auto csvPath = tempDir / "test_read.csv";
    {
        std::ofstream ofs(csvPath);
        ofs << "1,2\n";
        ofs << "3,4\n";
    }
    IOOptions opts;

    // Register the CSV plugin
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH, catalog, 0));
    auto reader = catalog.getReader(".csv", PhyDataType::DENSEMATRIX, "Daphne");

    // Invoke reader
    Structure *res = nullptr;
    // Create metadata: rows=2, cols=2, single value type int32
    FileMetaData fmd(2, 2, true, ValueTypeCode::SI32);
    DaphneContext *ctx = nullptr;
    REQUIRE_NOTHROW(reader(&res, fmd, csvPath.c_str(), opts, ctx));

    // Check result matrix
    auto *mat = dynamic_cast<DenseMatrix<int32_t> *>(res);
    // REQUIRE(mat != nullptr);
    REQUIRE(mat->getNumRows() == 2);
    REQUIRE(mat->getNumCols() == 2);
    const int32_t *data = mat->getValues();
    REQUIRE(data[0 * 2 + 0] == 1);
    REQUIRE(data[0 * 2 + 1] == 2);
    REQUIRE(data[1 * 2 + 0] == 3);
    REQUIRE(data[1 * 2 + 1] == 4);
    catalog.resetToBaseline();
}

TEST_CASE("FileIO csv_write writes DenseMatrix<double> to CSV", "[csv][write]") {
    // Create a 2x2 double matrix
    size_t rows = 2, cols = 2;
    auto *mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
    double *vals = mat->getValues();
    vals[0] = 1.5;
    vals[1] = 2.5;
    vals[2] = 3.5;
    vals[3] = 4.5;

    // Prepare output file path
    auto tempDir = std::filesystem::temp_directory_path();
    auto outPath = tempDir / "test_write.csv";

    // IOOptions opts;

    // Register the CSV plugin
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH, catalog, 0));

    write(mat, outPath.c_str(), emptyFrame, ctx);
    // Read back file
    std::ifstream ifs(outPath);
    REQUIRE(ifs.good());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) {
        lines.push_back(line);
    }
    REQUIRE(lines.size() == 2);
    REQUIRE(lines[0] == "1.5,2.5");
    REQUIRE(lines[1] == "3.5,4.5");
    catalog.resetToBaseline();
}

TEMPLATE_PRODUCT_TEST_CASE("FileIO CSV Reader via Catalog and Read kernel for matrix", TAG_IO, (DenseMatrix),
                           (std::string)) {
    using DT = TestType;
    DT *m = nullptr;

    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH, catalog, 0));

    REQUIRE_NOTHROW(read(m, CSV_FILE.c_str(), emptyFrame, nullptr));

    // Verify dimensions (3 rows, 3 cols)
    REQUIRE(m->getNumRows() == 3);
    REQUIRE(m->getNumCols() == 3);

    // Check values (all read as strings)
    CHECK(m->get(0, 0) == "Alice");
    CHECK(m->get(0, 1) == "30");
    CHECK(m->get(0, 2) == "60000.0");

    CHECK(m->get(1, 0) == "Bob");
    CHECK(m->get(1, 1) == "25");
    CHECK(m->get(1, 2) == "55000.5");

    CHECK(m->get(2, 0) == "Charlie");
    CHECK(m->get(2, 1) == "35");
    CHECK(m->get(2, 2) == "70000.75");

    // Clean up
    DataObjectFactory::destroy(m);
    catalog.resetToBaseline();
}

TEST_CASE("FileIOCatalogParser parses options correctly", "[io][catalog]") {
    // Parse the catalog

    // Retrieve the parsed options from the catalog

    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH, catalog, 0));

    IOOptions opts = catalog.getOptions(".csv", PhyDataType::DENSEMATRIX, "Daphne");

    // Validate options fields

    REQUIRE(opts.size() == 3);
    CHECK(opts.at("delimiter") == ",");
    CHECK(opts.at("hasHeader") == "false");
    CHECK(opts.at("threads") == "1");
    FileIOCatalog::instance().resetToBaseline();
}

TEMPLATE_PRODUCT_TEST_CASE("FileIO CSV Reader with delimiter '!' and no header using options Frame", TAG_IO,
                           (DenseMatrix), (std::string)) {
    using DT = TestType;
    DT *m = nullptr;

    // Parse catalog (to simulate normal system setup)
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH, catalog, 0));

    // Create override options in a Frame
    std::vector<Structure *> columns(2);

    auto *keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto *valCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);

    auto *val1 = keyCol->getValues();
    auto *val2 = valCol->getValues();

    val1[0] = "false";
    val2[0] = "!";

    columns[0] = keyCol;
    columns[1] = valCol;

    const char *labels[2] = {"hasHeader", "delimiter"};

    Frame *opts = nullptr;
    createFrame(opts, columns.data(), 2, labels, 2, ctx);

    // Read CSV using the options Frame
    REQUIRE_NOTHROW(read(m, specialCSV.c_str(), opts, ctx));

    // Verify content
    REQUIRE(m->getNumRows() == 3);
    REQUIRE(m->getNumCols() == 3);

    CHECK(m->get(0, 0) == "Alice");
    CHECK(m->get(0, 1) == "30");
    CHECK(m->get(0, 2) == "60000.0");

    CHECK(m->get(1, 0) == "Bob");
    CHECK(m->get(1, 1) == "25");
    CHECK(m->get(1, 2) == "55000.5");

    CHECK(m->get(2, 1) == "35");
    CHECK(m->get(2, 2) == "70000.75");

    // Step 6: Cleanup
    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(opts);
    FileIOCatalog::instance().resetToBaseline();
}

TEMPLATE_PRODUCT_TEST_CASE("FileIO ReadParquet, DenseMatrix", TAG_IO, (DenseMatrix), (double)) {
    using DT = TestType;
    DT *m = nullptr;
    FileIOCatalog::instance().resetToBaseline();
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json", catalog, 0));

    read(m, "./test/runtime/local/io/ReadParquet1.parquet", emptyFrame, ctx);

    CHECK(m->get(0, 0) == -0.1);
    CHECK(m->get(0, 1) == -0.2);
    CHECK(m->get(0, 2) == 0.1);
    CHECK(m->get(0, 3) == 0.2);

    CHECK(m->get(1, 0) == 3.14);
    CHECK(m->get(1, 1) == 5.41);
    CHECK(m->get(1, 2) == 6.22216);
    CHECK(m->get(1, 3) == 5);

    FileIOCatalog::instance().resetToBaseline();
    DataObjectFactory::destroy(m);
}

TEST_CASE("FileIO parquet_write_frame writes Frame to Parquet", "[parquet][write][frame]") {
    const size_t rows = 2;
    const size_t cols = 3;

    auto *c0 = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, 1, /*alloc*/ false);
    auto *c1 = DataObjectFactory::create<DenseMatrix<double>>(rows, 1, /*alloc*/ false);
    auto *c2 = DataObjectFactory::create<DenseMatrix<std::string>>(rows, 1, /*alloc*/ false);

    c0->getValues()[0] = 1;
    c0->getValues()[1] = 2;
    c1->getValues()[0] = 1.5;
    c1->getValues()[1] = 2.5;
    c2->getValues()[0] = "a";
    c2->getValues()[1] = "b";

    Structure *dataCols[3] = {c0, c1, c2};
    const char *dataLabels[3] = {"id", "value", "name"};

    Frame *fr = nullptr;
    DaphneContext *ctx = nullptr;
    createFrame(fr, dataCols, cols, dataLabels, cols, ctx);
    REQUIRE(fr != nullptr);
    REQUIRE(fr->getNumRows() == rows);
    REQUIRE(fr->getNumCols() == cols);

    // Temp output path
    auto outPath = std::filesystem::temp_directory_path() / "test_frame_write.parquet";
    if (std::filesystem::exists(outPath))
        std::filesystem::remove(outPath);

    // Register catalog (must map ".parquet"+"Frame" to parquet_write_frame)

    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json", catalog, 0));

    std::vector<Structure *> columns(1);
    auto *keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto *val1 = keyCol->getValues();
    val1[0] = "Daphne";
    columns[0] = keyCol;
    const char *labels[1] = {"engine"};
    Frame *opts = nullptr;
    createFrame(opts, columns.data(), 1, labels, 1, ctx);

    // ---------- Act ----------
    REQUIRE_NOTHROW(write(fr, outPath.c_str(), opts, ctx));
    REQUIRE(std::filesystem::exists(outPath));

    // ---------- Assert: read back with Arrow ----------
    auto fh_res = arrow::io::ReadableFile::Open(outPath.string());
    REQUIRE(fh_res.ok());
    std::shared_ptr<arrow::io::ReadableFile> infile = *fh_res;

    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto st_open = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
    REQUIRE(st_open.ok());

    std::shared_ptr<arrow::Table> table;
    auto st_tbl = reader->ReadTable(&table);
    REQUIRE(st_tbl.ok());

    REQUIRE(table->num_rows() == static_cast<int64_t>(rows));
    REQUIRE(table->num_columns() == 3);

    // Check column names match labels
    auto schema = table->schema();
    REQUIRE(schema->field(0)->name() == "id");
    REQUIRE(schema->field(1)->name() == "value");
    REQUIRE(schema->field(2)->name() == "name");

    // Column 0: int32
    {
        auto arr = std::static_pointer_cast<arrow::Int32Array>(table->column(0)->chunk(0));
        REQUIRE(arr->length() == static_cast<int64_t>(rows));
        REQUIRE(arr->Value(0) == 1);
        REQUIRE(arr->Value(1) == 2);
    }
    // Column 1: double
    {
        auto arr = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
        REQUIRE(arr->length() == static_cast<int64_t>(rows));
        REQUIRE(arr->Value(0) == Approx(1.5));
        REQUIRE(arr->Value(1) == Approx(2.5));
    }
    // Column 2: string
    {
        auto arr = std::static_pointer_cast<arrow::StringArray>(table->column(2)->chunk(0));
        REQUIRE(arr->length() == static_cast<int64_t>(rows));
        REQUIRE(arr->GetString(0) == "a");
        REQUIRE(arr->GetString(1) == "b");
    }

    // Cleanup
    std::filesystem::remove(outPath);
    catalog.resetToBaseline();
}

TEST_CASE("FileIO parquet_write writes DenseMatrix<double> to Parquet", "[parquet][write]") {
    // --- Arrange ---
    const size_t rows = 2, cols = 2;
    auto *mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, /*alloc*/ false);
    double *vals = mat->getValues();
    // Row-major: [ [1.5, 2.5],
    //              [3.5, 4.5] ]
    vals[0] = 1.5;
    vals[1] = 2.5;
    vals[2] = 3.5;
    vals[3] = 4.5;

    auto tempDir = std::filesystem::temp_directory_path();
    auto outPath = tempDir / "test_write.parquet";
    if (std::filesystem::exists(outPath))
        std::filesystem::remove(outPath);

    // Catalog (must include the parquet_write symbol mapping)
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json", catalog, 0));

    std::vector<Structure *> columns(1);
    auto *keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto *val1 = keyCol->getValues();
    val1[0] = "Daphne";
    columns[0] = keyCol;
    const char *labels[1] = {"engine"};
    Frame *opts = nullptr;
    createFrame(opts, columns.data(), 1, labels, 1, ctx);

    // --- Act: write via generic write() that dispatches through the catalog ---
    REQUIRE_NOTHROW(write(mat, outPath.c_str(), opts, ctx));
    REQUIRE(std::filesystem::exists(outPath));

    // --- Assert: read back with Arrow and validate ---
    std::shared_ptr<arrow::io::ReadableFile> infile;
    auto st_open = arrow::io::ReadableFile::Open(outPath.string());
    REQUIRE(st_open.ok());
    infile = *st_open;

    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto st_r = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
    REQUIRE(st_r.ok());

    std::shared_ptr<arrow::Table> table;
    auto st_tbl = reader->ReadTable(&table);
    REQUIRE(st_tbl.ok());

    REQUIRE(table->num_rows() == static_cast<int64_t>(rows));
    REQUIRE(table->num_columns() == static_cast<int64_t>(cols));

    auto col0 = std::static_pointer_cast<arrow::DoubleArray>(table->column(0)->chunk(0));
    auto col1 = std::static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));
    REQUIRE(col0->length() == static_cast<int64_t>(rows));
    REQUIRE(col1->length() == static_cast<int64_t>(rows));

    // row 0
    REQUIRE(col0->Value(0) == Approx(1.5));
    REQUIRE(col1->Value(0) == Approx(2.5));
    // row 1
    REQUIRE(col0->Value(1) == Approx(3.5));
    REQUIRE(col1->Value(1) == Approx(4.5));

    // Cleanup (optional)
    std::filesystem::remove(outPath);
    FileIOCatalog::instance().resetToBaseline();
}

static std::atomic<int> MYREADER_CALLS{0};
static std::atomic<int> YOURREADER_CALLS{0};

static void myReader(void *res, const FileMetaData &, const char *, const IOOptions &, DaphneContext *) {
    MYREADER_CALLS.fetch_add(1);
    *reinterpret_cast<Frame **>(res) = DataObjectFactory::create<Frame>(0, 0, nullptr, nullptr, false);
}
static void yourReader(void *res, const FileMetaData &, const char *, const IOOptions &, DaphneContext *) {
    YOURREADER_CALLS.fetch_add(1);
    *reinterpret_cast<Frame **>(res) = DataObjectFactory::create<Frame>(0, 0, nullptr, nullptr, false);
}

TEST_CASE("FileIO Direct call proves selected engine runs") {
    auto &reg = FileIOCatalog::instance();
    reg.resetToBaseline();
    MYREADER_CALLS = 0;
    YOURREADER_CALLS = 0;

    IOOptions defaultOpts;
    // Register both engines for the same (.csv, FRAME)
    reg.registerReader(".csv", PhyDataType::FRAME, "myReader", 5, defaultOpts, myReader);
    reg.registerReader(".csv", PhyDataType::FRAME, "yourReader", 10, defaultOpts, yourReader);

    std::vector<Structure *> columns(1);
    auto *keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto *val1 = keyCol->getValues();
    val1[0] = "myReader";
    columns[0] = keyCol;
    const char *labels[1] = {"engine"};
    Frame *opts = nullptr;
    createFrame(opts, columns.data(), 1, labels, 1, ctx);

    // ----- A) No engine hint -> highest priority -> yourReader -----
    {
        Frame *out = nullptr;
        REQUIRE_NOTHROW(read(out, CSV_FILE.c_str(), emptyFrame, ctx));
        REQUIRE(YOURREADER_CALLS.load() == 1);
        REQUIRE(MYREADER_CALLS.load() == 0);
        DataObjectFactory::destroy(out);
    }

    // ----- B) Explicit engine -> myReader even if lower priority -----
    {
        Frame *out = nullptr;
        read(out, CSV_FILE.c_str(), opts, ctx);
        REQUIRE(YOURREADER_CALLS.load() == 1);
        REQUIRE(MYREADER_CALLS.load() == 1);
        DataObjectFactory::destroy(out);
    }

    reg.resetToBaseline();
}

TEST_CASE("FileIO Duplicate Registration is rejected") {
    IOOptions opts;

    catalog.registerReader(".csv", PhyDataType::FRAME, "myReader", 5, opts, myReader);
    REQUIRE_THROWS(catalog.registerReader(".csv", PhyDataType::FRAME, "myReader", 5, opts, yourReader));

    catalog.resetToBaseline();
}

TEST_CASE("FileIO Duplicate lazy Registration is rejected") {
    IOOptions opts;

    catalog.registerLazy(".csv", PhyDataType::FRAME, "daphne", 5, opts, "lib.so", "myReader", "mywriter");
    REQUIRE_THROWS(
        catalog.registerLazy(".csv", PhyDataType::FRAME, "daphne", 5, opts, "lib.so", "myReader", "mywriter"));

    catalog.resetToBaseline();
}