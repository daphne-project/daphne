#include <catch.hpp>
#include <string>
#include <filesystem>
#include <dlfcn.h>
#include <cstdint>
#include <tags.h>

#include "runtime/local/io/FileIOCatalogParser.h"
#include "runtime/local/io/FileIORegistry.h"
#include "runtime/local/io/FileMetaData.h"
#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/Frame.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/ValueTypeCode.h"
#include <runtime/local/kernels/Read.h>
#include <runtime/local/kernels/CreateFrame.h>


using namespace std;

// Hard-coded path to the example JSON catalog
static const string JSON_PATH = "scripts/examples/extensions/csv/myIO.json";
IODataType frameHash = FRAME;
IODataType matrixHash = DENSEMATRIX;

static const std::string CSV_FILE = "scripts/examples/extensions/csv/data.csv";
static const std::string specialCSV = "scripts/examples/extensions/csv/specialCSV.csv";

DaphneContext* ctx = nullptr;
static Frame *emptyFrame = DataObjectFactory::create<Frame>(0, 0, nullptr, nullptr, false);


TEST_CASE("FileIOCatalogParser registers CSV plugin via registry", "[io][catalog]") {
    // Ensure registry is clean (if supported); otherwise run in fresh process
    auto &registry = FileIORegistry::instance();

    FileIOCatalogParser parser;
    // Should parse without throwing
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH));

    // Now registry should have reader and writer for .csv and Frame type
    
    REQUIRE_NOTHROW(registry.getReader(".csv", matrixHash));
    REQUIRE_NOTHROW(registry.getWriter(".csv", matrixHash));
}

TEST_CASE("FileIORegistry registerReader and getReader", "[io][registry]") {
    auto &registry = FileIORegistry::instance();
    IOOptions opts;

    // Register a dummy reader for Frame type
    registry.registerReader(
        ".test",
        frameHash,
        opts,
        [](void* res, const FileMetaData &fmd, const char* filename, IOOptions opts, DaphneContext* ctx) {
            // no-op
        }
    );

    SECTION("Lookup existing reader succeeds") {
        REQUIRE_NOTHROW(registry.getReader(".test", frameHash));
    }

    SECTION("Lookup non-registered reader throws") {
        REQUIRE_THROWS_AS(registry.getReader(".unknown", frameHash), std::out_of_range);
    }
}

TEST_CASE("FileIOPlugin dynamic load and registration validity", "[io][plugin]") {
    // The parser already loads and registers the plugin
    auto &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH));

    // Verify registry lookups work
    auto reader = registry.getReader(".csv", matrixHash);
    auto writer = registry.getWriter(".csv", matrixHash);

    REQUIRE(reader != nullptr);
    REQUIRE(writer != nullptr);

    // Optionally, test that the .so exists
    filesystem::path pluginDir = filesystem::path(JSON_PATH).parent_path();
    filesystem::path soPath = pluginDir / "libCsvIO.so";
    INFO("Expecting plugin file at: " << soPath.string());
    REQUIRE(filesystem::exists(soPath));
}

TEST_CASE("FileIO csv_read loads numeric CSV into DenseMatrix<int32_t>", "[csv][read]") {
    // Prepare a temporary CSV file
    auto tempDir = std::filesystem::temp_directory_path();
    auto csvPath = tempDir / "test_read.csv";
    {
        std::ofstream ofs(csvPath);
        ofs << "col1,col2\n";
        ofs << "1,2\n";
        ofs << "3,4\n";
    }
    IOOptions opts;

    // Register the CSV plugin
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH));

    auto &registry = FileIORegistry::instance();
    auto reader = registry.getReader(".csv", matrixHash);

    // Invoke reader
    Structure* res = nullptr;
    // Create metadata: rows=2, cols=2, single value type int32
    FileMetaData fmd(2, 2, true, ValueTypeCode::SI32);
    DaphneContext* ctx = nullptr;
    REQUIRE_NOTHROW(reader(&res, fmd, csvPath.c_str(), opts, ctx));

    // Check result matrix
    auto *mat = dynamic_cast<DenseMatrix<int32_t>*>(res);
    //REQUIRE(mat != nullptr);
    REQUIRE(mat->getNumRows() == 2);
    REQUIRE(mat->getNumCols() == 2);
    const int32_t *data = mat->getValues();
    REQUIRE(data[0 * 2 + 0] == 1);
    REQUIRE(data[0 * 2 + 1] == 2);
    REQUIRE(data[1 * 2 + 0] == 3);
    REQUIRE(data[1 * 2 + 1] == 4);
}

TEST_CASE("FileIO csv_write writes DenseMatrix<double> to CSV", "[csv][write]") {
    // Create a 2x2 double matrix
    size_t rows = 2, cols = 2;
    auto *mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, false);
    double *vals = mat->getValues();
    vals[0] = 1.5; vals[1] = 2.5;
    vals[2] = 3.5; vals[3] = 4.5;

    // Prepare output file path
    auto tempDir = std::filesystem::temp_directory_path();
    auto outPath = tempDir / "test_write.csv";

    IOOptions opts;

    // Register the CSV plugin
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH));
    auto &registry = FileIORegistry::instance();
    auto writer = registry.getWriter(".csv", matrixHash);

    // Invoke writer
    // Create metadata: rows=2, cols=2, single value type double
    FileMetaData fmd2(2, 2, true, ValueTypeCode::F64);
    REQUIRE_NOTHROW(writer(mat, fmd2, outPath.c_str(), opts, ctx));

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
}

TEMPLATE_PRODUCT_TEST_CASE("FileIO CSV Reader via Registry and Read kernel for matrix", TAG_IO, (DenseMatrix), (std::string)) {
    using DT = TestType;
    DT *m = nullptr;

    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH));

    REQUIRE_NOTHROW(read(m, CSV_FILE.c_str(), emptyFrame, nullptr));

    // Verify dimensions (3 rows, 3 cols)
    REQUIRE(m->getNumRows() == 3);
    REQUIRE(m->getNumCols() == 3);

    // Check values (all read as strings)
    CHECK(m->get(0,0) == "Alice");
    CHECK(m->get(0,1) == "30");
    CHECK(m->get(0,2) == "60000.0");

    CHECK(m->get(1,0) == "Bob");
    CHECK(m->get(1,1) == "25");
    CHECK(m->get(1,2) == "55000.5");

    CHECK(m->get(2,0) == "Charlie");
    CHECK(m->get(2,1) == "35");
    CHECK(m->get(2,2) == "70000.75");

    // Clean up
    DataObjectFactory::destroy(m);

}

TEST_CASE("FileIOCatalogParser parses options correctly", "[io][catalog]") {
    // Parse the catalog
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH));

    // Retrieve the parsed options from the registry
    auto &registry = FileIORegistry::instance();
    IOOptions opts = registry.getOptions(".csv", IODataType::DENSEMATRIX);

    // Validate options fields

    REQUIRE(opts.extra.size() == 4);
    CHECK(opts.extra.at("delimiter") == ",");
    CHECK(opts.extra.at("hasHeader") == "true");
    CHECK(opts.extra.at("threads") == "4");
    CHECK(opts.extra.at("dateFormat") == "YYYY-MM-DD");
}


TEMPLATE_PRODUCT_TEST_CASE("FileIO CSV Reader with delimiter '!' and no header using options Frame", TAG_IO, (DenseMatrix), (std::string)) {
    using DT = TestType;
    DT *m = nullptr;

    // Parse catalog (to simulate normal system setup)
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH));

    // Create override options in a Frame
    std::vector<Structure*> columns(2);

    auto* keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto* valCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);

    auto* val1 = keyCol->getValues();
    auto* val2 = valCol->getValues();

    val1[0] = "false";
    val2[0] = "!";

    columns[0] = keyCol;
    columns[1] = valCol;

    const char* labels[2] = {"hasHeader", "delimiter"};

    Frame* optsFrame = nullptr;
    createFrame(optsFrame, columns.data(), 2, labels, 2, ctx);

    // Read CSV using the options Frame
    REQUIRE_NOTHROW(read(m, specialCSV.c_str(), optsFrame, ctx));

    // Verify content
    REQUIRE(m->getNumRows() == 3);
    REQUIRE(m->getNumCols() == 3);

    CHECK(m->get(0,0) == "Alice");
    CHECK(m->get(0,1) == "30");
    CHECK(m->get(0,2) == "60000.0");

    CHECK(m->get(1,0) == "Bob");
    CHECK(m->get(1,1) == "25");
    CHECK(m->get(1,2) == "55000.5");

    CHECK(m->get(2,0) == "Charlie");
    CHECK(m->get(2,1) == "35");
    CHECK(m->get(2,2) == "70000.75");

    // Step 6: Cleanup
    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(optsFrame);
}
