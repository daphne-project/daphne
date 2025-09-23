#include <iostream>
#include <catch.hpp>

#include <string>
#include <filesystem>
#include <dlfcn.h>
#include <cstdint>
#include <tags.h>
#include <sys/resource.h>

#include "runtime/local/io/FileIOCatalogParser.h"
#include "runtime/local/io/FileIORegistry.h"
#include "runtime/local/io/FileMetaData.h"
#include "runtime/local/context/DaphneContext.h"
#include "runtime/local/datastructures/Frame.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/ValueTypeCode.h"
#include <runtime/local/kernels/Read.h>
#include <runtime/local/kernels/Write.h>
#include <runtime/local/kernels/CreateFrame.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

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
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH,registry));

    // Now registry should have reader and writer for .csv and Frame type
    
    REQUIRE_NOTHROW(registry.getReader(".csv", matrixHash));
    REQUIRE_NOTHROW(registry.getWriter(".csv", matrixHash));
    registry.clear();
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
    registry.clear();
}

TEST_CASE("FileIOPlugin dynamic load and registration validity", "[io][plugin]") {
    // The parser already loads and registers the plugin
    auto &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH,registry));

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
    FileIORegistry &registry = FileIORegistry::instance();

    // Register the CSV plugin
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH,registry));
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

    //IOOptions opts;
    FileIORegistry &registry = FileIORegistry::instance();


    // Register the CSV plugin
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH,registry));

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
}

TEMPLATE_PRODUCT_TEST_CASE("FileIO CSV Reader via Registry and Read kernel for matrix", TAG_IO, (DenseMatrix), (std::string)) {
    using DT = TestType;
    DT *m = nullptr;


    FileIORegistry &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH,registry));

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

    // Retrieve the parsed options from the registry
    auto &registry = FileIORegistry::instance();
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH,registry));

    IOOptions opts = registry.getOptions(".csv", IODataType::DENSEMATRIX,"default");

    // Validate options fields

    REQUIRE(opts.extra.size() == 4);
    CHECK(opts.extra.at("delimiter") == ",");
    CHECK(opts.extra.at("hasHeader") == "false");
    CHECK(opts.extra.at("threads") == "16");
    CHECK(opts.extra.at("dateFormat") == "YYYY-MM-DD");
    FileIORegistry::instance().clear();
}


TEMPLATE_PRODUCT_TEST_CASE("FileIO CSV Reader with delimiter '!' and no header using options Frame", TAG_IO, (DenseMatrix), (std::string)) {
    using DT = TestType;
    DT *m = nullptr;

    // Parse catalog (to simulate normal system setup)
    FileIOCatalogParser parser;
    FileIORegistry &registry = FileIORegistry::instance();
    REQUIRE_NOTHROW(parser.parseFileIOCatalog(JSON_PATH,registry));

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
    FileIORegistry::instance().clear();
}

TEMPLATE_PRODUCT_TEST_CASE("FileIO ReadParquet, DenseMatrix", TAG_IO, (DenseMatrix), (double)) {
    using DT = TestType;
    DT *m = nullptr;
    FileIORegistry::instance().clear();
    FileIORegistry &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json",registry));


    read(m, "./test/runtime/local/io/ReadParquet1.parquet", emptyFrame, ctx);
  

    CHECK(m->get(0, 0) == -0.1);
    CHECK(m->get(0, 1) == -0.2);
    CHECK(m->get(0, 2) == 0.1);
    CHECK(m->get(0, 3) == 0.2);

    CHECK(m->get(1, 0) == 3.14);
    CHECK(m->get(1, 1) == 5.41);
    CHECK(m->get(1, 2) == 6.22216);
    CHECK(m->get(1, 3) == 5);
    
    FileIORegistry::instance().clear();
    DataObjectFactory::destroy(m);
}

TEST_CASE("FileIOw parquet_write_frame writes Frame to Parquet", "[parquet][write][frame]") {
    // ---------- Arrange ----------
    const size_t rows = 2;
    const size_t cols = 3;

    auto *c0 = DataObjectFactory::create<DenseMatrix<int32_t>>(rows, 1, /*alloc*/false);
    auto *c1 = DataObjectFactory::create<DenseMatrix<double>>( rows, 1, /*alloc*/false);
    auto *c2 = DataObjectFactory::create<DenseMatrix<std::string>>(rows, 1, /*alloc*/false);

    c0->getValues()[0] = 1;   c0->getValues()[1] = 2;
    c1->getValues()[0] = 1.5; c1->getValues()[1] = 2.5;
    c2->getValues()[0] = "a"; c2->getValues()[1] = "b";

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
    if (std::filesystem::exists(outPath)) std::filesystem::remove(outPath);

    // Register catalog (must map ".parquet"+"Frame" to parquet_write_frame)
    FileIORegistry &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;

    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json", registry));

    // ---------- Act ----------
    REQUIRE_NOTHROW(write(fr, outPath.c_str(), emptyFrame, ctx));
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
    FileIORegistry::instance().clear();
}

TEST_CASE("FileIOw parquet_write writes DenseMatrix<double> to Parquet", "[parquet][write]") {
    // --- Arrange ---
    const size_t rows = 2, cols = 2;
    auto *mat = DataObjectFactory::create<DenseMatrix<double>>(rows, cols, /*alloc*/false);
    double *vals = mat->getValues();
    // Row-major: [ [1.5, 2.5],
    //              [3.5, 4.5] ]
    vals[0] = 1.5; vals[1] = 2.5;
    vals[2] = 3.5; vals[3] = 4.5;

    auto tempDir = std::filesystem::temp_directory_path();
    auto outPath = tempDir / "test_write.parquet";
    if (std::filesystem::exists(outPath))
        std::filesystem::remove(outPath);

    // Registry + catalog (must include the parquet_write symbol mapping)
    FileIORegistry &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json", registry));

    // --- Act: write via generic write() that dispatches through the registry ---
    REQUIRE_NOTHROW(write(mat, outPath.c_str(), emptyFrame, ctx));
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
    FileIORegistry::instance().clear();
}

// Linux: ru_maxrss is in KB; macOS: bytes
static size_t peak_rss_bytes() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return static_cast<size_t>(ru.ru_maxrss);
#else
    return static_cast<size_t>(ru.ru_maxrss) * 1024;
#endif
}
TEMPLATE_PRODUCT_TEST_CASE("FileIO_parquetProbe_1_thread",
                           TAG_IO, (DenseMatrix), (double)){
    using DT = TestType;
    FileIORegistry::instance().clear();
    FileIORegistry &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;
    const std::string path = "scripts/examples/extensions/parquetReader/random_doubles2.parquet";

    FileIORegistry::instance().clear();

    try {
        parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json", registry);
    } catch (const std::length_error& e) {
        std::cerr << "length_error in parseFileIOCatalog: " << e.what() << "\n";
        throw; // or FAIL();
    }

    DT* m = nullptr;

    auto t0 = std::chrono::steady_clock::now();
    read(m, path.c_str() , emptyFrame, ctx);
    auto t1 = std::chrono::steady_clock::now();
    REQUIRE(m != nullptr);
    
    const size_t rows = m->getNumRows();
    const size_t cols = m->getNumCols();

    // Force full decode: touch data
    double checksum = 0.0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            checksum += m->get(i, j);

    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    const size_t peak_bytes = peak_rss_bytes();

    nlohmann::json j{
        {"tool","parquet_plugin_1_thread"},
        {"file",path},
        {"rows",rows},
        {"cols",cols},
        {"checksum",checksum},
        {"wall_s",wall_s},
        {"peak_rss_bytes",peak_bytes}
    };
    std::cout << "PROBE " << j.dump() << "\n";


    FileIORegistry::instance().clear();
    REQUIRE_NOTHROW(parser.parseFileIOCatalog("scripts/examples/extensions/parquetReader/parquet.json", registry));


    DT* m1 = nullptr;
    
    std::vector<Structure*> columns(1);

    auto* keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);

    auto* val1 = keyCol->getValues();

    val1[0] = "16";

    columns[0] = keyCol;

    const char* labels[1] = {"threads"};

    Frame* optsFrame = nullptr;
    createFrame(optsFrame, columns.data(), 1, labels, 1, ctx);

    auto t01 = std::chrono::steady_clock::now();
    read(m1, path.c_str(), optsFrame, ctx);            // your existing call
    auto t11 = std::chrono::steady_clock::now();
    REQUIRE(m1 != nullptr);

    const size_t rows1 = m1->getNumRows();
    const size_t cols1 = m1->getNumCols();

    // Force full decode: touch data
    double checksum1 = 0.0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            checksum1 += m1->get(i, j);

    const double wall_s1 = std::chrono::duration<double>(t11 - t01).count();
    const size_t peak_bytes1 = peak_rss_bytes();

    nlohmann::json j1{
        {"tool","parquet_plugin_16_thread"},
        {"file",path},
        {"rows",rows1},
        {"cols",cols1},
        {"checksum",checksum1},
        {"wall_s",wall_s1},
        {"peak_rss_bytes",peak_bytes1}
    };
    std::cout << "PROBE " << j1.dump() << "\n";

    REQUIRE(*m == *m1);
    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m1);
}

TEST_CASE("FileIO_csvProbe_1_thread", "[FileIO][Parquet][Probe]") {
    using DT = DenseMatrix<double>;

    FileIORegistry::instance().clear();
    FileIORegistry &registry = FileIORegistry::instance();
    FileIOCatalogParser parser;
    const std::string path =
        "scripts/examples/extensions/csv/random_data.csv";

    DT *m = nullptr;

    auto t00 = std::chrono::steady_clock::now();
    read(m, path.c_str(), emptyFrame, ctx);            // your existing call
    auto t10 = std::chrono::steady_clock::now();
    REQUIRE(m != nullptr);

    const size_t rows0 = m->getNumRows();
    const size_t cols0 = m->getNumCols();

    const double wall_s0 = std::chrono::duration<double>(t10 - t00).count();
    const size_t peak_bytes0 = peak_rss_bytes();

    nlohmann::json j0{
        {"tool","daphne"},
        {"file",path},
        {"rows",rows0},
        {"cols",cols0},
        {"wall_s",wall_s0},
        {"peak_rss_bytes",peak_bytes0}
    };
    std::cout << "PROBE " << j0.dump() << "\n";
    //f->print(std::cout);

    FileIORegistry::instance().clear();

    REQUIRE_NOTHROW(parser.parseFileIOCatalog(
        "scripts/examples/extensions/csv/myIO.json", registry));

    DT *m1 = nullptr;

    auto t0 = std::chrono::steady_clock::now();
    read(m1, path.c_str(), emptyFrame, ctx);            // your existing call
    auto t1 = std::chrono::steady_clock::now();
    REQUIRE(m1 != nullptr);

    const size_t rows = m1->getNumRows();
    const size_t cols = m1->getNumCols();

    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    const size_t peak_bytes = peak_rss_bytes();

    nlohmann::json j1{
        {"tool","daphne"},
        {"file",path},
        {"rows",rows},
        {"cols",cols},
        {"wall_s",wall_s},
        {"peak_rss_bytes",peak_bytes}
    };
    std::cout << "PROBE " << j1.dump() << "\n";

    DT *m2 = nullptr;

    auto t000 = std::chrono::steady_clock::now();
    read(m2, path.c_str(), emptyFrame, ctx);            // your existing call
    auto t111 = std::chrono::steady_clock::now();
    REQUIRE(m2 != nullptr);

    const size_t rows1 = m2->getNumRows();
    const size_t cols1 = m2->getNumCols();

    const double wall_s1 = std::chrono::duration<double>(t111 - t000).count();
    const size_t peak_bytes1 = peak_rss_bytes();

    nlohmann::json j2{
        {"tool","daphne"},
        {"file",path},
        {"rows",rows1},
        {"cols",cols1},
        {"wall_s",wall_s1},
        {"peak_rss_bytes",peak_bytes1}
    };
    std::cout << "PROBE " << j2.dump() << "\n";

    //f1->print(std::cout);

    //REQUIRE(*m == *m1);
    DataObjectFactory::destroy(m);
    DataObjectFactory::destroy(m1);
    DataObjectFactory::destroy(m2);

    FileIORegistry::instance().clear();
}


static std::atomic<int> MYREADER_CALLS{0};
static std::atomic<int> YOURREADER_CALLS{0};


static void myReader(void* res, const FileMetaData&, const char*, const IOOptions&, DaphneContext*) {
    MYREADER_CALLS.fetch_add(1);
    *reinterpret_cast<Frame**>(res) = DataObjectFactory::create<Frame>(0,0,nullptr,nullptr,false);
}
static void yourReader(void* res, const FileMetaData&, const char*, const IOOptions&, DaphneContext*) {
    YOURREADER_CALLS.fetch_add(1);
    *reinterpret_cast<Frame**>(res) = DataObjectFactory::create<Frame>(0,0,nullptr,nullptr,false);
}

TEST_CASE("FileIO Direct call proves selected engine runs") {
    auto &reg = FileIORegistry::instance();
    reg.clear();
    MYREADER_CALLS = 0; YOURREADER_CALLS = 0;

    IOOptions opts;
    // Register both engines for the same (.csv, FRAME)
    reg.registerReader(".csv", IODataType::FRAME, "myReader", 5,  opts, myReader);
    reg.registerReader(".csv", IODataType::FRAME, "yourReader",10, opts, yourReader);

    std::vector<Structure*> columns(1);
    auto* keyCol = DataObjectFactory::create<DenseMatrix<std::string>>(1, 1, false);
    auto* val1 = keyCol->getValues();
    val1[0] = "myReader";
    columns[0] = keyCol;
    const char* labels[1] = {"engine"};
    Frame* optsFrame = nullptr;
    createFrame(optsFrame, columns.data(), 1, labels, 1, ctx);

    // ----- A) No engine hint -> highest priority -> yourReader -----
    {
        Frame* out = nullptr;
        REQUIRE_NOTHROW(read(out, CSV_FILE.c_str(), emptyFrame, ctx));
        REQUIRE(YOURREADER_CALLS.load() == 1);
        REQUIRE(MYREADER_CALLS.load() == 0);
        DataObjectFactory::destroy(out);
    }

    // ----- B) Explicit engine -> myReader even if lower priority -----
    {
        Frame* out = nullptr;
        read(out, CSV_FILE.c_str(), optsFrame, ctx);
        REQUIRE(YOURREADER_CALLS.load() == 1);
        REQUIRE(MYREADER_CALLS.load() == 1);
        DataObjectFactory::destroy(out);
    }

    reg.clear();
}

TEST_CASE("FileIO Duplicate Registration is rejected") {
    auto &reg = FileIORegistry::instance();
    reg.clear();
    IOOptions opts;

    reg.registerReader(".csv", IODataType::FRAME, "myReader", 5,  opts, myReader);
    REQUIRE_THROWS( reg.registerReader(".csv", IODataType::FRAME, "myReader",5, opts, yourReader));

    reg.clear();
}