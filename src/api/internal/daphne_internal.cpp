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

#include "runtime/local/datastructures/IAllocationDescriptor.h"
#include "runtime/local/io/FileIORegistry.h"

#ifdef USE_MPI
#include "runtime/distributed/worker/MPIWorker.h"
#endif

#include "compiler/execution/DaphneIrExecutor.h"
#include <api/cli/DaphneUserConfig.h>
#include <api/cli/StatusCode.h>
#include <api/daphnelib/DaphneLibResult.h>
#include <api/internal/daphne_internal.h>
#include <compiler/utils/CompilerUtils.h>
#include <parser/catalog/KernelCatalogParser.h>
#include <runtime/local/io/FileIOCatalogParser.h>
#include <parser/config/ConfigParser.h>
#include <parser/daphnedsl/DaphneDSLParser.h>
#include <runtime/local/vectorized/LoadPartitioningDefs.h>
#include <util/DaphneLogger.h>
#include <util/KernelDispatchMapping.h>
#include <util/PropertyLogger.h>
#include <util/Statistics.h>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/CommandLine.h"

#ifdef USE_CUDA
#include <runtime/local/kernels/CUDA/HostUtils.h>
#endif

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>

#include <csetjmp>
#include <csignal>
#include <cstring>
#include <execinfo.h>

// global logger handle for this executable
static std::unique_ptr<DaphneLogger> logger;

using namespace std;
using namespace mlir;
using namespace llvm::cl;

#include <filesystem>
#include <fstream>
#include <optional>

namespace fs = std::filesystem;

static std::optional<fs::path> readRepoRootMarker() {
    try {
        fs::path exeDir = fs::canonical("/proc/self/exe").parent_path();
        fs::path marker = exeDir / ".daphne_repo_root";
        if (fs::exists(marker)) {
            std::ifstream in(marker);
            std::string line; std::getline(in, line);
            if (!line.empty() && fs::exists(line))
                return fs::canonical(fs::path(line));
        }
    } catch (...) {}
    return std::nullopt;
}

static std::string resolveRepoPathAuto(const std::string &rel) {
    fs::path r(rel);
    if (r.is_absolute())
        return fs::canonical(r).string();

    if (auto rr = readRepoRootMarker()) {
        fs::path p = *rr / r;
        if (fs::exists(p)) return fs::canonical(p).string();
    }

    if (fs::exists(r))  // CWD
        return fs::canonical(r).string();

    try {
        fs::path exeDir = fs::canonical("/proc/self/exe").parent_path();
        for (fs::path cur = exeDir; !cur.empty(); cur = cur.parent_path()) {
            fs::path p = cur / r;
            if (fs::exists(p)) return fs::canonical(p).string();
            if (cur == cur.parent_path()) break;
        }
    } catch (...) {}

    if (const char *env = std::getenv("DAPHNE_REPO")) {
        fs::path p = fs::path(env) / r;
        if (fs::exists(p)) return fs::canonical(p).string();
    }

    throw std::runtime_error("Could not resolve repo-relative path: " + rel);
}


void parseScriptArgs(const llvm::cl::list<string> &scriptArgsCli, unordered_map<string, string> &scriptArgsFinal) {
    for (const std::string &pair : scriptArgsCli) {
        size_t pos = pair.find('=');
        if (pos == string::npos)
            throw std::runtime_error("script arguments must be specified as "
                                     "name=value, but found '" +
                                     pair + "'");
        const string argName = pair.substr(0, pos);
        const string argValue = pair.substr(pos + 1, pair.size());
        if (scriptArgsFinal.count(argName))
            throw runtime_error("script argument: '" + argName + "' was provided more than once");
        scriptArgsFinal.emplace(argName, argValue);
    }
}
void printVersion(llvm::raw_ostream &os) {
    // TODO Include some of the important build flags into the version string.
    os << "DAPHNE Version 0.3\n"
       << "An Open and Extensible System Infrastructure for Integrated Data "
          "Analysis Pipelines\n"
       << "https://github.com/daphne-eu/daphne\n";
}

namespace {
volatile std::sig_atomic_t gSignalStatus;
jmp_buf return_from_handler;
} // namespace

void handleSignals(int signal) {
    constexpr int callstackMaxSize = 25;
    void *callstack[callstackMaxSize];
    auto callstacksReturned = backtrace(callstack, callstackMaxSize);
    backtrace_symbols_fd(callstack, callstacksReturned, STDOUT_FILENO);
    gSignalStatus = signal;
    longjmp(return_from_handler, gSignalStatus);
}

void logErrorDaphneLibAware(DaphneLibResult *daphneLibRes, std::string msg) {
    if (daphneLibRes != nullptr) // For DaphneLib (Python API), error message is handled later in script.py.
        daphneLibRes->error_message = msg;
    else
        spdlog::error(msg);
}

int startDAPHNE(int argc, const char **argv, DaphneLibResult *daphneLibRes, int *id, DaphneUserConfig &user_config) {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point tpBeg = clock::now();

    // install signal handler to catch information from shared libraries (for exception handling)
    std::signal(SIGABRT, handleSignals);
    std::signal(SIGSEGV, handleSignals);

    // ************************************************************************
    // Parse command line arguments
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Define options
    // ------------------------------------------------------------------------

    // All the variables concerned with the LLVM command line parser (those of
    // type OptionCategory, opt, ...) must be declared static here, because
    // this function may run multiple times in the context of DaphneLib
    // (DAPHNE's Python API). Without static, the second invocation of this
    // function crashes because the options set in the first invocation are
    // still present in some global state. This must be due to the way the LLVM
    // command line library handles its internal state.

    // Option categories ------------------------------------------------------

    // TODO We will probably subdivide the options into multiple groups later.
    static OptionCategory daphneOptions("DAPHNE Options");
    static OptionCategory schedulingOptions("Advanced Scheduling Knobs");
    static OptionCategory distributedBackEndSetupOptions("Distributed Backend Knobs");
    static OptionCategory HDFSOptions("HDFS Knobs");

    // Options ----------------------------------------------------------------

    // Distributed backend Knobs
    static opt<ALLOCATION_TYPE> distributedBackEndSetup(
        "dist_backend", cat(distributedBackEndSetupOptions), desc("Choose the options for the distribution backend:"),
        values(clEnumValN(ALLOCATION_TYPE::DIST_MPI, "MPI",
                          "Use message passing interface for internode data "
                          "exchange (default)"),
               clEnumValN(ALLOCATION_TYPE::DIST_GRPC_SYNC, "sync-gRPC",
                          "Use remote procedure call (synchronous gRPC with "
                          "threading) for internode data exchange"),
               clEnumValN(ALLOCATION_TYPE::DIST_GRPC_ASYNC, "async-gRPC",
                          "Use remote procedure call (asynchronous gRPC) for "
                          "internode data exchange")),
        init(ALLOCATION_TYPE::DIST_MPI));
    static opt<size_t> maxDistrChunkSize("max-distr-chunk-size", cat(distributedBackEndSetupOptions),
                                         desc("Define the maximum chunk size per message for the distributed "
                                              "runtime (in bytes)"
                                              "(default is close to maximum allowed ~2GB)"),
                                         init(std::numeric_limits<int>::max() - 1024));

    // HDFS knobs
    static opt<bool> use_hdfs("enable-hdfs", cat(HDFSOptions), desc("Enable HDFS filesystem"));
    static opt<string> hdfs_Address("hdfs-ip", cat(HDFSOptions), desc("IP of the HDFS filesystem (including port)."),
                                    init(""));
    static opt<string> hdfs_username("hdfs-username", cat(HDFSOptions), desc("Username of the HDFS filesystem."),
                                     init(""));

    // Scheduling options
    using enum SelfSchedulingScheme;
    using enum QueueTypeOption;
    using enum VictimSelectionLogic;

    static opt<SelfSchedulingScheme> taskPartitioningScheme(
        "partitioning", cat(schedulingOptions), desc("Choose task partitioning scheme:"),
        values(clEnumVal(STATIC, "Static (default)"), clEnumVal(SS, "Self-scheduling"),
               clEnumVal(GSS, "Guided self-scheduling"), clEnumVal(TSS, "Trapezoid self-scheduling"),
               clEnumVal(FAC2, "Factoring self-scheduling"), clEnumVal(TFSS, "Trapezoid Factoring self-scheduling"),
               clEnumVal(FISS, "Fixed-increase self-scheduling"), clEnumVal(VISS, "Variable-increase self-scheduling"),
               clEnumVal(PLS, "Performance loop-based self-scheduling"),
               clEnumVal(MSTATIC, "Modified version of Static, i.e., instead "
                                  "of n/p, it uses n/(4*p) where n is number "
                                  "of tasks and p is number of threads"),
               clEnumVal(MFSC, "Modified version of fixed size chunk self-scheduling, "
                               "i.e., MFSC does not require profiling information as FSC"),
               clEnumVal(PSS, "Probabilistic self-scheduling"), clEnumVal(AUTO, "Automatic partitioning")),
        init(STATIC));

    static opt<QueueTypeOption> queueSetupScheme(
        "queue_layout", cat(schedulingOptions), desc("Choose queue setup scheme:"),
        values(clEnumVal(CENTRALIZED, "One queue (default)"), clEnumVal(PERGROUP, "One queue per CPU group"),
               clEnumVal(PERCPU, "One queue per CPU core")),
        init(CENTRALIZED));

    static opt<VictimSelectionLogic> victimSelection(
        "victim_selection", cat(schedulingOptions), desc("Choose work stealing victim selection logic:"),
        values(clEnumVal(SEQ, "Steal from next adjacent worker (default)"),
               clEnumVal(SEQPRI, "Steal from next adjacent worker, prioritize same NUMA domain"),
               clEnumVal(RANDOM, "Steal from random worker"),
               clEnumVal(RANDOMPRI, "Steal from random worker, prioritize same NUMA domain")),
        init(SEQ));

    static opt<int> numberOfThreads("num-threads", cat(schedulingOptions),
                                    desc("Define the number of the CPU threads used by the vectorized "
                                         "execution engine "
                                         "(default is equal to the number of physical cores on the target "
                                         "node that executes the code)"));
    static opt<int> minimumTaskSize("grain-size", cat(schedulingOptions),
                                    desc("Define the minimum grain size of a task (default is 1)"), init(1));
    static opt<size_t> phyBatchSize(
        "phy-batch-size", cat(schedulingOptions),
        desc("The physical size in bytes into which the whole of all inputs and results of a "
             "vectorized pipeline are split by CPU workers (within a task) for cache-efficiency (default is 96 KiB)"),
        init(96 * 1024));
    static opt<bool> useVectorizedPipelines("vec", cat(schedulingOptions), desc("Enable vectorized execution engine"));
    static opt<bool> useVectorizedRestricted(
        "vec-restricted", cat(schedulingOptions),
        desc("Restrict the processing in the vectorized engine to DaphneIR operations that have the attribute \"" +
             CompilerUtils::ATTR_VEC + "\" set to true; only effective if used in combination with --vec"));
    static opt<bool> useDistributedRuntime("distributed", cat(daphneOptions), desc("Enable distributed runtime"));
    static opt<bool> prePartitionRows("pre-partition", cat(schedulingOptions),
                                      desc("Partition rows into the number of queues before applying "
                                           "scheduling technique"));
    static opt<bool> pinWorkers("pin-workers", cat(schedulingOptions), desc("Pin workers to CPU cores"));
    static opt<bool> hyperthreadingEnabled("hyperthreading", cat(schedulingOptions),
                                           desc("Utilize multiple logical CPUs located on the same physical CPU"));
    static opt<bool> debugMultiThreading("debug-mt", cat(schedulingOptions),
                                         desc("Prints debug information about the Multithreading Wrapper"));

    // Other options

    static opt<bool> noObjRefMgnt("no-obj-ref-mgnt", cat(daphneOptions),
                                  desc("Switch off garbage collection by not managing data "
                                       "objects' reference counters"));
    static opt<bool> noIPAConstPropa("no-ipa-const-propa", cat(daphneOptions),
                                     desc("Switch off inter-procedural constant propagation"));
    static opt<bool> noPhyOpSelection("no-phy-op-selection", cat(daphneOptions),
                                      desc("Switch off physical operator selection, use default kernels for "
                                           "all operations"));
    static opt<bool> selectMatrixRepr("select-matrix-repr", cat(daphneOptions),
                                      desc("Automatically choose physical matrix representations "
                                           "(e.g., dense/sparse)"));
    static alias selectMatrixReprAlias( // to still support the longer old form
        "select-matrix-representations", aliasopt(selectMatrixRepr), desc("Alias for --select-matrix-repr"));
    static opt<bool> useColumnar("columnar", cat(daphneOptions),
                                 desc("Use columnar operations instead of frame/matrix operations for relational query "
                                      "processing where possible"));
    static opt<bool> useOptimisticSplitting(
        "optimistic-splitting", cat(daphneOptions),
        desc("Use optimistic splitting for grouped SUM aggregation; requires --columnar"));
    static opt<bool> cuda("cuda", cat(daphneOptions), desc("Use CUDA"));
    static opt<bool> fpgaopencl("fpgaopencl", cat(daphneOptions), desc("Use FPGAOPENCL"));
    static opt<string> libDir("libdir", cat(daphneOptions),
                              desc("The directory containing the kernel catalog files "
                                   "(typically, but not necessarily, along with the kernel shared "
                                   "libraries)"));

    static opt<bool> mlirCodegen("codegen", cat(daphneOptions),
                                 desc("Enables lowering of certain DaphneIR operations on DenseMatrix "
                                      "to low-level MLIR operations."));
    static opt<int> matmul_vec_size_bits("matmul-vec-size-bits", cat(daphneOptions),
                                         desc("Set the vector size to be used in the lowering of the MatMul "
                                              "operation if possible. Value of 0 is interpreted as off switch."),
                                         init(0));
    static opt<bool> matmul_tile("matmul-tile", cat(daphneOptions),
                                 desc("Enables loop tiling in the lowering of the MatMul operation."));
    static opt<int> matmul_unroll_factor("matmul-unroll-factor", cat(daphneOptions),
                                         desc("Factor by which to unroll the finally resulting inner most loop "
                                              "in the lowered MatMul if tiling is used."),
                                         init(1));
    static opt<int> matmul_unroll_jam_factor("matmul-unroll-jam-factor", cat(daphneOptions),
                                             desc("Factor by which to unroll jam the two inner most loop in the "
                                                  "lowered MatMul if tiling is used."),
                                             init(4));
    static opt<int> matmul_num_vec_registers("matmul-num-vec-registers", cat(daphneOptions),
                                             desc("Number of vector registers. Used during automatic tiling in "
                                                  "lowering of MatMulOp"),
                                             init(16));
    static llvm::cl::list<unsigned> matmul_fixed_tile_sizes(
        "matmul-fixed-tile-sizes", cat(daphneOptions),
        desc("Set fixed tile sizes to be used for the lowering of MatMul if "
             "tiling is used. This also enables tiling."),
        CommaSeparated);
    static opt<bool> matmul_invert_loops("matmul-invert-loops", cat(daphneOptions),
                                         desc("Enable inverting of the inner two loops in the matrix "
                                              "multiplication as a fallback option, if tiling is not possible "
                                              "or deactivated."));

    static opt<bool> performHybridCodegen("mlir-hybrid-codegen", cat(daphneOptions),
                                          desc("Enables prototypical hybrid code generation combining "
                                               "pre-compiled kernels and MLIR code generation."));
    static opt<string> kernelExt("kernel-ext", cat(daphneOptions),
                                 desc("Additional kernel extension to register "
                                      "(path to a kernel catalog JSON file)."));
    static opt<std::string> FileIOExt("FileIO-ext",cat(daphneOptions),
                                        desc("Load an additional FileIO extention JSON file" 
                                            "path to a FileIO extention JSON file"));

    enum ExplainArgs {
        columnar,
        kernels,
        llvm,
        parsing,
        parsing_simplified,
        property_inference,
        select_matrix_repr,
        transfer_data_props,
        sql,
        phy_op_selection,
        type_adaptation,
        vectorized,
        obj_ref_mgnt,
        codegen,
        mlir_codegen_sparsity_exploiting_op_fusion,
        mlir_codegen_daphneir_to_mlir,
        mlir_codegen_mlir_specific
    };

    static llvm::cl::list<ExplainArgs> explainArgList(
        "explain", cat(daphneOptions),
        llvm::cl::desc("Show DaphneIR after certain compiler passes (separate "
                       "multiple values by comma, the order is irrelevant)"),
        llvm::cl::values(
            clEnumVal(parsing, "Show DaphneIR after parsing"),
            clEnumVal(parsing_simplified, "Show DaphneIR after parsing and some simplifications"),
            clEnumVal(sql, "Show DaphneIR after SQL parsing"),
            clEnumVal(columnar, "Show DaphneIR after lowering to columnar operations"),
            clEnumVal(property_inference, "Show DaphneIR after property inference"),
            clEnumVal(select_matrix_repr, "Show DaphneIR after selecting "
                                          "physical matrix representations"),
            clEnumVal(transfer_data_props, "Show DaphneIR after inserting ops for transferring compile-time "
                                           "information on data properties to the runtime"),
            clEnumVal(phy_op_selection, "Show DaphneIR after selecting physical operators"),
            clEnumVal(type_adaptation, "Show DaphneIR after adapting types to available kernels"),
            clEnumVal(vectorized, "Show DaphneIR after vectorization"),
            clEnumVal(obj_ref_mgnt, "Show DaphneIR after managing object references"),
            clEnumVal(kernels, "Show DaphneIR after kernel lowering"),
            clEnumVal(codegen, "Show DaphneIR after MLIR codegen"),
            clEnumVal(mlir_codegen_sparsity_exploiting_op_fusion,
                      "Show DaphneIR after MLIR codegen (sparsity-exploiting operator fusion)"),
            clEnumVal(mlir_codegen_daphneir_to_mlir, "Show DaphneIR after MLIR codegen (DaphneIR to MLIR)"),
            clEnumVal(mlir_codegen_mlir_specific, "Show DaphneIR after MLIR codegen (MLIR-specific)"),
            clEnumVal(llvm, "Show DaphneIR after llvm lowering")),
        CommaSeparated);

    static llvm::cl::list<string> scriptArgs1("args", cat(daphneOptions),
                                              desc("Alternative way of specifying arguments to the DaphneDSL "
                                                   "script; must be a comma-separated list of name-value-pairs, "
                                                   "e.g., `--args x=1,y=2.2`"),
                                              CommaSeparated);
    const std::string configFileInitValue = "-";
    static opt<string> configFile("config", cat(daphneOptions),
                                  desc("A JSON file that contains the DAPHNE configuration"), value_desc("filename"),
                                  llvm::cl::init(configFileInitValue));

    static opt<bool> enableStatistics("statistics", cat(daphneOptions), desc("Enables runtime statistics output."));

    static opt<bool> enablePropertyRecording(
        "enable-property-recording", cat(daphneOptions),
        desc("Record data properties of all matrix-typed intermediate results at run-time and save them in the JSON "
             "file specified by --properties-file-path"));

    static opt<bool> enablePropertyInsert(
        "enable-property-insert", cat(daphneOptions),
        desc("Load data properties previously recorded by --enable-property-recording from the JSON file specified by "
             "--properties-file-path and insert them into the IR for further optimization based on the true data "
             "properties"));

    static opt<std::string> propertiesFilePath(
        "properties-file-path", cat(daphneOptions),
        llvm::cl::desc("The path to the JSON file used by --enable-property-recording and --enable-property-insert"),
        llvm::cl::init("properties.json"));

    static opt<bool> enableProfiling("enable-profiling", cat(daphneOptions), desc("Enable profiling support"));
    static opt<bool> timing("timing", cat(daphneOptions),
                            desc("Enable timing of high-level steps (start-up, parsing, compilation, execution) and "
                                 "print the times to stderr in JSON format"));

    // Positional arguments ---------------------------------------------------

    static opt<string> inputFile(Positional, desc("script"), Required);
    static llvm::cl::list<string> scriptArgs2(ConsumeAfter, desc("[arguments]"));

    // ------------------------------------------------------------------------
    // Parse arguments
    // ------------------------------------------------------------------------

    static std::vector<const llvm::cl::OptionCategory *> visibleCategories;
    visibleCategories.push_back(&daphneOptions);
    visibleCategories.push_back(&schedulingOptions);
    visibleCategories.push_back(&distributedBackEndSetupOptions);
    visibleCategories.push_back(&HDFSOptions);

    HideUnrelatedOptions(visibleCategories);

    extrahelp("\nEXAMPLES:\n\n"
              "  daphne example.daphne\n"
              "  daphne --vec example.daphne x=1 y=2.2 z=\"foo\"\n"
              "  daphne --vec --args x=1,y=2.2,z=\"foo\" example.daphne\n"
              "  daphne --vec --args x=1,y=2.2 example.daphne z=\"foo\"\n");
    SetVersionPrinter(&printVersion);
    ParseCommandLineOptions(argc, argv,
                            "The DAPHNE Prototype.\n\nThis program compiles "
                            "and executes a DaphneDSL script.\n");

    // ************************************************************************
    // Process parsed arguments
    // ************************************************************************

    try {
        if (configFile != configFileInitValue && ConfigParser::fileExists(configFile)) {
            ConfigParser::readUserConfig(configFile, user_config);
        }
    } catch (std::exception &e) {
        logErrorDaphneLibAware(daphneLibRes, "Parser error while reading user config:\n" + std::string(e.what()));
        return StatusCode::PARSER_ERROR;
    }

    // initialize logging facility
    if (not logger)
        logger = std::make_unique<DaphneLogger>(user_config);

    user_config.use_vectorized_exec = useVectorizedPipelines;
    user_config.use_vectorized_restricted = useVectorizedRestricted;
    user_config.use_distributed = useDistributedRuntime;
    user_config.use_obj_ref_mgnt = !noObjRefMgnt;
    user_config.use_ipa_const_propa = !noIPAConstPropa;
    user_config.use_phy_op_selection = !noPhyOpSelection;
    user_config.use_mlir_codegen = mlirCodegen;
    user_config.matmul_vec_size_bits = matmul_vec_size_bits;
    user_config.matmul_tile = matmul_tile;
    user_config.matmul_unroll_factor = matmul_unroll_factor;
    user_config.matmul_unroll_jam_factor = matmul_unroll_jam_factor;
    user_config.matmul_num_vec_registers = matmul_num_vec_registers;
    user_config.matmul_invert_loops = matmul_invert_loops;
    if (matmul_fixed_tile_sizes.size() > 0) {
        user_config.matmul_use_fixed_tile_sizes = true;
        user_config.matmul_fixed_tile_sizes = matmul_fixed_tile_sizes;
        // Specifying a fixed tile size will be interpreted as wanting to use tiling.
        user_config.matmul_tile = true;
    }
    user_config.use_mlir_hybrid_codegen = performHybridCodegen;

    if (!libDir.getValue().empty())
        user_config.libdir = libDir.getValue();
    user_config.resolveLibDir();

    user_config.taskPartitioningScheme = taskPartitioningScheme;
    user_config.queueSetupScheme = queueSetupScheme;
    user_config.victimSelection = victimSelection;

    // only overwrite with non-defaults
    if (numberOfThreads != 0) {
        spdlog::trace("Overwriting config file supplied numberOfThreads={} with command line argument --num-threads={}",
                      user_config.numberOfThreads, static_cast<int>(numberOfThreads));
        user_config.numberOfThreads = numberOfThreads;
    }

    user_config.minimumTaskSize = minimumTaskSize;
    user_config.phyBatchSize = phyBatchSize;
    user_config.pinWorkers = pinWorkers;
    user_config.hyperthreadingEnabled = hyperthreadingEnabled;
    user_config.debugMultiThreading = debugMultiThreading;
    user_config.prePartitionRows = prePartitionRows;
    user_config.distributedBackEndSetup = distributedBackEndSetup;
    user_config.use_columnar = useColumnar;
    if (useOptimisticSplitting && !useColumnar)
        throw std::runtime_error("--optimistic-splitting requires --columnar");
    else
        user_config.use_optimistic_splitting = useOptimisticSplitting;
    if (user_config.use_distributed) {
        if (user_config.distributedBackEndSetup != ALLOCATION_TYPE::DIST_MPI &&
            user_config.distributedBackEndSetup != ALLOCATION_TYPE::DIST_GRPC_SYNC &&
            user_config.distributedBackEndSetup != ALLOCATION_TYPE::DIST_GRPC_ASYNC)
            spdlog::warn("No backend has been selected. Wiil use the default 'MPI'");
    }
    user_config.max_distributed_serialization_chunk_size = maxDistrChunkSize;

    // only overwrite with non-defaults
    if (use_hdfs) {
        user_config.use_hdfs = use_hdfs;
    }
    if (hdfs_Address != "") {
        user_config.hdfs_Address = hdfs_Address;
    }
    if (hdfs_username != "") {
        user_config.hdfs_username = hdfs_username;
    }
    if (user_config.use_hdfs && (user_config.hdfs_Address == "" || user_config.hdfs_username == "")) {
        spdlog::warn("HDFS is enabled, but the HDFS IP address or username "
                     "were not provided.");
    }
#ifndef USE_HDFS
    if (user_config.use_hdfs) {
        throw std::runtime_error("you are trying to use HDFS, but Daphne was "
                                 "not build with --hdfs option\n");
    }
#endif

    for (auto explain : explainArgList) {
        switch (explain) {
        case columnar:
            user_config.explain_columnar = true;
            break;
        case kernels:
            user_config.explain_kernels = true;
            break;
        case llvm:
            user_config.explain_llvm = true;
            break;
        case parsing:
            user_config.explain_parsing = true;
            break;
        case parsing_simplified:
            user_config.explain_parsing_simplified = true;
            break;
        case property_inference:
            user_config.explain_property_inference = true;
            break;
        case select_matrix_repr:
            user_config.explain_select_matrix_repr = true;
            break;
        case transfer_data_props:
            user_config.explain_transfer_data_props = true;
            break;
        case sql:
            user_config.explain_sql = true;
            break;
        case phy_op_selection:
            user_config.explain_phy_op_selection = true;
            break;
        case type_adaptation:
            user_config.explain_type_adaptation = true;
            break;
        case vectorized:
            user_config.explain_vectorized = true;
            break;
        case obj_ref_mgnt:
            user_config.explain_obj_ref_mgnt = true;
            break;
        case codegen:
            user_config.explain_mlir_codegen = true;
            break;
        case mlir_codegen_sparsity_exploiting_op_fusion:
            user_config.explain_mlir_codegen_sparsity_exploiting_op_fusion = true;
            break;
        case mlir_codegen_daphneir_to_mlir:
            user_config.explain_mlir_codegen_daphneir_to_mlir = true;
            break;
        case mlir_codegen_mlir_specific:
            user_config.explain_mlir_codegen_mlir_specific = true;
            break;
        }
    }

    user_config.enable_property_recording = enablePropertyRecording;
    user_config.enable_property_insert = enablePropertyInsert;
    user_config.properties_file_path = propertiesFilePath.getValue();
    if (user_config.enable_property_recording && user_config.enable_property_insert)
        throw std::runtime_error("--enable-property-recording and --enable-property-insert are mutually exclusive, "
                                 "specify at most one of them");

    user_config.enable_statistics = enableStatistics;

    if (user_config.use_distributed && distributedBackEndSetup == ALLOCATION_TYPE::DIST_MPI) {
#ifndef USE_MPI
        throw std::runtime_error("you are trying to use the MPI backend. But, "
                                 "Daphne was not build with --mpi option\n");
#else
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, id);
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size <= 1) {
            throw std::runtime_error("you need to rerun with at least 2 MPI "
                                     "ranks (1 Master + 1 Worker)\n");
        }
        if (*id != COORDINATOR) {
            return *id;
        }
#endif
    }
    if (cuda) {
        int device_count = 0;
#ifdef USE_CUDA
        CHECK_CUDART(cudaGetDeviceCount(&device_count));
#endif
        if (device_count < 1)
            spdlog::warn("CUDA ops requested by user option but no suitable "
                         "device found");
        else {
            user_config.use_cuda = true;
        }
    }

    if (fpgaopencl) {
        user_config.use_fpgaopencl = true;
    }

    if (enableProfiling) {
#ifndef USE_PAPI
        throw std::runtime_error("you are trying to use profiling, but daphne "
                                 "was built with --no-papi\n");
#else
        user_config.enable_profiling = true;
#endif
    }

    // For DaphneLib (Python API).
    user_config.result_struct = daphneLibRes;

    // Extract script args.
    unordered_map<string, string> scriptArgsFinal;
    try {
        parseScriptArgs(scriptArgs2, scriptArgsFinal);
        parseScriptArgs(scriptArgs1, scriptArgsFinal);
    } catch (exception &e) {
        logErrorDaphneLibAware(daphneLibRes, "Parser error: " + std::string(e.what()));
        return StatusCode::PARSER_ERROR;
    }

    // ************************************************************************
    // Create DaphneIrExecutor and get MLIR context
    // ************************************************************************

    // Creates an MLIR context and loads the required MLIR dialects.
    DaphneIrExecutor executor(selectMatrixRepr, user_config);
    mlir::MLIRContext *mctx = executor.getContext();

    // ************************************************************************
    // Populate FileIO extension catalog
    // ************************************************************************
    // --- FileIO catalogs (fully automatic) ---
    FileIOCatalogParser fileIOParser;
    FileIORegistry &registry = executor.getUserConfig().registry;

    try{
        const std::string builtinsRel =
            "scripts/examples/extensions/builtInIO/BuiltIns.json";

        // resolve and load built-ins
        fileIOParser.parseFileIOCatalog(resolveRepoPathAuto(builtinsRel), registry, std::nullopt);
        registry.captureBaseline();

        // resolve and load optional --FileIO-ext <path[:prio]>
        if (!FileIOExt.empty()) {
            std::string arg = FileIOExt;
            std::optional<int> prio = std::nullopt;

            if (auto pos = arg.rfind(':'); pos != std::string::npos) {
                size_t idx = 0; long long v = std::stoll(arg.substr(pos+1), &idx);
                if (idx != arg.size() - (pos+1))
                    throw std::runtime_error("invalid priority in --FileIO-ext: " + FileIOExt);
                prio = static_cast<int>(v);
                arg.resize(pos);
            }

            const std::string path = resolveRepoPathAuto(arg);
            fileIOParser.parseFileIOCatalog(path, registry, prio);
        }
    }
    catch (const std::exception &e) {
        std::cerr << "Error parsing FileIO catalog: " << e.what() << "\n";
        return 1;
    }


    // ************************************************************************
    // Populate kernel extension catalog
    // ************************************************************************

    KernelCatalog &kc = executor.getUserConfig().kernelCatalog;
    try {
        // kc.dump();
        KernelCatalogParser kcp(mctx);
        kcp.parseKernelCatalog(user_config.libdir + "/catalog.json", kc, 0);
        if (user_config.use_cuda)
            kcp.parseKernelCatalog(user_config.libdir + "/CUDAcatalog.json", kc, 0);
        // kc.dump();
        if (!kernelExt.empty()) {
            std::string extCatalogFile;
            int64_t extPriority;

            const std::string prioritySep = ":";
            const size_t pos = kernelExt.rfind(prioritySep);
            if (pos != std::string::npos) { // a priority was specified for the extension
                extCatalogFile = kernelExt.substr(0, pos);
                const std::string extPriorityStr(kernelExt.substr(pos + prioritySep.size()));
                try {
                    size_t idx;
                    extPriority = std::stoll(extPriorityStr, &idx);
                    if (idx != extPriorityStr.size())
                        // stoll() did not consume all characters in extPriorityStr, there is some non-integer part at
                        // the end of the string.
                        throw std::runtime_error(""); // the error message is generated in the catch-block below
                } catch (std::exception &e) {
                    throw std::runtime_error("invalid priority for kernel extension, expected an integer after the '" +
                                             prioritySep + "', but found '" + extPriorityStr + "': '" + kernelExt +
                                             "'");
                }
            } else { // no priority was specified for the extension
                extCatalogFile = kernelExt;
                extPriority = 0;
            }

            kcp.parseKernelCatalog(extCatalogFile, kc, extPriority);
        }
    } catch (std::exception &e) {
        logErrorDaphneLibAware(daphneLibRes, "Parser error: " + std::string(e.what()));
        return StatusCode::PARSER_ERROR;
    } catch (...) {
        logErrorDaphneLibAware(daphneLibRes, "Parser error: Unknown exception");
        return StatusCode::PARSER_ERROR;
    }

    // ************************************************************************
    // Parse, compile and execute DaphneDSL script
    // ************************************************************************

    clock::time_point tpBegPars = clock::now();

    // Create an OpBuilder and an MLIR module and set the builder's insertion
    // point to the module's body, such that subsequently created DaphneIR
    // operations are inserted into the module.
    OpBuilder builder(mctx);
    auto loc = mlir::FileLineColLoc::get(builder.getStringAttr(inputFile), 0, 0);
    auto moduleOp = ModuleOp::create(loc);
    auto *body = moduleOp.getBody();
    builder.setInsertionPoint(body, body->begin());

    // Parse the input file and generate the corresponding DaphneIR operations
    // inside the module, assuming DaphneDSL as the input format.
    DaphneDSLParser parser(scriptArgsFinal, user_config);
    try {
        parser.parseFile(builder, inputFile);
    } catch (std::exception &e) {
        logErrorDaphneLibAware(daphneLibRes, "While parsing: " + std::string(e.what()));
        return StatusCode::PARSER_ERROR;
    }

    clock::time_point tpBegComp = clock::now();

    // Further, process the module, including optimization and lowering passes.
    try {
        if (!executor.runPasses(moduleOp)) {
            return StatusCode::PASS_ERROR;
        }
    } catch (std::exception &e) {
        logErrorDaphneLibAware(daphneLibRes, "Lowering pipeline error.{}\nPassManager failed module lowering, "
                                             "responsible IR written to module_fail.log.\n" +
                                                 std::string(e.what()));
        return StatusCode::PASS_ERROR;
    } catch (...) {
        logErrorDaphneLibAware(daphneLibRes, "Lowering pipeline error: Unknown exception");
        return StatusCode::PASS_ERROR;
    }

    // JIT-compile the module and execute it.
    // module->dump(); // print the LLVM IR representation
    clock::time_point tpBegExec;
    std::unique_ptr<mlir::ExecutionEngine> engine;
    try {
        engine = executor.createExecutionEngine(moduleOp);
        tpBegExec = clock::now();

        // set jump address for catching exceptions in kernel libraries via
        // signal handling
        if (setjmp(return_from_handler) == 0) {
            // Call main directly by looking up function pointer since it takes no args/returns void
            auto expectedFPtr = engine->lookup("main");
            if (!expectedFPtr) {
                llvm::errs() << "Failed to lookup main function: " << expectedFPtr.takeError();
                return StatusCode::EXECUTION_ERROR;
            }
            auto mainFn = reinterpret_cast<void (*)()>(*expectedFPtr);
            mainFn();
        } else {
            logErrorDaphneLibAware(daphneLibRes, "Got an abort signal from the execution engine. Most likely an "
                                                 "exception in a shared library. Check logs!\n"
                                                 "Execution error: Returning from signal " +
                                                     std::to_string(gSignalStatus));
            return StatusCode::EXECUTION_ERROR;
        }
        // Explicitly reset the engine before any other cleanup happens
        engine.reset();
    } catch (std::runtime_error &re) {
        logErrorDaphneLibAware(daphneLibRes, "Execution error: " + std::string(re.what()));
        return StatusCode::EXECUTION_ERROR;
    } catch (std::exception &e) {
        logErrorDaphneLibAware(daphneLibRes, "Execution error " + std::string(e.what()));
        return StatusCode::EXECUTION_ERROR;
    }
    clock::time_point tpEnd = clock::now();

    if (timing) {
        // Calculate durations of the individual high-level steps of DAPHNE.
        double durStrt = chrono::duration_cast<chrono::duration<double>>(tpBegPars - tpBeg).count();
        double durPars = chrono::duration_cast<chrono::duration<double>>(tpBegComp - tpBegPars).count();
        double durComp = chrono::duration_cast<chrono::duration<double>>(tpBegExec - tpBegComp).count();
        double durExec = chrono::duration_cast<chrono::duration<double>>(tpEnd - tpBegExec).count();
        double durTotal = chrono::duration_cast<chrono::duration<double>>(tpEnd - tpBeg).count();
        // ToDo: use logger
        // Output durations in JSON.
        std::cerr << "{";
        std::cerr << "\"startup_seconds\": " << durStrt << ", ";
        std::cerr << "\"parsing_seconds\": " << durPars << ", ";
        std::cerr << "\"compilation_seconds\": " << durComp << ", ";
        std::cerr << "\"execution_seconds\": " << durExec << ", ";
        std::cerr << "\"total_seconds\": " << durTotal;
        std::cerr << "}" << std::endl;
    }

    if (user_config.enable_statistics)
        Statistics::instance().dumpStatistics(KernelDispatchMapping::instance());

    if (user_config.enable_property_recording)
        PropertyLogger::instance().savePropertiesAsJson(user_config.properties_file_path);

    // explicitly destroying the moduleOp here due to valgrind complaining about
    // a memory leak otherwise.
    moduleOp->destroy();
    return StatusCode::SUCCESS;
}

int mainInternal(int argc, const char **argv, DaphneLibResult *daphneLibRes) {
    int id = -1; // this  -1 would not change if the user did not select mpi
                 // backend during execution

    // Initialize user configuration.
    DaphneUserConfig user_config{};

    int res = startDAPHNE(argc, argv, daphneLibRes, &id, user_config);

#ifdef USE_MPI
    if (id == COORDINATOR) {
        int size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        unsigned char terminateMessage = 0x00;
        for (int i = 1; i < size; i++) {
            MPI_Send(&terminateMessage, 1, MPI_UNSIGNED_CHAR, i, DETACH, MPI_COMM_WORLD);
        }
        MPI_Finalize();
    } else if (id > -1) {
        MPIWorker worker(user_config);
        worker.joinComputingTeam();
        res = StatusCode::SUCCESS;
        MPI_Finalize();
    }
#endif

    return res;
}
