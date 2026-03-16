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

#pragma once

#include <api/daphnelib/DaphneLibResult.h>
#include <compiler/catalog/KernelCatalog.h>
#include <runtime/local/datastructures/IAllocationDescriptor.h>
#include <runtime/local/vectorized/LoadPartitioningDefs.h>
#include <util/DaphneLogger.h>
#include <util/LogConfig.h>
class DaphneLogger;

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <climits>
#endif

#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

/*
 * Container to pass around user configuration
 */
struct DaphneUserConfig {
    // Remember to update UserConfig.json accordingly!

    bool use_columnar = false;
    bool use_cuda = false;
    bool use_vectorized_exec = false;
    bool use_vectorized_restricted = false;
    bool use_distributed = false;
    bool use_obj_ref_mgnt = true;
    bool use_optimistic_splitting = false;
    bool use_ipa_const_propa = true;
    bool use_phy_op_selection = true;
    bool use_mlir_codegen = false;
    int matmul_vec_size_bits = 0;
    bool matmul_tile = false;
    int matmul_unroll_factor = 1;
    int matmul_unroll_jam_factor = 4;
    int matmul_num_vec_registers = 16;
    bool matmul_use_fixed_tile_sizes = false;
    std::vector<unsigned> matmul_fixed_tile_sizes = {4, 4};
    bool matmul_invert_loops = false;
    bool use_mlir_hybrid_codegen = false;
    bool cuda_fuse_any = false;
    bool vectorized_single_queue = false;
    bool prePartitionRows = false;
    bool pinWorkers = false;
    bool hyperthreadingEnabled = false;
    bool debugMultiThreading = false;
    bool use_fpgaopencl = false;
    bool enable_profiling = false;

    bool debug_llvm = false;
    bool explain_columnar = false;
    bool explain_kernels = false;
    bool explain_llvm = false;
    bool explain_parsing = false;
    bool explain_parsing_simplified = false;
    bool explain_property_inference = false;
    bool explain_select_matrix_repr = false;
    bool explain_transfer_data_props = false;
    bool explain_sql = false;
    bool explain_phy_op_selection = false;
    bool explain_type_adaptation = false;
    bool explain_vectorized = false;
    bool explain_obj_ref_mgnt = false;
    bool explain_mlir_codegen = false;
    bool explain_mlir_codegen_sparsity_exploiting_op_fusion = false;
    bool explain_mlir_codegen_daphneir_to_mlir = false;
    bool explain_mlir_codegen_mlir_specific = false;
    bool enable_property_recording = false;
    bool enable_property_insert = false;
    std::string properties_file_path = "properties.json";
    bool enable_statistics = false;

    bool force_cuda = false;

    SelfSchedulingScheme taskPartitioningScheme = SelfSchedulingScheme::STATIC;
    QueueTypeOption queueSetupScheme = QueueTypeOption::CENTRALIZED;
    VictimSelectionLogic victimSelection = VictimSelectionLogic::SEQPRI;
    ALLOCATION_TYPE distributedBackEndSetup = ALLOCATION_TYPE::DIST_MPI; // default value
    size_t max_distributed_serialization_chunk_size =
        std::numeric_limits<int>::max() - 1024; // 2GB (-1KB to make up for gRPC headers etc.) - which is the
                                                // maximum size allowed by gRPC / MPI. TODO: Investigate what
                                                // might be the optimal.
    int numberOfThreads = -1;
    int minimumTaskSize = 1;
    size_t phyBatchSize = 96 * 1024;

    // hdfs
    bool use_hdfs = false;
    std::string hdfs_Address = "";
    std::string hdfs_username = "";

    // minimum considered log level (e.g., no logging below ERROR (essentially
    // suppressing WARN, INFO, DEBUG and TRACE)
    spdlog::level::level_enum log_level_limit = spdlog::level::err;
    std::vector<LogConfig> loggers;
    DaphneLogger *log_ptr{};
    float sparsity_threshold = 0.25;

#ifdef USE_CUDA
    // User config holds once context atm for convenience until we have proper
    // system infrastructure

    // CUDA device IDs (future work, as we create only one context atm)
    std::vector<int> cuda_devices;

    // ToDo: This is an arbitrary default taken from sample code
//    int cublas_workspace_size = 1024 * 1024 * 4;
#endif
#ifdef USE_FPGAOPENCL
    std::vector<int> fpga_devices;
#endif

    std::string libdir = "{exedir}/../lib";
    std::map<std::string, std::vector<std::string>> daphnedsl_import_paths;

    // TODO Maybe the DaphneLib result should better reside in the
    // DaphneContext, but having it here is simpler for now.
    DaphneLibResult *result_struct = nullptr;

    KernelCatalog kernelCatalog;

    /**
     * @brief Replaces the prefix `"{exedir}/"` in the field `libdir` by the
     * path of the directory in which the currently running executable resides.
     *
     * Note that the current executable is not necessarily `daphne`. It could
     * also be a distributed worker (e.g., `DistributedWorker`) or Python
     * (`python3`).
     */
    void resolveLibDir() {
        const std::string exedirPlaceholder = "{exedir}/";
        if (libdir.substr(0, exedirPlaceholder.size()) == exedirPlaceholder) {
#ifdef __APPLE__
            char pathBuf[PATH_MAX];
            uint32_t size = sizeof(pathBuf);
            _NSGetExecutablePath(pathBuf, &size);
            std::filesystem::path daphneExeDir(std::filesystem::canonical(pathBuf).parent_path());
#else
            std::filesystem::path daphneExeDir(std::filesystem::canonical("/proc/self/exe").parent_path());
#endif
            libdir = daphneExeDir / libdir.substr(exedirPlaceholder.size());
        }
    }
};
