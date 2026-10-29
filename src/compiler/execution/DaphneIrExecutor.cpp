/*
 *  Copyright 2021 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "DaphneIrExecutor.h"
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SparseTensor/Pipelines/Passes.h>
#include <mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/SparseTensor/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <util/ErrorHandler.h>

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h.inc>
#include <ir/daphneir/Passes.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>

#include <filesystem>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

DaphneIrExecutor::DaphneIrExecutor(bool selectMatrixRepresentations, DaphneUserConfig cfg)
    : userConfig_(std::move(cfg)), selectMatrixRepresentations_(selectMatrixRepresentations) {
    // register loggers
    if (userConfig_.log_ptr != nullptr)
        userConfig_.log_ptr->registerLoggers();

    mlir::DialectRegistry registry;
    mlir::LLVM::registerInlinerInterface(registry);
    mlir::func::registerInlinerExtension(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);

    context_.appendDialectRegistry(registry);

    context_.getOrLoadDialect<mlir::daphne::DaphneDialect>();
    context_.getOrLoadDialect<mlir::arith::ArithDialect>();
    context_.getOrLoadDialect<mlir::func::FuncDialect>();
    context_.getOrLoadDialect<mlir::scf::SCFDialect>();
    context_.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context_.getOrLoadDialect<mlir::affine::AffineDialect>();
    context_.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context_.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context_.getOrLoadDialect<mlir::math::MathDialect>();
    context_.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    context_.getOrLoadDialect<mlir::sparse_tensor::SparseTensorDialect>();

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
}

bool DaphneIrExecutor::runPasses(mlir::ModuleOp module) {
    // FIXME: operations in `template` functions (functions with unknown inputs)
    // can't be verified
    //  as their type constraints are not met.
    // if (failed(mlir::verify(module))) {
    // module->emitError("failed to verify the module right after parsing");
    // return false;
    //}

    if (!module)
        return false;

    // This flag is really useful to figure out why the lowering failed
    llvm::DebugFlag = userConfig_.debug_llvm;

    mlir::PassManager pm(&context_);
    // TODO Enable the verifier for all passes where it is possible.
    // Originally, it was only turned off for the
    // SpecializeGenericFunctionsPass.
    pm.enableVerifier(false);

    if (userConfig_.explain_parsing)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after parsing:"));

    // First simplification pass right after parsing: canonicalize, then CSE to
    // drop the duplicate ops it exposes.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    if (userConfig_.explain_parsing_simplified)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after parsing and some simplifications:"));

    pm.addPass(mlir::daphne::createRewriteSqlOpPass()); // calls SQL Parser
    if (userConfig_.explain_sql)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after SQL parsing:"));

    pm.addPass(mlir::daphne::createSpecializeGenericFunctionsPass(userConfig_));
    if (userConfig_.explain_property_inference)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after inference:"));

    // Note that property inference and canonicalization have already been done
    // in the SpecializeGenericFunctionsPass, so actually, it's not necessary
    // here anymore.

    // TODO There is a cyclic dependency between (shape) inference and
    // constant folding (included in canonicalization), at the moment we
    // run only three iterations of both passes (see #173).
    pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::daphne::createAlgebraicSimplifyPass());

    if (userConfig_.enable_property_recording)
        pm.addPass(mlir::daphne::createRecordPropertiesPass());
    if (userConfig_.enable_property_insert) {
        pm.addPass(mlir::daphne::createInsertPropertiesPass(userConfig_.properties_file_path));
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createInferencePass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    }

    if (userConfig_.use_columnar) {
        // Rewrite certain matrix/frame ops from linear/relational algebra to columnar ops from column algebra.
        pm.addPass(mlir::daphne::createRewriteToColumnarOpsPass(userConfig_));
        // Infer the result types of the newly created columnar ops.
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createInferencePass());
        // Simplify the IR.
        pm.addPass(mlir::createCanonicalizerPass());
        // Remove duplicate ops exposed by the columnar rewrite and
        // canonicalization. One CSE pass is enough (was previously looped
        // five times): CSE visits each op after its operands, so a whole
        // duplicated chain collapses in a single pass.
        pm.addPass(mlir::createCSEPass());
    }
    if (userConfig_.explain_columnar)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after lowering to columnar ops:"));

    if (selectMatrixRepresentations_) {
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createSelectMatrixRepresentationsPass(userConfig_));
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    }
    if (userConfig_.explain_select_matrix_repr)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after selecting matrix representations:"));

    // pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createTransferDataPropertiesPass());
    if (userConfig_.explain_transfer_data_props)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after transferring data properties:"));

    if (userConfig_.use_phy_op_selection) {
        pm.addPass(mlir::daphne::createPhyOperatorSelectionPass());
        pm.addPass(mlir::createCSEPass());
    }
    if (userConfig_.explain_phy_op_selection)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after selecting physical operators:"));

    pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createAdaptTypesToKernelsPass());
    if (userConfig_.explain_type_adaptation)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after type adaptation:"));

    // For now, in order to use the distributed runtime we also require the
    // vectorized engine to be enabled to create pipelines. Therefore, *if*
    // distributed runtime is enabled, we need to make a vectorization pass.
    if (userConfig_.use_vectorized_exec || userConfig_.use_distributed) {
        // TODO: add inference here if we have rewrites that could apply to
        // vectorized pipelines due to smaller sizes
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::daphne::createVectorizeComputationsPass(userConfig_.use_vectorized_restricted));
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createHorizontalFusionPass());
        pm.addPass(mlir::createCanonicalizerPass());
    }
    if (userConfig_.explain_vectorized)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after vectorization:"));

    if (userConfig_.use_distributed)
        pm.addPass(mlir::daphne::createDistributePipelinesPass());

    if (userConfig_.enable_profiling)
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createProfilingPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createInsertDaphneContextPass(userConfig_));

#ifdef USE_CUDA
    if (userConfig_.use_cuda)
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createMarkCUDAOpsPass(userConfig_));
#endif

#ifdef USE_FPGAOPENCL
    if (userConfig_.use_fpgaopencl)
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createMarkFPGAOPENCLOpsPass(userConfig_));
#endif

    // Tidy up the IR before managing object reference counters with IncRefOp
    // and DecRefOp. This is important, because otherwise, an SSA value whose
    // references are managed could be cleared away by common subexpression
    // elimination (CSE), while retaining its IncRefOps/DecRefOps, which could
    // lead to double frees etc.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    if (userConfig_.use_obj_ref_mgnt)
        pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createManageObjRefsPass());
    if (userConfig_.explain_obj_ref_mgnt)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after managing object references:"));

    if (userConfig_.use_mlir_codegen || userConfig_.use_mlir_hybrid_codegen)
        buildCodegenPipeline(pm);

    pm.addNestedPass<mlir::func::FuncOp>(mlir::daphne::createRewriteToCallKernelOpPass(userConfig_, usedLibPaths));
    if (userConfig_.explain_kernels)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after kernel lowering:"));

    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::LLVM::createLLVMRequestCWrappersPass());
    pm.addPass(mlir::daphne::createLowerToLLVMPass(userConfig_));
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (userConfig_.explain_llvm)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after llvm lowering:"));

    // Initialize the use of each distinct kernels library to false.
    usedLibPaths = userConfig_.kernelCatalog.getLibPaths();

    try {
        if (failed(pm.run(module))) {
            module->dump();
            module->emitError("module pass error");
            return false;
        }
    } catch (...) {
        ErrorHandler::dumpModuleToDisk(module);
        throw;
    }

    return true;
}

std::unique_ptr<mlir::ExecutionEngine> DaphneIrExecutor::createExecutionEngine(mlir::ModuleOp module) {
    if (!module)
        return nullptr;
    // An optimization pipeline to use within the execution engine.
    unsigned optLevel = 0;
    unsigned sizeLevel = 0;
    llvm::TargetMachine *targetMachine = nullptr;
    auto optPipeline = mlir::makeOptimizingTransformer(optLevel, sizeLevel, targetMachine);

    // Determine the actually used kernels libraries.
    std::vector<llvm::StringRef> sharedLibRefs;
    for (auto it = usedLibPaths.begin(); it != usedLibPaths.end(); it++)
        if (it->second) {
            std::string usedLibPath = it->first;
            sharedLibRefPaths.push_back(usedLibPath);
            sharedLibRefs.emplace_back(sharedLibRefPaths.back());

            // Check if the used kernels library really exists at the expected
            // path and throw an understandable error, otherwise.
            if (!std::filesystem::exists(usedLibPath))
                throw std::runtime_error("the shared library `" + usedLibPath +
                                         "` is needed for some kernel, but the file does not exist");
        }

    registerLLVMDialectTranslation(context_);
    mlir::registerBuiltinDialectTranslation(context_);
    // module.dump();
    mlir::ExecutionEngineOptions options;
    options.llvmModuleBuilder = nullptr;
    options.transformer = optPipeline;
    options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
    options.sharedLibPaths = llvm::ArrayRef<llvm::StringRef>(sharedLibRefs);
    options.enableObjectDump = true;
    options.enableGDBNotificationListener = true;
    options.enablePerfNotificationListener = true;
    auto maybeEngine = mlir::ExecutionEngine::create(module, options);

    if (!maybeEngine) {
        llvm::errs() << "Failed to create JIT-Execution engine: " << maybeEngine.takeError();
        return nullptr;
    }
    return std::move(maybeEngine.get());
}

void DaphneIrExecutor::buildCodegenPipeline(mlir::PassManager &pm) {
    if (userConfig_.explain_mlir_codegen)
        pm.addPass(mlir::daphne::createPrintIRPass("IR before codegen pipeline"));

    pm.addPass(mlir::daphne::createConvertDaphneToLinalgPass());
    // pm.addPass(mlir::daphne::createPrintIRPass("IR after createConvertDaphneToLinalgPass"));
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgGeneralizeNamedOpsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());
    // Elementwise fusion often exposes duplicate memref/tensor/linalg ops;
    // run CSE once to deduplicate them before the next stage.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    // pm.addPass(mlir::daphne::createPrintIRPass("IR after LinalgPasses"));

    auto bufOpts = mlir::getBufferizationOptionsForSparsification(/*analysisOnly=*/false);
    mlir::SparsificationOptions sparseOpts; // defaults match the simple ctor
    pm.addPass(mlir::createSparsificationAndBufferizationPass(
        bufOpts, sparseOpts,
        /*createSparseDeallocs=*/false,
        /*enableRuntimeLibrary=*/false, // turn off runtime support
        /*enableBufferInitialization=*/false,
        /*vectorLength=*/0,
        /*enableVLAVectorization=*/false,
        /*enableSIMDIndex32=*/false,
        /*enableGPULibgen=*/false, mlir::SparseEmitStrategy::kFunctional, mlir::SparseParallelizationStrategy::kNone));
    // pm.addPass(mlir::createSparsificationAndBufferizationPass());
    pm.addPass(mlir::createStorageSpecifierToLLVMPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    // Canonicalization can expose further duplicates; run CSE once more so the
    // subsequent lowering sees the smallest possible IR.
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::memref::createExpandReallocPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFToControlFlowPass());

    // pm.addNestedPass<mlir::func::FuncOp>(mlir::memref::createFoldMemRefAliasOpsPass());
    pm.addPass(mlir::memref::createNormalizeMemRefsPass());
    // pm.addPass(mlir::memref::createExpandReallocPass());
    // pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (userConfig_.explain_mlir_codegen)
        pm.addPass(mlir::daphne::createPrintIRPass("IR after codegen pipeline"));
}
