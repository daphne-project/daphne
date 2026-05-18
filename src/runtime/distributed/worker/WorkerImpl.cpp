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

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include <llvm/Support/SourceMgr.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <ir/daphneir/Daphne.h>
#include <parser/catalog/KernelCatalogParser.h>

#include "WorkerImpl.h"

#include <compiler/execution/DaphneIrExecutor.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/io/File.h>
#include <runtime/local/io/ReadCsv.h>
#include <runtime/local/kernels/Read.h>

#include <stdexcept>

const std::string WorkerImpl::DISTRIBUTED_FUNCTION_NAME = "dist";

WorkerImpl::WorkerImpl(DaphneUserConfig &_cfg) : cfg(_cfg), tmp_file_counter_(0), localData_() {}

static Frame *emptyFrame = DataObjectFactory::create<Frame>(0, 0, nullptr, nullptr, false);

template <> WorkerImpl::StoredInfo WorkerImpl::Store<Structure>(Structure *mat) {
    auto identifier = "tmp_" + std::to_string(tmp_file_counter_++);
    localData_[identifier] = mat;
    return StoredInfo({identifier, mat->getNumRows(), mat->getNumCols()});
}
template <> WorkerImpl::StoredInfo WorkerImpl::Store<double>(double *val) {
    auto identifier = "tmp_" + std::to_string(tmp_file_counter_++);
    // The vectorized engine expects as input, a pointer value
    // to the memory holding a value. Therefore we need to allocate memory
    // and save the value of the pointer to that address.

    // TODO: We need to implement a free operation (this applies to
    // objects/matrices too). It's probably best that coordinator decides when
    // memory should be freed, (either when requesting data or by specifing a
    // "FreeMemory" RPC call).
    double *valPtr = new double(*val);
    localData_[identifier] = valPtr;
    return StoredInfo({identifier, 0, 0});
}

WorkerImpl::Status WorkerImpl::Compute(std::vector<WorkerImpl::StoredInfo> *outputs,
                                       const std::vector<WorkerImpl::StoredInfo> &inputs, const std::string &mlirCode) {
    cfg.use_vectorized_exec = true;
    cfg.use_distributed = false;

    // TODO Decide if vectorized pipelines should be used on this worker.
    // TODO Decide if selectMatrixReprs should be used on this worker.
    // TODO Once we hand over longer pipelines to the workers, we might not
    // want to hardcode insertFreeOp to false anymore. But maybe we will insert
    // the FreeOps at the coordinator already.
    DaphneIrExecutor executor(false, cfg);

    KernelCatalog &kc = executor.getUserConfig().kernelCatalog;
    KernelCatalogParser kcp(executor.getContext());
    kcp.parseKernelCatalog(cfg.libdir + "/catalog.json", kc, 0);
    if (executor.getUserConfig().use_cuda)
        kcp.parseKernelCatalog(cfg.libdir + "/CUDAcatalog.json", kc, 0);

    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::parseSourceString<mlir::ModuleOp>(mlirCode, executor.getContext()));
    if (!module) {
        auto message = "Failed to parse source string.\n";
        llvm::errs() << message;
        return WorkerImpl::Status(false, message);
    }

    auto *distOp = module->lookupSymbol(DISTRIBUTED_FUNCTION_NAME);
    mlir::func::FuncOp distFunc;
    if (!(distFunc = llvm::dyn_cast_or_null<mlir::func::FuncOp>(distOp))) {
        auto message = "MLIR fragment has to contain `dist` FuncOp\n";
        llvm::errs() << message;
        return WorkerImpl::Status(false, message);
    }
    auto distFuncTy = distFunc.getFunctionType();

    std::vector<void *> inputsObj;
    std::vector<void *> outputsObj;
    auto packedInputsOutputs = createPackedCInterfaceInputsOutputs(distFuncTy, inputs, outputsObj, inputsObj);

    // Increase the reference counters of all inputs to the `dist` function.
    // (But only consider data objects, not scalars.)
    // This is necessary to avoid them from being destroyed within the
    // function. Note that this increasing is in-line with the treatment of
    // local function calls, where we also increase the inputs' reference
    // counters before the call, for the same reason. See ManageObjsRefsPass
    // for details.
    for (size_t i = 0; i < inputsObj.size(); i++)
        // TODO Use CompilerUtils::isObjType() once this branch has been
        // rebased. if(CompilerUtils::isObjType(distFuncTy.getInput(i)))
        if (llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType>(distFuncTy.getInput(i)))
            reinterpret_cast<Structure *>(inputsObj[i])->increaseRefCounter();

    // Execution
    // TODO Before we run the passes, we should insert information on shape
    // (and potentially other properties) into the types of the arguments of
    // the DISTRIBUTED_FUNCTION_NAME function. At least the shape can be
    // obtained from the cached data partitions in localData_. Then, shape
    // inference etc. should work within this function.
    if (!executor.runPasses(module.get())) {
        std::stringstream ss;
        ss << "Module Pass Error.\n";
        // module->print(ss, llvm::None);
        llvm::errs() << ss.str();
        return WorkerImpl::Status(false, ss.str());
    }

    mlir::registerLLVMDialectTranslation(*module->getContext());

    auto engine = executor.createExecutionEngine(module.get());
    if (!engine) {
        return WorkerImpl::Status(false, std::string("Failed to create JIT-Execution engine"));
    }
    auto error = engine->invokePacked(DISTRIBUTED_FUNCTION_NAME,
                                      llvm::MutableArrayRef<void *>{&packedInputsOutputs[0], (size_t)0});

    if (error) {
        std::stringstream ss("JIT-Engine invocation failed.");
        llvm::errs() << "JIT-Engine invocation failed: " << error << '\n';
        return WorkerImpl::Status(false, ss.str());
    }

    for (auto zipped : llvm::zip(outputsObj, distFuncTy.getResults())) {
        auto output = std::get<0>(zipped);

        auto identification = "tmp_" + std::to_string(tmp_file_counter_++);
        localData_[identification] = output;

        auto mat = static_cast<Structure *>(output);

        outputs->push_back(StoredInfo({identification, mat->getNumRows(), mat->getNumCols()}));
    }
    // TODO: cache management (Write to file/evict matrices present as files)
    return WorkerImpl::Status(true);
}

// distributed::WorkData::DataCase WorkerImpl::dataCaseForType(mlir::Type type)
// {
//     distributed::WorkData::DataCase dataCase;
//     if (llvm::isa<mlir::daphne::MatrixType>(type)) {
//         dataCase = distributed::WorkData::kStored;
//     }
//     else {
//         // TODO: further types data cases
//         throw std::runtime_error("WorlerImpl: TODO: further types data
//         cases");
//     }
//     return dataCase;
// }

Structure *WorkerImpl::Transfer(StoredInfo info) {
    Structure *mat = readOrGetMatrix(info.identifier, info.numRows, info.numCols);
    return mat;
}

std::vector<void *> WorkerImpl::createPackedCInterfaceInputsOutputs(mlir::FunctionType functionType,
                                                                    std::vector<WorkerImpl::StoredInfo> workInputs,
                                                                    std::vector<void *> &outputs,
                                                                    std::vector<void *> &inputs) {
    if (static_cast<size_t>(functionType.getNumInputs()) != workInputs.size())
        throw std::runtime_error("WorkerImpl: Number of inputs received have "
                                 "to match number of MLIR fragment inputs");
    std::vector<void *> inputsAndOutputs;

    // No realloc is allowed to happen, otherwise the pointers are invalid
    inputs.reserve(workInputs.size());
    outputs.reserve(functionType.getNumResults());

    for (const auto &typeAndWorkInput : llvm::zip(functionType.getInputs(), workInputs)) {
        auto type = std::get<0>(typeAndWorkInput);
        auto workInput = std::get<1>(typeAndWorkInput);

        inputs.push_back(loadWorkInputData(type, workInput));
        inputsAndOutputs.push_back(&inputs.back());
    }

    //    for (const auto &type : functionType.getResults()) {
    for (auto i = 0ul; i < functionType.getResults().size(); ++i) {
        outputs.push_back(nullptr);
        inputsAndOutputs.push_back(&outputs.back());
    }
    return inputsAndOutputs;
}

void *WorkerImpl::loadWorkInputData(mlir::Type mlirType, StoredInfo &workInput) {
    // TODO: all types
    bool isSparse = false;
    bool isFloat = false;
    bool isScalar = false;
    if (llvm::isa<mlir::daphne::MatrixType>(mlirType)) {
        auto matTy = llvm::dyn_cast<mlir::daphne::MatrixType>(mlirType);
        isSparse = matTy.getRepresentation() == mlir::daphne::MatrixRepresentation::Sparse;
        isFloat = llvm::isa<mlir::Float64Type>(matTy.getElementType());
    } else
        isScalar = true;
    return readOrGetMatrix(workInput.identifier, workInput.numRows, workInput.numCols, isSparse, isFloat, isScalar);
}

Structure *WorkerImpl::readOrGetMatrix(const std::string &identifier, size_t numRows, size_t numCols,
                                       bool isSparse /*= false */, bool isFloat /* = false*/,
                                       bool isScalar /* = false */) {
    auto data_it = localData_.find(identifier);
    if (data_it != localData_.end()) {
        // Data already cached
        if (isScalar) {
            auto valAddress = (double *)(data_it->second);
            auto structurePtrPtr = (Structure **)valAddress;
            return (*structurePtrPtr);
        } else
            return static_cast<Structure *>(data_it->second);
    } else {
        // Data not yet loaded -> load from file
        Structure *m = nullptr;
        // TODO do we need to check for sparsity here? Why Dense and CSR use
        // different read method?
        if (isSparse) {
            if (isFloat) {
                CSRMatrix<double> *m2 = nullptr;
                read<CSRMatrix<double>>(m2, identifier.c_str(),emptyFrame, nullptr);
                m = m2;
            } else {
                CSRMatrix<int64_t> *m2 = nullptr;
                read<CSRMatrix<int64_t>>(m2, identifier.c_str(), emptyFrame, nullptr);
                m = m2;
            }
        } else {
            struct File *file = openFile(identifier.c_str());
            char delim = ',';
            // TODO use read
            if (isFloat) {
                DenseMatrix<double> *m2 = nullptr;
                readCsvFile<DenseMatrix<double>>(m2, file, numRows, numCols, delim);
                m = m2;
            } else {
                DenseMatrix<int64_t> *m2 = nullptr;
                readCsvFile<DenseMatrix<int64_t>>(m2, file, numRows, numCols, delim);
                m = m2;
            }
            closeFile(file);
        }

        if (!localData_.insert({identifier, m}).second)
            throw std::runtime_error("WorkerImpl: Value should always be inserted");

        return m;
    }
}

// grpc::Status WorkerImpl::FreeMem(::grpc::ServerContext *context,
//                                const ::distributed::StoredData *request,
//                                ::distributed::Empty *emptyMessg)
// {
//     // switch (request->type())
//     // {
//     // case distributed::StoredData::Type::StoredData_Type_DenseMatrix_i64:
//     // case distributed::StoredData::Type::StoredData_Type_CSRMatrix_i64:
//     //     return FreeMemType<int64_t>(context, request, emptyMessg);
//     //     break;
//     // case distributed::StoredData::Type::StoredData_Type_DenseMatrix_f64:
//     // case distributed::StoredData::Type::StoredData_Type_CSRMatrix_f64:
//     //     return FreeMemType<double>(context, request, emptyMessg);
//     //     break;
//     // default:
//     //     break;
//     // }
//     // return grpc::Status::CANCELLED;
// }

// template<typename VT>
// grpc::Status WorkerImpl::FreeMemType(::grpc::ServerContext *context,
//                                const ::distributed::StoredData *request,
//                                ::distributed::Empty *emptyMessg)
// {
//     auto identifier = request->identifier();
//     auto data_it = localData_.find(identifier);

//     if (data_it != localData_.end()) {
//         auto * mat = reinterpret_cast<Matrix<VT> *>(data_it->second);
//         if(auto m = dynamic_cast<DenseMatrix<VT> *>(mat))
//             DataObjectFactory::destroy(m);
//         else if(auto m = dynamic_cast<CSRMatrix<VT> *>(mat))
//             DataObjectFactory::destroy(m);
//         localData_.erase(identifier);
//     }
//     return grpc::Status::OK;
// }
