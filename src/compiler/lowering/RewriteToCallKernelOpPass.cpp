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

#include "compiler/utils/CompilerUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include <compiler/utils/TypePrinting.h>
#include <util/ErrorHandler.h>
#include <util/KernelDispatchMapping.h>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::affine;

namespace {
class KernelReplacement : public RewritePattern {
    // TODO This method is only required since MLIR does not seem to
    // provide a means to get this information.
    static size_t getNumODSOperands(Operation *op) {
        if (llvm::isa<daphne::ThetaJoinOp>(op))
            return 4;
        if (llvm::isa<daphne::OrderOp>(op))
            return 4;
        if (llvm::isa<daphne::GroupOp>(op))
            return 3;
        if (llvm::isa<daphne::CreateFrameOp, daphne::SetColLabelsOp>(op))
            return 2;
        if (llvm::isa<daphne::DistributedComputeOp, daphne::CreateListOp>(op))
            return 1;

        throw ErrorHandler::compilerError(op, "RewriteToCallKernelOpPass",
                                          "lowering to kernel call not yet supported for this variadic "
                                          "operation: " +
                                              op->getName().getStringRef().str());
    }

    // TODO This method is only required since MLIR does not seem to
    // provide a means to get this information. But, for instance, the
    // isVariadic boolean array is automatically generated *within* the
    // getODSOperandIndexAndLength method.
    static std::tuple<unsigned, unsigned, bool> getODSOperandInfo(Operation *op, unsigned index) {
        // TODO Simplify those by a macro.
        if (auto concreteOp = llvm::dyn_cast<daphne::CreateFrameOp>(op)) {
            auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
            static bool isVariadic[] = {true, true};
            return std::make_tuple(idxAndLen.first, idxAndLen.second, isVariadic[index]);
        }
        if (auto concreteOp = llvm::dyn_cast<daphne::CreateListOp>(op)) {
            auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
            static bool isVariadic[] = {true};
            return std::make_tuple(idxAndLen.first, idxAndLen.second, isVariadic[index]);
        }
        if (auto concreteOp = llvm::dyn_cast<daphne::SetColLabelsOp>(op)) {
            auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
            static bool isVariadic[] = {false, true};
            return std::make_tuple(idxAndLen.first, idxAndLen.second, isVariadic[index]);
        }
        if (auto concreteOp = llvm::dyn_cast<daphne::DistributedComputeOp>(op)) {
            auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
            static bool isVariadic[] = {true};
            return std::make_tuple(idxAndLen.first, idxAndLen.second, isVariadic[index]);
        }
        if (auto concreteOp = llvm::dyn_cast<daphne::GroupOp>(op)) {
            auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
            static bool isVariadic[] = {false, true, true};
            return std::make_tuple(idxAndLen.first, idxAndLen.second, isVariadic[index]);
        }
        if (auto concreteOp = llvm::dyn_cast<daphne::ThetaJoinOp>(op)) {
            auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
            static bool isVariadic[] = {false, false, true, true};
            return std::make_tuple(idxAndLen.first, idxAndLen.second, isVariadic[index]);
        }
        if (auto concreteOp = llvm::dyn_cast<daphne::OrderOp>(op)) {
            auto idxAndLen = concreteOp.getODSOperandIndexAndLength(index);
            static bool isVariadic[] = {false, true, true, false};
            return std::make_tuple(idxAndLen.first, idxAndLen.second, isVariadic[index]);
        }
        throw ErrorHandler::compilerError(op, "RewriteToCallKernelOpPass",
                                          "lowering to kernel call not yet supported for this variadic "
                                          "operation: " +
                                              op->getName().getStringRef().str());
    }

    /**
     * @brief The value of type `DaphneContext` to insert as the last
     * argument to all kernel calls.
     */
    Value dctx;

    const DaphneUserConfig &userConfig;
    std::unordered_map<std::string, bool> &usedLibPaths;

  public:
    /**
     * Creates a new KernelReplacement rewrite pattern.
     *
     * @param mctx The MLIR context.
     * @param dctx The DaphneContext to pass to the kernels.
     * @param userConfig The user config.
     * @param benefit
     */
    KernelReplacement(MLIRContext *mctx, Value dctx, const DaphneUserConfig &userConfig,
                      std::unordered_map<std::string, bool> &usedLibPaths, PatternBenefit benefit = 1)
        : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, mctx), dctx(dctx), userConfig(userConfig),
          usedLibPaths(usedLibPaths) {}

    /**
     * @brief Rewrites the given operation to a `CallKernelOp`.
     *
     * This involves looking up a matching kernel from the kernel catalog based
     * on the mnemonic, argument/result types, and backend (e.g., hardware
     * accelerator) of the given operation. Variadic operands are also taken
     * into account.
     *
     * @param op The operation to rewrite.
     * @param rewriter The rewriter.
     * @result Always returns `mlir::success()` unless an exception is thrown.
     */
    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        Location loc = op->getLoc();

        // The argument/result types of the given operation.
        Operation::operand_type_range opArgTys = op->getOperandTypes();
        Operation::result_type_range opResTys = op->getResultTypes();

        // The argument/result types to use for kernel look-up.
        std::vector<mlir::Type> lookupArgTys;
        std::vector<mlir::Type> lookupResTys;
        // Differences between op argument types and look-up argument types:
        // - The look-up argument types summarize n occurrences of a variadic
        //   operand into one variadic pack and one number of occurrences.
        // - The look-up argument types omit most of the properties of the op
        //   argument types, because those would complicate the search for matching kernels.
        // Differences between op result types and look-up result types:
        // - The look-up result types omit most of the properties of the op
        //   result types, because those would complicate the search for matching kernels.

        // The operands to use for the CallKernelOp to be created. These may
        // differ from the operands of the given operation, if it has a variadic
        // operand.
        std::vector<Value> kernelArgs;

        // *****************************************************************************
        // Prepare the kernel look-up and the creation of the CallKernelOp
        // *****************************************************************************
        // Determine the argument/result types for the kernel look-up as well as
        // the arguments of the CallKernelOp to be created. Variadic operands
        // are taken into account.

        // Find out if argument types shall the generalized from matrix/frame to
        // the supertype structure.
        // TODO Don't enumerate all ops, decide based on a trait.
        const bool generalizeInputTypes =
            llvm::isa<daphne::CreateFrameOp>(op) || llvm::isa<daphne::DistributedComputeOp>(op) ||
            llvm::isa<daphne::NumCellsOp>(op) || llvm::isa<daphne::NumColsOp>(op) || llvm::isa<daphne::NumRowsOp>(op) ||
            llvm::isa<daphne::IncRefOp>(op) || llvm::isa<daphne::DecRefOp>(op);

        // Append converted op result types to the look-up result types.
        for (size_t i = 0; i < opResTys.size(); i++)
            lookupResTys.push_back(KernelCatalog::normalizeTypeForKernelLookup(opResTys[i], false));

        // Append converted op argument types to the look-up argument types.
        // Variadic operands, which can have an arbitrary number of occurrences,
        // are treated specially.
        if (
            // TODO Unfortunately, one needs to know the exact N for
            // AtLeastNOperands... There seems to be no simple way to
            // detect if an operation has variadic ODS operands with any N.
            op->hasTrait<OpTrait::VariadicOperands>() || op->hasTrait<OpTrait::AtLeastNOperands<1>::Impl>() ||
            op->hasTrait<OpTrait::AtLeastNOperands<2>::Impl>()) {
            // For operations with variadic ODS operands, we replace all
            // occurrences of a variadic ODS operand by a single operand of
            // type VariadicPack as well as an operand for the number of
            // occurrences. All occurrences of the variadic ODS operand are
            // stored in the VariadicPack.
            // Note that a variadic ODS operand may have zero occurrences.
            // In that case, there is no operand corresponding to the
            // variadic ODS operand.
            const size_t numODSOperands = getNumODSOperands(op);
            for (size_t i = 0; i < numODSOperands; i++) {
                auto odsOpInfo = getODSOperandInfo(op, i);
                const unsigned idx = std::get<0>(odsOpInfo);
                const unsigned len = std::get<1>(odsOpInfo);
                const bool isVariadic = std::get<2>(odsOpInfo);

                // Determine the MLIR type of the current ODS operand.
                Type odsOperandTy;
                if (len > 0) {
                    // If the current ODS operand has occurrences, then
                    // we use the type of the first operand belonging to
                    // the current ODS operand.
                    odsOperandTy = opArgTys[idx];
                } else { // len == 0
                    // If the current ODS operand does not have any occurrences
                    // (e.g., a variadic ODS operand with zero concrete operands
                    // provided), then we cannot derive the type of the
                    // current ODS operand from any given operand. Instead,
                    // we use a default type depending on which ODS operand of
                    // which operation it is.
                    // Note that we cannot simply omit the type, since the
                    // underlying kernel expects an "empty list" (represented
                    // in the DAPHNE compiler by an empty VariadicPack).
                    if (llvm::dyn_cast<daphne::GroupOp>(op) && i == 2)
                        // A GroupOp may have zero aggregation column names.
                        odsOperandTy = daphne::StringType::get(rewriter.getContext());
                    else
                        throw std::runtime_error("RewriteToCallKernelOpPass encountered a variadic "
                                                 "ODS operand with zero occurrences, "
                                                 "but does not know how to handle it: ODS operand " +
                                                 std::to_string(i) + " of operation " +
                                                 op->getName().getStringRef().str());
                }

                lookupArgTys.push_back(KernelCatalog::normalizeTypeForKernelLookup(odsOperandTy, generalizeInputTypes));

                if (isVariadic) {
                    // Variadic operand.
                    lookupArgTys.push_back(rewriter.getIndexType());
                    auto cvpOp = rewriter.create<daphne::CreateVariadicPackOp>(
                        loc, daphne::VariadicPackType::get(rewriter.getContext(), odsOperandTy),
                        rewriter.getI64IntegerAttr(len));
                    for (int64_t k = 0; k < len; k++)
                        rewriter.create<daphne::StoreVariadicPackOp>(loc, cvpOp, op->getOperand(idx + k),
                                                                     rewriter.getI64IntegerAttr(k));
                    kernelArgs.push_back(cvpOp);
                    kernelArgs.push_back(
                        rewriter.create<daphne::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(len)));
                } else
                    // Non-variadic operand.
                    kernelArgs.push_back(op->getOperand(idx));
            }
        } else
            // For operations without variadic operands, we simply append
            // the type of each operand to the vector of types to use for
            // kernel look-up, and pass all operands to the CallKernelOp as-is.
            for (size_t i = 0; i < opArgTys.size(); i++) {
                lookupArgTys.push_back(KernelCatalog::normalizeTypeForKernelLookup(opArgTys[i], generalizeInputTypes));
                kernelArgs.push_back(op->getOperand(i));
            }

        if (auto cdm2m = dyn_cast<daphne::ConvertDenseMatrixToMemRef>(op)) {
            // The kernel for the conversion op is registered with signedness
            // semantics, but the `DaphneTypeConverter` converst to a signless memref. To still match the proper kernel,
            // we normalize and set the type to the `daphne.Matrix` element type.
            auto matTy = llvm::cast<daphne::MatrixType>(cdm2m.getArg().getType());
            auto resMemRef = llvm::cast<MemRefType>(cdm2m.getResult().getType());
            auto memRefSignedness = MemRefType::get(resMemRef.getShape(), matTy.getElementType(), resMemRef.getLayout(),
                                                    resMemRef.getMemorySpace());
            lookupResTys[0] = KernelCatalog::normalizeTypeForKernelLookup(memRefSignedness, false);
        }

        if (auto groupOp = llvm::dyn_cast<daphne::GroupOp>(op)) {
            // GroupOp carries the aggregation functions to apply as an
            // attribute. Since attributes do not automatically become
            // inputs to the kernel call, we need to add them explicitly
            // here.

            ArrayAttr aggFuncs = groupOp.getAggFuncs();
            const size_t numAggFuncs = aggFuncs.size();
            const Type t = rewriter.getIntegerType(32, false);
            auto cvpOp = rewriter.create<daphne::CreateVariadicPackOp>(
                loc, daphne::VariadicPackType::get(rewriter.getContext(), t), rewriter.getI64IntegerAttr(numAggFuncs));
            size_t k = 0;
            for (Attribute aggFunc : aggFuncs.getValue())
                rewriter.create<daphne::StoreVariadicPackOp>(
                    loc, cvpOp,
                    rewriter.create<daphne::ConstantOp>(
                        loc, t,
                        rewriter.getIntegerAttr(
                            t, static_cast<uint32_t>(llvm::dyn_cast<daphne::GroupEnumAttr>(aggFunc).getValue()))),
                    rewriter.getI64IntegerAttr(k++));
            kernelArgs.push_back(cvpOp);
            kernelArgs.push_back(
                rewriter.create<daphne::ConstantOp>(loc, rewriter.getIndexType(), rewriter.getIndexAttr(numAggFuncs)));
        }

        if (auto thetaJoinOp = llvm::dyn_cast<daphne::ThetaJoinOp>(op)) {
            // ThetaJoinOp carries multiple CompareOperation as an
            // attribute. Since attributes do not automatically become
            // inputs to the kernel call, we need to add them explicitly
            // here.

            // get array of CompareOperations
            ArrayAttr compareOperations = thetaJoinOp.getCmp();
            const size_t numCompareOperations = compareOperations.size();
            const Type t = rewriter.getIntegerType(32, false);
            // create Variadic Pack
            auto cvpOp = rewriter.create<daphne::CreateVariadicPackOp>(
                loc, daphne::VariadicPackType::get(rewriter.getContext(), t),
                rewriter.getI64IntegerAttr(numCompareOperations));
            // fill variadic pack
            size_t k = 0;
            for (Attribute compareOperation : compareOperations.getValue())
                rewriter.create<daphne::StoreVariadicPackOp>(
                    loc, cvpOp,
                    rewriter.create<daphne::ConstantOp>(
                        loc, t,
                        rewriter.getIntegerAttr(
                            t, static_cast<uint32_t>(
                                   llvm::dyn_cast<daphne::CompareOperationAttr>(compareOperation).getValue()))),
                    rewriter.getI64IntegerAttr(k++));
            // add created variadic pack and size of this pack as
            // new operands / parameters of the ThetaJoin-Kernel call
            kernelArgs.push_back(cvpOp);
            kernelArgs.push_back(rewriter.create<daphne::ConstantOp>(loc, rewriter.getIndexType(),
                                                                     rewriter.getIndexAttr(numCompareOperations)));
        }

        if (auto distCompOp = llvm::dyn_cast<daphne::DistributedComputeOp>(op)) {
            MLIRContext newContext; // TODO Reuse the existing context.
            OpBuilder tempBuilder(&newContext);
            std::string funcName = "dist";

            auto &bodyBlock = distCompOp.getBody().front();
            auto funcType =
                tempBuilder.getFunctionType(bodyBlock.getArgumentTypes(), bodyBlock.getTerminator()->getOperandTypes());
            auto funcOp = tempBuilder.create<func::FuncOp>(loc, funcName, funcType);

            IRMapping mapper;
            distCompOp.getBody().cloneInto(&funcOp.getRegion(), mapper);

            // write recompile region as string constant
            std::string s;
            llvm::raw_string_ostream stream(s);
            funcOp.print(stream);

            auto strTy = daphne::StringType::get(rewriter.getContext());
            Value rewriteStr = rewriter.create<daphne::ConstantOp>(loc, strTy, rewriter.getStringAttr(stream.str()));
            lookupArgTys.push_back(mlir::daphne::StringType::get(&newContext));
            kernelArgs.push_back(rewriteStr);
        }

        // *****************************************************************************
        // Look up a matching kernel from the kernel catalog.
        // *****************************************************************************

        const KernelCatalog &kc = userConfig.kernelCatalog;
        const std::string opMnemonic = op->getName().stripDialect().data();
        std::vector<KernelInfo> kernelInfos = kc.getKernelInfos(opMnemonic);

        std::string libPath;
        std::string kernelFuncName;
        // TODO Don't hardcode the attribute name, put it in a central place.
        if (op->hasAttr("kernel_hint")) {
            // The operation has a kernel hint. Lower to the hinted kernel if
            // possible.

            // TODO Check if the attribute has the right type.
            kernelFuncName = op->getAttrOfType<mlir::StringAttr>("kernel_hint").getValue().str();
            bool found = false;
            for (size_t i = 0; i < kernelInfos.size() && !found; i++) {
                auto ki = kernelInfos[i];
                if (ki.kernelFuncName == kernelFuncName) {
                    libPath = ki.libPath;
                    found = true;
                }
            }
            if (!found)
                throw ErrorHandler::compilerError(loc, "RewriteToCallKernelOpPass",
                                                  "no kernel found for operation `" + opMnemonic +
                                                      "` with hinted name `" + kernelFuncName + "`");
        } else {
            // The operation does not have a kernel hint. Search for a kernel
            // for this operation and the given result/argument types and
            // backend.

            std::string backend;
            if (op->hasAttr("cuda_device"))
                backend = "CUDA";
            else if (op->hasAttr("fpgaopencl_device"))
                backend = "FPGAOPENCL";
            else
                backend = "CPP";

            auto kernelOpt = kc.findKernel(opMnemonic, lookupArgTys, lookupResTys, backend);
            if (!kernelOpt) {
                throw ErrorHandler::compilerError(
                    loc, "RewriteToCallKernelOpPass",
                    kc.formatNoKernelError(opMnemonic, lookupArgTys, lookupResTys, backend));
            }
            libPath = kernelOpt->libPath;
            kernelFuncName = kernelOpt->kernelFuncName;
        }

        // *****************************************************************************
        // Add kernel id and DAPHNE context as arguments
        // *****************************************************************************
        auto kId = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(KernelDispatchMapping::instance().registerKernel(kernelFuncName, op)));

        // Inject the KernelDispatchMapping id as the second to last input parameter to
        // all kernel calls, unless it's a non-instrumented operation (CreateDaphneContextOp, DestroyDaphneContextOp).
        if (!llvm::isa<daphne::CreateDaphneContextOp, daphne::DestroyDaphneContextOp>(op))
            // NOTE: kId has to be added before CreateDaphneContextOp because
            // there is an assumption that the CTX is the last argument
            // (LowerToLLVMPass.cpp::623,702). This means the kId is expected to
            // be the second to last argument.
            kernelArgs.push_back(kId);
        // Inject the current DaphneContext as the last input parameter to
        // all kernel calls, unless it's a CreateDaphneContextOp.
        if (!llvm::isa<daphne::CreateDaphneContextOp>(op))
            kernelArgs.push_back(dctx);

        // *****************************************************************************
        // Create the CallKernelOp
        // *****************************************************************************

        // Mark the shared library the chosen kernel comes from as used. This
        // means we will link this library into the JIT-compiled program later.
        usedLibPaths.at(libPath) = true;

        // Create a CallKernelOp for the kernel function to call and return
        // success().
        auto kernel = rewriter.create<daphne::CallKernelOp>(loc, kernelFuncName, kernelArgs, opResTys);
        rewriter.replaceOp(op, kernel.getResults());
        return success();
    }
};

class DistributedPipelineKernelReplacement : public OpConversionPattern<daphne::DistributedPipelineOp> {
    Value dctx;
    const DaphneUserConfig &userConfig;
    std::unordered_map<std::string, bool> &usedLibPaths;

  public:
    using OpConversionPattern::OpConversionPattern;
    DistributedPipelineKernelReplacement(MLIRContext *mctx, Value dctx, const DaphneUserConfig &userConfig,
                                         std::unordered_map<std::string, bool> &usedLibPaths,
                                         PatternBenefit benefit = 2)
        : OpConversionPattern(mctx, benefit), dctx(dctx), userConfig(userConfig), usedLibPaths(usedLibPaths) {}

    LogicalResult matchAndRewrite(daphne::DistributedPipelineOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        size_t numOutputs = op.getOutputs().size();
        size_t numInputs = op.getInputs().size();

        std::stringstream callee;
        callee << "_distributedPipeline";         // kernel name
        callee << "__DenseMatrix_double_variadic" // outputs
               << "__size_t"                      // numOutputs
               << "__Structure_variadic"          // inputs
               << "__size_t"                      // numInputs
               << "__int64_t"                     // outRows
               << "__int64_t"                     // outCols
               << "__int64_t"                     // splits
               << "__int64_t"                     // combines
               << "__char";                       // irCode

        MLIRContext *mctx = rewriter.getContext();

        Location loc = op.getLoc();
        Type vptObj = daphne::VariadicPackType::get(mctx, daphne::MatrixType::get(mctx, rewriter.getF64Type()));
        Type vptSize = daphne::VariadicPackType::get(mctx, rewriter.getIntegerType(64, false));
        Type vptInt64 = daphne::VariadicPackType::get(mctx, rewriter.getIntegerType(64, true));

        // Variadic pack for inputs.
        auto cvpInputs =
            rewriter.create<daphne::CreateVariadicPackOp>(loc, vptObj, rewriter.getI64IntegerAttr(numInputs));
        for (size_t i = 0; i < numInputs; i++)
            rewriter.create<daphne::StoreVariadicPackOp>(loc, cvpInputs, op.getInputs()[i],
                                                         rewriter.getI64IntegerAttr(i));
        // Constants for #inputs.
        auto coNumInputs = rewriter.create<daphne::ConstantOp>(loc, numInputs);
        [[maybe_unused]] auto coNumOutputs = rewriter.create<daphne::ConstantOp>(loc, numOutputs);
        // Variadic pack for out_rows.
        auto cvpOutRows =
            rewriter.create<daphne::CreateVariadicPackOp>(loc, vptSize, rewriter.getI64IntegerAttr(numOutputs));
        for (size_t i = 0; i < numOutputs; i++)
            rewriter.create<daphne::StoreVariadicPackOp>(loc, cvpOutRows, op.getOutRows()[i],
                                                         rewriter.getI64IntegerAttr(i));
        // Variadic pack for out_cols.
        auto cvpOutCols =
            rewriter.create<daphne::CreateVariadicPackOp>(loc, vptSize, rewriter.getI64IntegerAttr(numOutputs));
        for (size_t i = 0; i < numOutputs; i++)
            rewriter.create<daphne::StoreVariadicPackOp>(loc, cvpOutCols, op.getOutCols()[i],
                                                         rewriter.getI64IntegerAttr(i));
        // Variadic pack for splits.
        auto cvpSplits =
            rewriter.create<daphne::CreateVariadicPackOp>(loc, vptInt64, rewriter.getI64IntegerAttr(numInputs));
        for (size_t i = 0; i < numInputs; i++)
            rewriter.create<daphne::StoreVariadicPackOp>(
                loc, cvpSplits,
                rewriter.create<daphne::ConstantOp>(
                    loc, static_cast<int64_t>(llvm::dyn_cast<daphne::VectorSplitAttr>(op.getSplits()[i]).getValue())),
                rewriter.getI64IntegerAttr(i));
        // Variadic pack for combines.
        auto cvpCombines =
            rewriter.create<daphne::CreateVariadicPackOp>(loc, vptInt64, rewriter.getI64IntegerAttr(numOutputs));
        for (size_t i = 0; i < numOutputs; i++)
            rewriter.create<daphne::StoreVariadicPackOp>(
                loc, cvpCombines,
                rewriter.create<daphne::ConstantOp>(
                    loc,
                    static_cast<int64_t>(llvm::dyn_cast<daphne::VectorCombineAttr>(op.getCombines()[i]).getValue())),
                rewriter.getI64IntegerAttr(i));

        // Create CallKernelOp.
        std::vector<Value> newOperands = {cvpInputs, coNumInputs, cvpOutRows, cvpOutCols,
                                          cvpSplits, cvpCombines, op.getIr(), dctx};
        auto cko = rewriter.replaceOpWithNewOp<daphne::CallKernelOp>(op.getOperation(), callee.str(), newOperands,
                                                                     op.getOutputs().getTypes());
        cko->setAttr(CompilerUtils::ATTR_HAS_VARIADIC_RESULTS, rewriter.getBoolAttr(true));

        return success();
    }
};

struct RewriteToCallKernelOpPass : public PassWrapper<RewriteToCallKernelOpPass, OperationPass<func::FuncOp>> {
    const DaphneUserConfig &userConfig;
    std::unordered_map<std::string, bool> &usedLibPaths;

    explicit RewriteToCallKernelOpPass(const DaphneUserConfig &cfg, std::unordered_map<std::string, bool> &usedLibPaths)
        : userConfig(cfg), usedLibPaths(usedLibPaths) {}

    void runOnOperation() final;
};
} // namespace

void RewriteToCallKernelOpPass::runOnOperation() {
    func::FuncOp func = getOperation();

    // Check if this function has a DaphneContext. Functions generated by MLIR
    // internal passes (e.g., from codegen pipeline) don't have one and don't
    // need kernel call rewriting - they contain standard MLIR ops that get
    // lowered directly to LLVM.
    mlir::Value dctx = nullptr;
    auto ops = func.getBody().front().getOps<mlir::daphne::CreateDaphneContextOp>();
    for (auto op : ops) {
        dctx = op.getResult();
        break;
    }
    if (!dctx) {
        // No DaphneContext found - this is an MLIR-generated helper function.
        // Skip kernel rewriting for this function.
        return;
    }

    RewritePatternSet patterns(&getContext());

    // Specification of (il)legal dialects/operations. All DaphneIR operations
    // but those explicitly marked as legal will be replaced by CallKernelOp.
    ConversionTarget target(getContext());
    target.addLegalDialect<mlir::affine::AffineDialect, LLVM::LLVMDialect, scf::SCFDialect, memref::MemRefDialect,
                           mlir::linalg::LinalgDialect, mlir::arith::ArithDialect, mlir::BuiltinDialect,
                           mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect>();

    target.addLegalOp<ModuleOp, func::FuncOp, func::CallOp, func::ReturnOp>();
    target.addIllegalDialect<daphne::DaphneDialect>();
    target.addLegalOp<daphne::ConstantOp, daphne::ReturnOp, daphne::CallKernelOp, daphne::CreateVariadicPackOp,
                      daphne::StoreVariadicPackOp, daphne::VectorizedPipelineOp, scf::ForOp, memref::LoadOp,
                      daphne::GenericCallOp, daphne::MapOp>();
    target.addDynamicallyLegalOp<daphne::CastOp>(
        [](daphne::CastOp op) { return op.isTrivialCast() || op.isRemovePropertyCast(); });

    // All runtime kernels (except for the createDaphneContext-kernel) expect a DaphneContext as their last argument. To
    // this end, when rewriting a DaphneIR op to a CallKernelOp, the DaphneContext is appended to the arguments of the
    // CallKernelOp after the original op's arguments. However, some DaphneIR ops are not lowered to CallKernelOp in
    // this pass but later (in LowerToLLVMPass), because they need some more low-level information (e.g., function
    // pointers) which is not available at this stage. Those ops must be given the DaphneContext here, because after
    // this pass, the CreateDaphneContextOps will have been lowered and so an extraction of the DaphneContext from the
    // IR will not be possible/convenient anymore.
    func->walk([&](daphne::VectorizedPipelineOp vpo) { vpo.getCtxMutable().assign(dctx); });
    func->walk([&](daphne::MapOp mo) { mo.getCtxMutable().assign(dctx); });

    // Apply conversion to CallKernelOps.
    patterns.insert<KernelReplacement, DistributedPipelineKernelReplacement>(&getContext(), dctx, userConfig,
                                                                             usedLibPaths);
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createRewriteToCallKernelOpPass(const DaphneUserConfig &cfg,
                                                              std::unordered_map<std::string, bool> &usedLibPaths) {
    return std::make_unique<RewriteToCallKernelOpPass>(cfg, usedLibPaths);
}
