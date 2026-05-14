/*
 *  Copyright 2023 The DAPHNE Consortium
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

#pragma once

#include <compiler/utils/TypePrinting.h>

#include "ir/daphneir/Daphne.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief Stores information on a single kernel.
 */
struct KernelInfo {
    /**
     * @brief The name of the pre-compiled kernel function.
     */
    const std::string kernelFuncName;

    // TODO Add the path to the shared library containing the kernel function.

    /**
     * @brief The kernel's result types.
     */
    const std::vector<mlir::Type> resTypes;

    /**
     * @brief The kernel's argument types.
     */
    const std::vector<mlir::Type> argTypes;

    // TODO Maybe unify this with ALLOCATION_TYPE.
    /**
     * @brief The targeted backend (e.g., hardware accelerator).
     */
    const std::string backend;

    /**
     * @brief The path to the shared library containing the pre-compiled kernel.
     *
     * This path can be absolute or relative to the present working directory.
     */
    const std::string libPath;

    /**
     * @brief The priority of this kernel.
     *
     * If there are multiple applicable kernels, the one with the highest priority will be preferred.
     */
    const int64_t priority;

    KernelInfo(const std::string kernelFuncName, const std::vector<mlir::Type> resTypes,
               const std::vector<mlir::Type> argTypes, const std::string backend, const std::string libPath,
               int64_t priority)
        : kernelFuncName(kernelFuncName), resTypes(resTypes), argTypes(argTypes), backend(backend), libPath(libPath),
          priority(priority) {
        //
    }
};

/**
 * @brief Stores information on kernels registered in the DAPHNE compiler.
 */
class KernelCatalog {
    /**
     * @brief The central data structure mapping DaphneIR operations to
     * registered kernels.
     *
     * The DaphneIR operation is represented by its mnemonic. The kernels are
     * represented by their kernel information.
     */
    std::unordered_map<std::string, std::vector<KernelInfo>> kernelInfosByOp;

    /**
     * @brief Prints the given kernel information.
     *
     * @param opMnemonic The mnemonic of the corresponding DaphneIR operation.
     * @param kernelInfos The kernel information to print.
     * @param os The stream to print to. Defaults to `std::cerr`.
     */
    void dumpKernelInfos(const std::string &opMnemonic, const std::vector<KernelInfo> &kernelInfos,
                         std::ostream &os = std::cerr) const {
        os << "- operation `" << opMnemonic << "` (" << kernelInfos.size() << " kernels)" << std::endl;
        for (KernelInfo ki : kernelInfos)
            os << "  - kernel `" << ki.kernelFuncName << "`: (" << ki.argTypes << ") -> (" << ki.resTypes
               << ") for backend `" << ki.backend << "` (in `" << ki.libPath << "`)" << std::endl;
    }

  public:
    /**
     * @brief Normalizes a type for kernel lookup by removing shape information
     * and optionally generalizing to structure types.
     *
     * This is used to match operation types against registered kernel types,
     * which typically don't include concrete shape information.
     *
     * @param t The type to normalize.
     * @param generalizeToStructure If true, Matrix/Frame/Column/List types are
     *        generalized to StructureType.
     * @return The normalized type suitable for kernel lookup.
     */
    static mlir::Type normalizeTypeForKernelLookup(mlir::Type t, bool generalizeToStructure = false) {
        mlir::MLIRContext *mctx = t.getContext();
        if (generalizeToStructure && llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType,
                                               mlir::daphne::ColumnType, mlir::daphne::ListType>(t))
            return mlir::daphne::StructureType::get(mctx);
        if (auto mt = llvm::dyn_cast<mlir::daphne::MatrixType>(t))
            return mt.withSameElementTypeAndRepr();
        if (llvm::isa<mlir::daphne::FrameType>(t))
            return mlir::daphne::FrameType::get(mctx, {mlir::daphne::UnknownType::get(mctx)});
        if (auto ct = llvm::dyn_cast<mlir::daphne::ColumnType>(t))
            return ct.withSameValueType();
        if (auto lt = llvm::dyn_cast<mlir::daphne::ListType>(t))
            return mlir::daphne::ListType::get(
                mctx, normalizeTypeForKernelLookup(lt.getElementType(), generalizeToStructure));
        if (auto mrt = llvm::dyn_cast<mlir::MemRefType>(t)) {
            // Drop concrete shapes; keep rank and element type.
            int64_t mrtRank = mrt.getRank();
            if (mrtRank == 1) {
                return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, mrt.getElementType());
            } else if (mrtRank == 2) {
                return mlir::MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
                                             mrt.getElementType());
            } else {
                throw std::runtime_error("KernelCatalog: expected MemRef to be of rank 1 or 2 but was given " +
                                         std::to_string(mrtRank));
            }
        }
        return t;
    }

    /**
     * @brief Registers the given kernel information as a kernel for the
     * DaphneIR operation with the given mnemonic.
     *
     * @param opMnemonic The DaphneIR operation's mnemonic.
     * @param kernelInfo The information on the kernel.
     */
    void registerKernel(std::string opMnemonic, KernelInfo kernelInfo) {
        kernelInfosByOp[opMnemonic].push_back(kernelInfo);
    }

    /**
     * @brief Retrieves information on all kernels registered for the given
     * DaphneIR operation.
     *
     * @param opMnemonic The mnemonic of the DaphneIR operation.
     * @return A vector of kernel information, or an empty vector if no kernels
     * are registered for the given operation.
     */
    const std::vector<KernelInfo> getKernelInfos(const std::string &opMnemonic) const {
        auto it = kernelInfosByOp.find(opMnemonic);
        if (it != kernelInfosByOp.end())
            return it->second;
        else
            return {};
    }

    /**
     * @brief Finds the best matching kernel for the given operation, types, and backend.
     *
     * @param opMnemonic The mnemonic of the DaphneIR operation.
     * @param argTypes The argument types to match.
     * @param resTypes The result types to match.
     * @param backend The backend to target (e.g., "CPP", "CUDA").
     * @return An optional containing the matching KernelInfo if found, or std::nullopt if no match.
     */
    std::optional<KernelInfo> findKernel(const std::string &opMnemonic, const std::vector<mlir::Type> &argTypes,
                                         const std::vector<mlir::Type> &resTypes,
                                         const std::string &backend = "CPP") const {
        std::vector<KernelInfo> kernelInfos = getKernelInfos(opMnemonic);

        if (kernelInfos.empty())
            return std::nullopt;

        const size_t numArgs = argTypes.size();
        const size_t numRess = resTypes.size();
        int chosenKernelIdx = -1;
        int64_t chosenKernelPriority = std::numeric_limits<int64_t>::min();

        for (size_t i = 0; i < kernelInfos.size(); i++) {
            const auto &ki = kernelInfos[i];
            if (ki.backend != backend)
                continue;
            if (numArgs != ki.argTypes.size())
                continue;
            if (numRess != ki.resTypes.size())
                continue;

            bool mismatch = false;
            for (size_t j = 0; j < numArgs && !mismatch; j++)
                if (argTypes[j] != ki.argTypes[j])
                    mismatch = true;
            for (size_t j = 0; j < numRess && !mismatch; j++)
                if (resTypes[j] != ki.resTypes[j])
                    mismatch = true;

            if (!mismatch && (ki.priority > chosenKernelPriority || chosenKernelIdx == -1)) {
                chosenKernelIdx = static_cast<int>(i);
                chosenKernelPriority = ki.priority;
            }
        }

        if (chosenKernelIdx == -1)
            return std::nullopt;

        return kernelInfos[chosenKernelIdx];
    }

    /**
     * @brief Formats an error message for when no matching kernel is found.
     *
     * @param opMnemonic The mnemonic of the DaphneIR operation.
     * @param argTypes The argument types that were searched for.
     * @param resTypes The result types that were searched for.
     * @param backend The backend that was targeted.
     * @return A formatted error message string.
     */
    std::string formatNoKernelError(const std::string &opMnemonic, const std::vector<mlir::Type> &argTypes,
                                    const std::vector<mlir::Type> &resTypes, const std::string &backend = "CPP") const {
        std::stringstream s;
        s << "no kernel for operation `" << opMnemonic << "` available for the required input types `(" << argTypes
          << ")` and output types `(" << resTypes << ")` for backend `" << backend
          << "`, registered kernels for this op:" << std::endl;
        dump(opMnemonic, s);
        return s.str();
    }

    /**
     * @brief Returns the mnemonic of the operation for which a kernel with the
     * given name is registered, or throws an exception if there is none.
     *
     * @param kernelFuncName The name of the kernel function to look for.
     * @return The mnemonic of the operation.
     */
    std::string getOpMnemonic(const std::string &kernelFuncName) {
        for (auto it : kernelInfosByOp) {
            std::string opMnemonic = it.first;
            const std::vector<KernelInfo> &kis = it.second;
            for (auto it2 : kis)
                if (it2.kernelFuncName == kernelFuncName)
                    return opMnemonic;
        }
        throw std::runtime_error("no kernel with name `" + kernelFuncName + "` registered in the kernel catalog");
    }

    /**
     * @brief Prints high-level statistics on the kernel catalog.
     *
     * @param os The stream to print to. Defaults to `std::cerr`.
     */
    void stats(std::ostream &os = std::cerr) const {
        const size_t numOps = kernelInfosByOp.size();
        size_t numKernels = 0;
        for (auto it = kernelInfosByOp.begin(); it != kernelInfosByOp.end(); it++)
            numKernels += it->second.size();
        os << "KernelCatalog (" << numOps << " ops, " << numKernels << " kernels)" << std::endl;
    }

    /**
     * @brief Prints this kernel catalog.
     *
     * @param opMnemonic If an empty string, print registered kernels for all
     * DaphneIR operations; otherwise, consider only the specified DaphneIR
     * operation.
     * @param os The stream to print to. Defaults to `std::cerr`.
     */
    void dump(std::string opMnemonic = "", std::ostream &os = std::cerr) const {
        stats(os);
        if (opMnemonic.empty())
            // Print info on all ops.
            for (auto it = kernelInfosByOp.begin(); it != kernelInfosByOp.end(); it++)
                dumpKernelInfos(it->first, it->second, os);
        else
            // Print info on specified op only.
            dumpKernelInfos(opMnemonic, getKernelInfos(opMnemonic), os);
    }

    /**
     * @brief Returns all distinct kernel libraries in the form of a mapping
     * from the library path to the constant `false`.
     *
     * @return A mapping from each distict kernel library path to the constant
     * `false`.
     */
    std::unordered_map<std::string, bool> getLibPaths() const {
        std::unordered_map<std::string, bool> res;

        for (auto it : kernelInfosByOp) {
            const std::vector<KernelInfo> &kis = it.second;
            for (auto it2 : kis)
                res[it2.libPath] = false;
        }

        return res;
    }
};
