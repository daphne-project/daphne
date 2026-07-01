/*
 * Copyright 2023 The DAPHNE Consortium
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

#include <compiler/catalog/KernelCatalog.h>
#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <parser/catalog/KernelCatalogParser.h>

#include <nlohmannjson/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

KernelCatalogParser::KernelCatalogParser(mlir::MLIRContext *mctx) {
    // Initialize the mapping from C++ type name strings to MLIR types for
    // parsing.

    mlir::OpBuilder builder(mctx);

    // Scalars and matrices.
    std::vector<mlir::Type> scalarTypes = {builder.getF64Type(),
                                           builder.getF32Type(),
                                           builder.getIntegerType(64, true),
                                           builder.getIntegerType(32, true),
                                           builder.getIntegerType(8, true),
                                           builder.getIntegerType(64, false),
                                           builder.getIntegerType(32, false),
                                           builder.getIntegerType(8, false),
                                           builder.getI1Type(),
                                           builder.getIndexType(),
                                           mlir::daphne::StringType::get(mctx),
                                           mlir::daphne::FixedStr16Type::get(mctx)};
    for (mlir::Type st : scalarTypes) {
        // Scalar type.
        typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(st), st);

        // Matrix type for DenseMatrix.
        // TODO This should have
        // withRepresentation(mlir::daphne::MatrixRepresentation::Dense).
        mlir::Type mtDense = mlir::daphne::MatrixType::get(mctx, st);
        typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(mtDense), mtDense);
        // Matrix type for CSRMatrix.
        mlir::Type mtCSR =
            mlir::daphne::MatrixType::get(mctx, st).withRepresentation(mlir::daphne::MatrixRepresentation::Sparse);
        typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(mtCSR), mtCSR);

        // List type for list of DenseMatrix.
        mlir::Type ltDense = mlir::daphne::ListType::get(mctx, mtDense);
        typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(ltDense), ltDense);
        // List type for list of CSRMatrix.
        mlir::Type ltCSR = mlir::daphne::ListType::get(mctx, mtCSR);
        typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(ltCSR), ltCSR);

        // Column type.
        mlir::Type ct = mlir::daphne::ColumnType::get(mctx, st);
        typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(ct), ct);

        // MemRef type.
        if (!llvm::isa<mlir::daphne::StringType, mlir::daphne::FixedStr16Type>(st)) {
            // DAPHNE's string-like types are not supported as the element type
            // of a MemRef. The shape of the MemRef is explicitly set to
            // `ShapedType::kDynamic` for each dimension as the kernel is shape
            // agnostic and handles shapes dynamically.
            mlir::Type mrt1d = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, st);
            mlir::Type mrt2d = mlir::MemRefType::get({mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, st);
            typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(mrt1d), mrt1d);
            typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(mrt2d), mrt2d);
        }
    }

    // Structure, Frame, DaphneContext, MemRef.
    std::vector<mlir::Type> otherTypes = {
        mlir::daphne::StructureType::get(mctx),
        mlir::daphne::FrameType::get(mctx, {mlir::daphne::UnknownType::get(mctx)}),
        mlir::daphne::DaphneContextType::get(mctx),
    };
    for (mlir::Type t : otherTypes) {
        typeMap.emplace(CompilerUtils::mlirTypeToCppTypeName(t), t);
    }
}

void KernelCatalogParser::mapTypes(const std::vector<std::string> &in, std::vector<mlir::Type> &out,
                                   const std::string &word, const std::string &kernelFuncName,
                                   const std::string &opMnemonic, const std::string &backend) const {
    for (size_t i = 0; i < in.size(); i++) {
        const std::string name = in[i];
        auto it = typeMap.find(name);
        if (it != typeMap.end())
            out.push_back(it->second);
        else {
            std::stringstream s;
            s << "KernelCatalogParser: error while parsing " + word + " types of kernel `" << kernelFuncName
              << "` for operation `" << opMnemonic << "` (backend `" << backend << "`): unknown type for " << word
              << " #" << i << ": `" << name << '`';
            throw std::runtime_error(s.str());
        }
    }
}

void KernelCatalogParser::parseKernelCatalog(const std::string &filePath, KernelCatalog &kc, int64_t priority) const {
    std::filesystem::path dirPath = std::filesystem::path(filePath).parent_path();
    try {
        std::ifstream kernelsConfigFile(filePath);
        if (!kernelsConfigFile.good())
            throw std::runtime_error("could not open file for reading");
        nlohmann::json kernelsConfigData = nlohmann::json::parse(kernelsConfigFile);
        for (auto kernelData : kernelsConfigData) {
            const std::string opMnemonic = kernelData["opMnemonic"].get<std::string>();
            // TODO Remove this workaround.
            // Skip these two problematic operations, which return multiple
            // results in the wrong way.
            if (opMnemonic == "Avg_Forward" || opMnemonic == "Max_Forward")
                continue;
            const std::string kernelFuncName = kernelData["kernelFuncName"].get<std::string>();
            const std::string backend = kernelData["backend"].get<std::string>();
            const std::string libPath = dirPath / kernelData["libPath"].get<std::string>();
            std::vector<mlir::Type> resTypes;
            mapTypes(kernelData["resTypes"], resTypes, "result", kernelFuncName, opMnemonic, backend);
            std::vector<mlir::Type> argTypes;
            mapTypes(kernelData["argTypes"], argTypes, "argument", kernelFuncName, opMnemonic, backend);
            kc.registerKernel(opMnemonic, KernelInfo(kernelFuncName, resTypes, argTypes, backend, libPath, priority));
        }
    } catch (std::exception &e) {
        throw std::runtime_error("error while parsing kernel catalog file `" + filePath + "`: " + e.what());
    }
}
