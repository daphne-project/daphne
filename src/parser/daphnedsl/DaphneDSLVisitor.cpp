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

#include <compiler/inference/TypeInferenceUtils.h>
#include <compiler/utils/CompilerUtils.h>
#include <compiler/utils/TypePrinting.h>
#include <ir/daphneir/Daphne.h>
#include <parser/CancelingErrorListener.h>
#include <parser/ScopedSymbolTable.h>
#include <parser/daphnedsl/DaphneDSLParser.h>
#include <parser/daphnedsl/DaphneDSLVisitor.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <util/ErrorHandler.h>

#include "DaphneDSLGrammarLexer.h"
#include "DaphneDSLGrammarParser.h"
#include "antlr4-runtime.h"

#include <mlir/Dialect/SCF/IR/SCF.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cstdint>
#include <cstdlib>
#include <filesystem>

// ****************************************************************************
// Utilities
// ****************************************************************************

mlir::Value DaphneDSLVisitor::renameIf(mlir::Value v) {
    if (symbolTable.has(v))
        return static_cast<mlir::Value>(builder.create<mlir::daphne::RenameOp>(v.getLoc(), v.getType(), v));
    else
        return v;
}

void DaphneDSLVisitor::handleAssignmentPart(mlir::Location loc, const std::string &var,
                                            DaphneDSLGrammarParser::IndexingContext *idxCtx,
                                            ScopedSymbolTable &symbolTable, mlir::Value val) {
    if (symbolTable.has(var) && symbolTable.get(var).isReadOnly)
        throw ErrorHandler::compilerError(loc, "DSLVisitor (handleAssignmentPart)",
                                          "trying to assign read-only variable " + var);

    if (idxCtx) { // left indexing `var[idxCtx] = val;`
        if (!symbolTable.has(var))
            throw ErrorHandler::compilerError(loc, "DSLVisitor (handleAssignmentPart)",
                                              "cannot use left indexing on variable " + var +
                                                  " before a value has been assigned to it");
        mlir::Value obj = symbolTable.get(var).value;

        auto indexing = visit(idxCtx).as<std::pair<std::pair<bool, antlrcpp::Any>, std::pair<bool, antlrcpp::Any>>>();
        auto rows = indexing.first;
        auto cols = indexing.second;

        // TODO Use location of rows/cols in utils.getLoc(...) for better
        // error messages.
        if (rows.first && cols.first) {
            // TODO Use a combined InsertOp (row+col) (see #238).
            mlir::Value rowSeg =
                applyRightIndexing<mlir::daphne::ExtractRowOp, mlir::daphne::SliceRowOp, mlir::daphne::NumRowsOp>(
                    utils.getLoc(idxCtx->start), obj, rows.second, false);
            rowSeg = applyLeftIndexing<mlir::daphne::InsertColOp, mlir::daphne::NumColsOp>(
                utils.getLoc(idxCtx->start), rowSeg, val, cols.second,
                llvm::isa<mlir::daphne::FrameType>(obj.getType()));
            obj = applyLeftIndexing<mlir::daphne::InsertRowOp, mlir::daphne::NumRowsOp>(
                utils.getLoc(idxCtx->start), obj, rowSeg, rows.second, false);
        } else if (rows.first) // rows specified
            obj = applyLeftIndexing<mlir::daphne::InsertRowOp, mlir::daphne::NumRowsOp>(utils.getLoc(idxCtx->start),
                                                                                        obj, val, rows.second, false);
        else if (cols.first) // cols specified
            obj = applyLeftIndexing<mlir::daphne::InsertColOp, mlir::daphne::NumColsOp>(
                utils.getLoc(idxCtx->start), obj, val, cols.second, llvm::isa<mlir::daphne::FrameType>(obj.getType()));
        else
            // no left indexing `var[, ] = val;`
            obj = renameIf(val);

        symbolTable.put(var, ScopedSymbolTable::SymbolInfo(obj, false));
    } else // no left indexing `var = val;`
        symbolTable.put(var, ScopedSymbolTable::SymbolInfo(renameIf(val), false));
}

template <class ExtractAxOp, class SliceAxOp, class NumAxOp>
mlir::Value DaphneDSLVisitor::applyRightIndexing(mlir::Location loc, mlir::Value arg, antlrcpp::Any ax,
                                                 bool allowLabel) {
    if (ax.is<mlir::Value>()) { // indexing with a single SSA value (no ':')
        mlir::Value axVal = ax.as<mlir::Value>();
        if (CompilerUtils::hasObjType(axVal)) // data object
            return CompilerUtils::retValWithInferredType(
                builder.create<ExtractAxOp>(loc, utils.unknownType, arg, axVal));
        else if (llvm::isa<mlir::daphne::StringType>(axVal.getType())) { // string
            if (allowLabel)
                return CompilerUtils::retValWithInferredType(
                    builder.create<ExtractAxOp>(loc, utils.unknownType, arg, axVal));
            else
                throw ErrorHandler::compilerError(loc, "DSLVisitor (applyRightIndexing)",
                                                  "cannot use right indexing with label in this case");
        } else // scalar
            return CompilerUtils::retValWithInferredType(builder.create<SliceAxOp>(
                loc, utils.unknownType, arg, utils.castSI64If(axVal),
                utils.castSI64If(builder.create<mlir::daphne::EwAddOp>(
                    loc, builder.getIntegerType(64, false), utils.castSI64If(axVal),
                    builder.create<mlir::daphne::ConstantOp>(loc, static_cast<int64_t>(1))))));
    } else if (ax.is<std::pair<mlir::Value, mlir::Value>>()) { // indexing with
                                                               // a range (':')
        auto axPair = ax.as<std::pair<mlir::Value, mlir::Value>>();
        auto axLowerIncl = axPair.first;
        auto axUpperExcl = axPair.second;

        // Use defaults if lower or upper bound not specified.
        if (axLowerIncl == nullptr)
            axLowerIncl = builder.create<mlir::daphne::ConstantOp>(loc, static_cast<int64_t>(0));
        if (axUpperExcl == nullptr)
            axUpperExcl = builder.create<NumAxOp>(loc, utils.sizeType, arg);

        return CompilerUtils::retValWithInferredType(builder.create<SliceAxOp>(
            loc, utils.unknownType, arg, utils.castSI64If(axLowerIncl), utils.castSI64If(axUpperExcl)));
    } else
        throw ErrorHandler::compilerError(loc, "DSLVisitor (applyRightIndexing)",
                                          "unsupported type for right indexing");
}

template <class InsertAxOp, class NumAxOp>
mlir::Value DaphneDSLVisitor::applyLeftIndexing(mlir::Location loc, mlir::Value arg, mlir::Value ins, antlrcpp::Any ax,
                                                bool allowLabel) {
    mlir::Type argType = arg.getType();

    if (ax.is<mlir::Value>()) { // indexing with a single SSA value (no ':')
        mlir::Value axVal = ax.as<mlir::Value>();
        if (CompilerUtils::hasObjType(axVal)) // data object
            throw ErrorHandler::compilerError(loc, "DSLVisitor (applyLeftIndexing)",
                                              "left indexing with positions as a data object is not "
                                              "supported (yet)");
        else if (llvm::isa<mlir::daphne::StringType>(axVal.getType())) { // string
            if (allowLabel)
                // TODO Support this (#239).
                throw ErrorHandler::compilerError(loc, "DSLVisitor (applyLeftIndexing)",
                                                  "left indexing by label is not supported yet");
            else
                throw ErrorHandler::compilerError(loc, "DSLVisitor (applyLeftIndexing)",
                                                  "cannot use left indexing with label in this case");
        } else // scalar
            return static_cast<mlir::Value>(builder.create<InsertAxOp>(
                loc, argType, arg, ins, utils.castSI64If(axVal),
                utils.castSI64If(builder.create<mlir::daphne::EwAddOp>(
                    loc, builder.getIntegerType(64, false), utils.castSI64If(axVal),
                    builder.create<mlir::daphne::ConstantOp>(loc, static_cast<int64_t>(1))))));
    } else if (ax.is<std::pair<mlir::Value, mlir::Value>>()) { // indexing with
                                                               // a range (':')
        auto axPair = ax.as<std::pair<mlir::Value, mlir::Value>>();
        auto axLowerIncl = axPair.first;
        auto axUpperExcl = axPair.second;

        // Use defaults if lower or upper bound not specified.
        if (axLowerIncl == nullptr)
            axLowerIncl = builder.create<mlir::daphne::ConstantOp>(loc, static_cast<int64_t>(0));
        if (axUpperExcl == nullptr)
            axUpperExcl = builder.create<NumAxOp>(loc, utils.sizeType, arg);

        return static_cast<mlir::Value>(builder.create<InsertAxOp>(
            loc, argType, arg, ins, utils.castSI64If(axLowerIncl), utils.castSI64If(axUpperExcl)));
    } else
        throw ErrorHandler::compilerError(loc, "DSLVisitor (applyLeftIndexing)", "unsupported type for left indexing");
}

std::vector<DaphneDSLVisitor::CallArgInfo>
DaphneDSLVisitor::collectCallArgs(DaphneDSLGrammarParser::CallExprContext *ctx, bool evaluate) {
    std::vector<CallArgInfo> callArgs;
    if (!ctx->callArgs())
        return callArgs;

    for (auto *callArgCtx : ctx->callArgs()->callArg()) {
        CallArgInfo info;
        info.isNamed = callArgCtx->paramName != nullptr;
        if (info.isNamed)
            info.name = callArgCtx->paramName->getText();
        info.ctx = callArgCtx;
        if (evaluate)
            info.value = valueOrErrorOnVisit(callArgCtx->expr());
        else
            info.value = mlir::Value();
        callArgs.push_back(info);
    }

    return callArgs;
}

bool DaphneDSLVisitor::hasNamedArguments(const std::vector<CallArgInfo> &callArgs) const {
    for (const auto &arg : callArgs)
        if (arg.isNamed)
            return true;
    return false;
}

std::optional<std::vector<std::string>> DaphneDSLVisitor::getUdfParamNames(const std::string &functionName,
                                                                           size_t numCallArgs) {
    auto range = functionsSymbolMap.equal_range(functionName);
    std::optional<std::vector<std::string>> paramNames;
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second.getFunctionType().getNumInputs() != numCallArgs)
            continue;
        const auto symName = it->second.getSymName().str();
        auto nameIt = funcParamNames.find(symName);
        if (nameIt == funcParamNames.end())
            continue;
        if (!paramNames)
            paramNames = nameIt->second;
        else if (paramNames != nameIt->second)
            throw ErrorHandler::compilerError(it->second.getLoc(), "DSLVisitor",
                                              "inconsistent parameter names for overloaded function `" + functionName +
                                                  "`");
    }
    return paramNames;
}

void DaphneDSLVisitor::storeFunctionParamNames(mlir::func::FuncOp funcOp, const std::vector<std::string> &paramNames) {
    funcParamNames[funcOp.getSymName().str()] = paramNames;
}

// Maps positional/keyword call arguments to parameter order, enforcing:
// - no mixing positional after keyword
// - keywords must match known parameters
// - no duplicates
// - required leading parameters must be present
// - no gaps before provided optional args
std::vector<mlir::Value> DaphneDSLVisitor::orderCallArguments(const std::string &funcName, mlir::Location callLoc,
                                                              const std::vector<std::string> &paramNames,
                                                              size_t numRequired,
                                                              const std::vector<CallArgInfo> &callArgs,
                                                              std::vector<const CallArgInfo *> *sourceOrder) {
    if (sourceOrder)
        sourceOrder->clear();

    if (paramNames.empty()) {
        for (const auto &arg : callArgs)
            if (arg.isNamed)
                throw ErrorHandler::compilerError(utils.getLoc(arg.ctx->start), "DSLVisitor",
                                                  "function `" + funcName + "` does not support named arguments");

        std::vector<mlir::Value> ordered;
        ordered.reserve(callArgs.size());
        for (const auto &arg : callArgs) {
            ordered.push_back(arg.value);
            if (sourceOrder)
                sourceOrder->push_back(&arg);
        }
        return ordered;
    }

    if (callArgs.size() > paramNames.size())
        throw ErrorHandler::compilerError(callLoc, "DSLVisitor",
                                          "function `" + funcName + "` expects at most " +
                                              std::to_string(paramNames.size()) + " arguments");

    numRequired = std::min(numRequired, paramNames.size());

    std::vector<std::optional<mlir::Value>> slots(paramNames.size());
    std::vector<const CallArgInfo *> sources(paramNames.size(), nullptr);
    size_t positionalIndex = 0;
    bool seenNamed = false;
    for (const auto &arg : callArgs) {
        if (!arg.isNamed) {
            if (seenNamed)
                throw ErrorHandler::compilerError(utils.getLoc(arg.ctx->start), "DSLVisitor",
                                                  "positional argument follows keyword argument in call to `" +
                                                      funcName + "`");
            if (positionalIndex >= paramNames.size())
                throw ErrorHandler::compilerError(utils.getLoc(arg.ctx->start), "DSLVisitor",
                                                  "too many positional arguments for function `" + funcName + "`");
            slots[positionalIndex] = arg.value;
            sources[positionalIndex] = &arg;
            ++positionalIndex;
        } else {
            seenNamed = true;
            size_t idx = paramNames.size();
            for (size_t i = 0; i < paramNames.size(); i++)
                if (paramNames[i] == arg.name) {
                    idx = i;
                    break;
                }
            if (idx == paramNames.size())
                throw ErrorHandler::compilerError(utils.getLoc(arg.ctx->start), "DSLVisitor",
                                                  "unknown keyword argument `" + arg.name + "` for function `" +
                                                      funcName + "`");
            if (slots[idx].has_value())
                throw ErrorHandler::compilerError(utils.getLoc(arg.ctx->start), "DSLVisitor",
                                                  "multiple values for argument `" + arg.name + "` in function `" +
                                                      funcName + "`");
            slots[idx] = arg.value;
            sources[idx] = &arg;
        }
    }

    size_t lastProvided = 0;
    bool hasProvided = false;
    for (size_t i = 0; i < slots.size(); i++)
        if (slots[i].has_value()) {
            lastProvided = i;
            hasProvided = true;
        }

    if (!hasProvided)
        return {};

    std::vector<mlir::Value> ordered;
    if (sourceOrder)
        sourceOrder->clear();
    for (size_t i = 0; i <= lastProvided; i++) {
        if (slots[i].has_value()) {
            ordered.push_back(*slots[i]);
            if (sourceOrder)
                sourceOrder->push_back(sources[i]);
        } else if (i < numRequired)
            throw ErrorHandler::compilerError(callLoc, "DSLVisitor",
                                              "missing value for argument `" + paramNames[i] + "` in function `" +
                                                  funcName + "`");
        else
            throw ErrorHandler::compilerError(callLoc, "DSLVisitor",
                                              "cannot skip optional argument `" + paramNames[i] +
                                                  "` before providing later arguments in function `" + funcName + "`");
    }

    return ordered;
}

// ****************************************************************************
// Visitor functions
// ****************************************************************************

antlrcpp::Any DaphneDSLVisitor::visitScript(DaphneDSLGrammarParser::ScriptContext *ctx) { return visitChildren(ctx); }

antlrcpp::Any DaphneDSLVisitor::visitStatement(DaphneDSLGrammarParser::StatementContext *ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitBlockStatement(DaphneDSLGrammarParser::BlockStatementContext *ctx) {
    symbolTable.pushScope();
    antlrcpp::Any res = visitChildren(ctx);
    symbolTable.put(symbolTable.popScope());
    return res;
}

antlrcpp::Any DaphneDSLVisitor::visitImportStatement(DaphneDSLGrammarParser::ImportStatementContext *ctx) {
    auto loc = utils.getLoc(ctx->start);
    if (symbolTable.getNumScopes() != 1)
        throw ErrorHandler::compilerError(loc, "DSLVisitor (ImportStatement)",
                                          "Imports can only be done in the main scope");

    const char prefixDelim = '.';
    std::string prefix;
    std::vector<std::string> importPaths;
    std::string path = ctx->filePath->getText();
    // Remove quotes
    path = path.substr(1, path.size() - 2);

    std::filesystem::path importerDirPath =
        std::filesystem::absolute(std::filesystem::path(scriptPaths.top())).parent_path();
    std::filesystem::path importingPath = path;

    // Determine the prefix from alias/filename
    if (ctx->alias) {
        prefix = ctx->alias->getText();
        prefix = prefix.substr(1, prefix.size() - 2);
    } else
        prefix = importingPath.stem().string();

    prefix += prefixDelim;

    // Absolute path can be used as is, we have to handle relative paths and
    // config paths
    if (importingPath.is_relative()) {
        std::filesystem::path absolutePath = importerDirPath / importingPath;
        if (std::filesystem::exists(absolutePath))
            absolutePath = std::filesystem::canonical(absolutePath);

        // Check directories in UserConfig (if provided)
        if (!userConf.daphnedsl_import_paths.empty()) {
            const auto &configPaths = userConf.daphnedsl_import_paths;
            // User specified _default_ paths.
            if (importingPath.has_extension() && (configPaths.find("default_dirs") != configPaths.end())) {
                for (std::filesystem::path defaultPath : configPaths.at("default_dirs")) {
                    std::filesystem::path libFile = defaultPath / path;
                    if (std::filesystem::exists(libFile)) {
                        if (std::filesystem::exists(absolutePath) &&
                            std::filesystem::canonical(libFile) != absolutePath)
                            throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                                              std::string("Ambiguous import: ")
                                                                  .append(importingPath)
                                                                  .append(", found another file with the same "
                                                                          "name in default paths of UserConfig: ")
                                                                  .append(libFile));
                        absolutePath = libFile;
                    }
                }
            }

            // User specified "libraries" -> import all files
            if (!importingPath.has_extension() && (configPaths.find(path) != configPaths.end()))
                for (std::filesystem::path const &dir_entry :
                     std::filesystem::directory_iterator{configPaths.at(path)[0]})
                    importPaths.push_back(dir_entry.string());
        }
        path = absolutePath.string();
    }

    if (importPaths.empty())
        importPaths.push_back(path);

    if (std::filesystem::absolute(scriptPaths.top()).string() == path)
        throw ErrorHandler::compilerError(
            loc, "DSLVisitor", std::string("You cannot import the file you are currently in: ").append(path));

    for (const auto &somePath : importPaths) {
        for (const auto &imported : importedFiles)
            if (std::filesystem::equivalent(somePath, imported))
                throw ErrorHandler::compilerError(
                    loc, "DSLVisitor", std::string("You cannot import the same file twice: ").append(somePath));

        importedFiles.push_back(somePath);
    }

    antlrcpp::Any res;
    for (const auto &importPath : importPaths) {
        if (!std::filesystem::exists(importPath))
            throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                              std::string("The import path doesn't exist: ").append(importPath));

        std::string finalPrefix = prefix;
        auto origScope = symbolTable.extractScope();

        // If we import a library, we insert a filename (e.g.,
        // "algorithms/kmeans.daphne" -> algorithms.kmeans.km)
        if (!importingPath.has_extension())
            finalPrefix += std::filesystem::path(importPath).stem().string() + prefixDelim;
        else {
            // If the prefix is already occupied (and is not part of some other
            // prefix), we append a parent directory name
            for (const auto &symbol : origScope)
                if (symbol.first.find(finalPrefix) == 0 &&
                    std::count(symbol.first.begin(), symbol.first.end(), '.') == 1) {
                    // Throw error when we want to use an explicit alias that
                    // results in a prefix clash
                    if (ctx->alias) {
                        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                                          std::string("Alias ")
                                                              .append(ctx->alias->getText())
                                                              .append(" results in a name clash with another "
                                                                      "prefix"));
                    }
                    finalPrefix.insert(0, importingPath.parent_path().filename().string() + prefixDelim);
                    break;
                }
        }

        CancelingErrorListener errorListener;
        std::ifstream ifs(importPath, std::ios::in);
        antlr4::ANTLRInputStream input(ifs);
        input.name = importPath;

        DaphneDSLGrammarLexer lexer(&input);
        lexer.removeErrorListeners();
        lexer.addErrorListener(&errorListener);
        antlr4::CommonTokenStream tokens(&lexer);

        DaphneDSLGrammarParser parser(&tokens);
        parser.removeErrorListeners();
        parser.addErrorListener(&errorListener);
        DaphneDSLGrammarParser::ScriptContext *importCtx = parser.script();

        std::multimap<std::string, mlir::func::FuncOp> origFuncMap = functionsSymbolMap;
        functionsSymbolMap.clear();

        std::vector<std::string> origImportedFiles = importedFiles;
        importedFiles.clear();

        symbolTable.pushScope();
        scriptPaths.push(path);
        res = visitScript(importCtx);
        scriptPaths.pop();

        ScopedSymbolTable::SymbolTable symbTable = symbolTable.extractScope();

        // If the current import file also imported something, we discard it
        for (const auto &symbol : symbTable)
            if (symbol.first.find('.') == std::string::npos)
                origScope[finalPrefix + symbol.first] = symbol.second;

        symbolTable.put(origScope);

        importedFiles = origImportedFiles;

        for (std::pair<std::string, mlir::func::FuncOp> funcSymbol : functionsSymbolMap)
            if (funcSymbol.first.find('.') == std::string::npos)
                origFuncMap.insert({finalPrefix + funcSymbol.first, funcSymbol.second});
        functionsSymbolMap.clear();
        functionsSymbolMap = origFuncMap;
    }
    return res;
}

antlrcpp::Any DaphneDSLVisitor::visitExprStatement(DaphneDSLGrammarParser::ExprStatementContext *ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitAssignStatement(DaphneDSLGrammarParser::AssignStatementContext *ctx) {
    const size_t numVars = ctx->IDENTIFIER().size();
    antlrcpp::Any rhsAny = visit(ctx->expr());
    bool rhsIsRR = rhsAny.is<mlir::ResultRange>();
    auto loc = utils.getLoc(ctx->start);
    if (numVars == 1) {
        // A single variable on the left-hand side.
        if (rhsIsRR)
            throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                              "trying to assign multiple results to a single variable");
        handleAssignmentPart(loc, ctx->IDENTIFIER(0)->getText(), ctx->indexing(0), symbolTable,
                             utils.valueOrError(utils.getLoc(ctx->expr()->start), rhsAny));
        return nullptr;
    } else if (numVars > 1) {
        // Multiple variables on the left-hand side; the expression must be an
        // operation returning multiple outputs.
        if (rhsIsRR) {
            auto rhsAsRR = rhsAny.as<mlir::ResultRange>();
            if (rhsAsRR.size() == numVars) {
                for (size_t i = 0; i < numVars; i++)
                    handleAssignmentPart(loc, ctx->IDENTIFIER(i)->getText(), ctx->indexing(i), symbolTable, rhsAsRR[i]);
                return nullptr;
            }
        }
        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                          "right-hand side expression of assignment to multiple "
                                          "variables must return multiple values, one for each "
                                          "variable on the left-hand side");
    }
    throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                      "the DaphneDSL grammar should prevent zero variables "
                                      "on the left-hand side of an assignment");
    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitIfStatement(DaphneDSLGrammarParser::IfStatementContext *ctx) {
    mlir::Value cond = utils.castBoolIf(valueOrErrorOnVisit(ctx->cond));

    mlir::Location loc = utils.getLoc(ctx->start);

    // Save the current state of the builder.
    mlir::OpBuilder oldBuilder = builder;

    // Generate the operations for the then-block.
    mlir::Block thenBlock;
    builder.setInsertionPointToEnd(&thenBlock);
    symbolTable.pushScope();
    visit(ctx->thenStmt);
    ScopedSymbolTable::SymbolTable owThen = symbolTable.popScope();

    // Generate the operations for the else-block, if it is present. Otherwise,
    // leave it empty; we might need to insert a yield-operation.
    mlir::Block elseBlock;
    ScopedSymbolTable::SymbolTable owElse;
    if (ctx->elseStmt) {
        builder.setInsertionPointToEnd(&elseBlock);
        symbolTable.pushScope();
        visit(ctx->elseStmt);
        owElse = symbolTable.popScope();
    }

    // Determine the result type(s) of the if-operation as well as the operands
    // to the yield-operation of both branches.
    std::set<std::string> owUnion = ScopedSymbolTable::mergeSymbols(owThen, owElse);
    std::vector<mlir::Value> resultsThen;
    std::vector<mlir::Value> resultsElse;
    for (auto it = owUnion.begin(); it != owUnion.end(); it++) {
        mlir::Value valThen = symbolTable.get(*it, owThen).value;
        mlir::Value valElse = symbolTable.get(*it, owElse).value;
        mlir::Type tyThen = valThen.getType();
        mlir::Type tyElse = valElse.getType();
        // TODO These checks should happen after type inference.
        if (!CompilerUtils::equalUnknownAware(tyThen, tyElse)) {
            // TODO We could try to cast the types.
            // TODO Use DaphneDSL types (not MLIR types) in error message.
            // TODO Adapt to the case of no else-branch in DaphneDSL (when there
            // is no else in DaphneDSL, "else" should not be mentioned in the
            // error message).
            std::stringstream s;
            s << "type of variable `" << symbolTable.getSymbol(valThen, owThen)
              << "` after if-statement is ambiguous, could be either " << tyThen << " (then-branch) or " << tyElse
              << " (else-branch)";
            throw ErrorHandler::compilerError(loc, "DSLVisitor", s.str());
        }
        resultsThen.push_back(valThen);
        resultsElse.push_back(valElse);
    }

    // Create yield-operations in both branches, possibly with empty results.
    builder.setInsertionPointToEnd(&thenBlock);
    builder.create<mlir::scf::YieldOp>(loc, resultsThen);
    builder.setInsertionPointToEnd(&elseBlock);
    builder.create<mlir::scf::YieldOp>(loc, resultsElse);

    // Restore the old state of the builder.
    builder = oldBuilder;

    // Helper functions to move the operations in the two blocks created above
    // into the actual branches of the if-operation.
    auto insertThenBlockDo = [&](mlir::OpBuilder &nested, mlir::Location loc) {
        nested.getBlock()->getOperations().splice(nested.getBlock()->end(), thenBlock.getOperations());
    };
    auto insertElseBlockDo = [&](mlir::OpBuilder &nested, mlir::Location loc) {
        nested.getBlock()->getOperations().splice(nested.getBlock()->end(), elseBlock.getOperations());
    };
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> insertElseBlockNo = nullptr;

    // Create the actual if-operation. Generate the else-block only if it was
    // explicitly given in the DSL script, or when it is needed to yield values.
    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc, cond, insertThenBlockDo, (ctx->elseStmt || !owUnion.empty()) ? insertElseBlockDo : insertElseBlockNo);

    // Rewire the results of the if-operation to their variable names.
    size_t i = 0;
    for (auto it = owUnion.begin(); it != owUnion.end(); it++)
        symbolTable.put(*it, ScopedSymbolTable::SymbolInfo(ifOp.getResults()[i++], false));

    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitWhileStatement(DaphneDSLGrammarParser::WhileStatementContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);

    auto ip = builder.saveInsertionPoint();

    // The two blocks for the SCF WhileOp.
    auto beforeBlock = new mlir::Block;
    auto afterBlock = new mlir::Block;

    const bool isDoWhile = ctx->KW_DO();

    mlir::Value cond;
    ScopedSymbolTable::SymbolTable ow;
    if (isDoWhile) { // It's a do-while loop.
        builder.setInsertionPointToEnd(beforeBlock);

        // Scope for body and condition, such that condition can see the body's
        // updates to variables existing before the loop.
        symbolTable.pushScope();

        // The body gets its own scope to not expose variables created inside
        // the body to the condition. While this is unnecessary if the body is
        // a block statement, there are nasty cases if no block statement is
        // used.
        symbolTable.pushScope();
        visit(ctx->bodyStmt);
        ow = symbolTable.popScope();

        // Make the body's updates visible to the condition.
        symbolTable.put(ow);

        cond = utils.castBoolIf(valueOrErrorOnVisit(ctx->cond));

        symbolTable.popScope();
    } else { // It's a while loop.
        builder.setInsertionPointToEnd(beforeBlock);
        cond = utils.castBoolIf(valueOrErrorOnVisit(ctx->cond));

        builder.setInsertionPointToEnd(afterBlock);
        symbolTable.pushScope();
        visit(ctx->bodyStmt);
        ow = symbolTable.popScope();
    }

    // Determine which variables created before the loop are updated in the
    // loop's body. These become the arguments and results of the WhileOp and
    // its "before" and "after" region.
    std::vector<mlir::Value> owVals;
    std::vector<mlir::Type> resultTypes;
    std::vector<mlir::Value> whileOperands;
    for (auto it = ow.begin(); it != ow.end(); it++) {
        mlir::Value owVal = it->second.value;
        mlir::Type type = owVal.getType();
        auto owLoc = owVal.getLoc();

        owVals.push_back(owVal);
        resultTypes.push_back(type);

        mlir::Value oldVal = symbolTable.get(it->first).value;
        whileOperands.push_back(oldVal);

        beforeBlock->addArgument(type, owLoc);
        afterBlock->addArgument(type, owLoc);
    }

    // Create the ConditionOp of the "before" block.
    builder.setInsertionPointToEnd(beforeBlock);
    if (isDoWhile)
        builder.create<mlir::scf::ConditionOp>(loc, cond, owVals);
    else
        builder.create<mlir::scf::ConditionOp>(loc, cond, beforeBlock->getArguments());

    // Create the YieldOp of the "after" block.
    builder.setInsertionPointToEnd(afterBlock);
    if (isDoWhile)
        builder.create<mlir::scf::YieldOp>(loc, afterBlock->getArguments());
    else
        builder.create<mlir::scf::YieldOp>(loc, owVals);

    builder.restoreInsertionPoint(ip);

    // Create the SCF WhileOp and insert the "before" and "after" blocks.
    auto whileOp = builder.create<mlir::scf::WhileOp>(loc, resultTypes, whileOperands);
    whileOp.getBefore().push_back(beforeBlock);
    whileOp.getAfter().push_back(afterBlock);

    size_t i = 0;
    for (auto &it : ow) {
        // Replace usages of the variables updated in the loop's body by the
        // corresponding block arguments.
        whileOperands[i].replaceUsesWithIf(beforeBlock->getArgument(i), [&](mlir::OpOperand &operand) {
            auto parentRegion = operand.getOwner()->getBlock()->getParent();
            return parentRegion != nullptr && whileOp.getBefore().isAncestor(parentRegion);
        });
        whileOperands[i].replaceUsesWithIf(afterBlock->getArgument(i), [&](mlir::OpOperand &operand) {
            auto parentRegion = operand.getOwner()->getBlock()->getParent();
            return parentRegion != nullptr && whileOp.getAfter().isAncestor(parentRegion);
        });

        // Rewire the results of the WhileOp to their variable names.
        symbolTable.put(it.first, ScopedSymbolTable::SymbolInfo(whileOp.getResults()[i++], false));
    }

    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitForStatement(DaphneDSLGrammarParser::ForStatementContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);

    // The type we assume for from, to, and step.
    mlir::Type t = builder.getIntegerType(64, true);

    // Parse from, to, and step.
    mlir::Value from = utils.castIf(t, valueOrErrorOnVisit(ctx->from));
    mlir::Value to = utils.castIf(t, valueOrErrorOnVisit(ctx->to));
    mlir::Value step;
    mlir::Value direction; // count upwards (+1) or downwards (-1)
    if (ctx->step) {
        // If the step is given, parse it and derive the counting direction.
        step = utils.castIf(t, valueOrErrorOnVisit(ctx->step));
        direction = builder.create<mlir::daphne::EwSignOp>(loc, t, step);
    } else {
        // If the step is not given, derive it as `-1 + 2 * (to >= from)`,
        // which always results in -1 or +1, even if to equals from.
        step = builder.create<mlir::daphne::EwAddOp>(
            loc, t, builder.create<mlir::daphne::ConstantOp>(loc, t, builder.getIntegerAttr(t, -1)),
            builder.create<mlir::daphne::EwMulOp>(
                loc, t, builder.create<mlir::daphne::ConstantOp>(loc, t, builder.getIntegerAttr(t, 2)),
                utils.castIf(t, builder.create<mlir::daphne::EwGeOp>(loc, t, to, from))));
        direction = step;
    }
    // Compensate for the fact that the upper bound of SCF's ForOp is exclusive,
    // while we want it to be inclusive.
    to = builder.create<mlir::daphne::EwAddOp>(loc, t, to, direction);
    // Compensate for the fact that SCF's ForOp can only count upwards.
    from = builder.create<mlir::daphne::EwMulOp>(loc, t, from, direction);
    to = builder.create<mlir::daphne::EwMulOp>(loc, t, to, direction);
    step = builder.create<mlir::daphne::EwMulOp>(loc, t, step, direction);
    // Compensate for the fact that SCF's ForOp expects its parameters to be of
    // MLIR's IndexType.
    mlir::Type idxType = builder.getIndexType();
    from = utils.castIf(idxType, from);
    to = utils.castIf(idxType, to);
    step = utils.castIf(idxType, step);

    auto ip = builder.saveInsertionPoint();

    // A block for the body of the for-loop.
    mlir::Block bodyBlock;
    builder.setInsertionPointToEnd(&bodyBlock);
    symbolTable.pushScope();

    // A placeholder for the loop's induction variable, since we do not know it
    // yet; will be replaced later.
    mlir::Value ph = builder.create<mlir::daphne::ConstantOp>(loc, builder.getIndexType(), builder.getIndexAttr(123));
    // Make the induction variable available by the specified name.
    symbolTable.put(ctx->var->getText(),
                    ScopedSymbolTable::SymbolInfo(
                        // Un-compensate for counting direction.
                        builder.create<mlir::daphne::EwMulOp>(loc, t, utils.castIf(t, ph), direction),
                        true // the for-loop's induction variable is read-only
                        ));

    // Parse the loop's body.
    visit(ctx->bodyStmt);

    // Determine which variables created before the loop are updated in the
    // loop's body. These become the arguments and results of the ForOp.
    ScopedSymbolTable::SymbolTable ow = symbolTable.popScope();
    std::vector<mlir::Value> resVals;
    std::vector<mlir::Value> forOperands;
    for (auto it = ow.begin(); it != ow.end(); it++) {
        resVals.push_back(it->second.value);
        forOperands.push_back(symbolTable.get(it->first).value);
    }

    builder.create<mlir::scf::YieldOp>(loc, resVals);

    builder.restoreInsertionPoint(ip);

    // Helper function for moving the operations in the block created above
    // into the actual body of the ForOp.
    auto insertBodyBlock = [&](mlir::OpBuilder &nested, mlir::Location loc, mlir::Value iv, mlir::ValueRange lcv) {
        nested.getBlock()->getOperations().splice(nested.getBlock()->end(), bodyBlock.getOperations());
    };

    // Create the actual ForOp.
    auto forOp = builder.create<mlir::scf::ForOp>(loc, from, to, step, forOperands, insertBodyBlock);

    // Substitute the induction variable, now that we know it.
    ph.replaceAllUsesWith(forOp.getInductionVar());

    size_t i = 0;
    for (auto it = ow.begin(); it != ow.end(); it++) {
        // Replace usages of the variables updated in the loop's body by the
        // corresponding block arguments.
        forOperands[i].replaceUsesWithIf(forOp.getRegionIterArgs()[i], [&](mlir::OpOperand &operand) -> bool {
            auto parentRegion = operand.getOwner()->getBlock()->getParent();
            return parentRegion != nullptr && forOp.getBodyRegion().isAncestor(parentRegion);
        });

        // Rewire the results of the ForOp to their variable names.
        symbolTable.put(it->first, ScopedSymbolTable::SymbolInfo(forOp.getResults()[i], false));

        i++;
    }

    return nullptr;
}

antlrcpp::Any DaphneDSLVisitor::visitLiteralExpr(DaphneDSLGrammarParser::LiteralExprContext *ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitArgExpr(DaphneDSLGrammarParser::ArgExprContext *ctx) {
    // Retrieve the name of the referenced CLI argument.
    std::string arg = ctx->arg->getText();

    // Find out if this argument was specified on the command line.
    auto it = args.find(arg);
    if (it == args.end())
        throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                          "argument " + arg +
                                              " referenced, but not provided as a command line argument");

    std::string argValue = it->second;
    bool hasMinus = false;
    if (!argValue.empty() && argValue[0] == '-') {
        hasMinus = true;
        argValue = argValue.substr(1);
    }

    // Parse the argument value as a literal
    // TODO: fix for string literals when " are not escaped or not present
    std::istringstream stream(argValue);
    antlr4::ANTLRInputStream input(stream);
    input.name = "argument";
    DaphneDSLGrammarLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    DaphneDSLGrammarParser parser(&tokens);

    CancelingErrorListener errorListener;
    lexer.removeErrorListeners();
    lexer.addErrorListener(&errorListener);
    parser.removeErrorListeners();
    parser.addErrorListener(&errorListener);

    DaphneDSLGrammarParser::LiteralContext *literalCtx = nullptr;
    try {
        literalCtx = parser.literal();
        if (tokens.LA(1) != antlr4::Token::EOF) {
            // Ensure entire input is consumed; if not, it's not a valid literal
            throw std::runtime_error("Extra input after literal");
        }
    } catch (std::exception &e) {
        throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                          "invalid literal value for argument '" + arg + "': " + argValue);
    }

    mlir::Value lit = visitLiteral(literalCtx);
    if (!hasMinus)
        return lit;
    else
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwMinusOp>(utils.getLoc(ctx->start), utils.unknownType, lit));
}

antlrcpp::Any DaphneDSLVisitor::visitIdentifierExpr(DaphneDSLGrammarParser::IdentifierExprContext *ctx) {
    std::string var;
    const auto &identifierVec = ctx->IDENTIFIER();
    for (size_t s = 0; s < identifierVec.size(); s++)
        var += (s < identifierVec.size() - 1) ? identifierVec[s]->getText() + '.' : identifierVec[s]->getText();

    try {
        return symbolTable.get(var).value;
    } catch (std::runtime_error &) {
        throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                          "variable `" + var + "` referenced before assignment");
    }
}

antlrcpp::Any DaphneDSLVisitor::visitParanthesesExpr(DaphneDSLGrammarParser::ParanthesesExprContext *ctx) {
    return valueOrErrorOnVisit(ctx->expr());
}

antlrcpp::Any DaphneDSLVisitor::visitCallArgs(DaphneDSLGrammarParser::CallArgsContext *ctx) {
    return visitChildren(ctx);
}

antlrcpp::Any DaphneDSLVisitor::visitCallArg(DaphneDSLGrammarParser::CallArgContext *ctx) { return visitChildren(ctx); }

bool DaphneDSLVisitor::argAndUDFParamCompatible(mlir::Type argTy, mlir::Type paramTy) const {
    auto argMatTy = llvm::dyn_cast<mlir::daphne::MatrixType>(argTy);
    auto paramMatTy = llvm::dyn_cast<mlir::daphne::MatrixType>(paramTy);

    // TODO This is rather a workaround than a thorough solution, since
    // unknown argument types do not really allow to check compatibility.

    // Argument type and parameter type are compatible if...
    return
        // ...they are the same, OR
        paramTy == argTy ||
        // ...at least one of them is unknown, OR
        argTy == utils.unknownType || paramTy == utils.unknownType ||
        // ...they are both matrices and at least one of them is of unknown
        // value type.
        (argMatTy && paramMatTy &&
         (argMatTy.getElementType() == utils.unknownType || paramMatTy.getElementType() == utils.unknownType));
}

std::optional<mlir::func::FuncOp> DaphneDSLVisitor::findMatchingUDF(const std::string &functionName,
                                                                    const std::vector<mlir::Value> &args,
                                                                    mlir::Location loc) const {
    // search user defined functions
    auto range = functionsSymbolMap.equal_range(functionName);
    // TODO: find not only a matching version, but the `most` specialized
    for (auto it = range.first; it != range.second; ++it) {
        auto userDefinedFunc = it->second;
        auto funcTy = userDefinedFunc.getFunctionType();
        auto compatible = true;

        if (funcTy.getNumInputs() != args.size()) {
            continue;
        }
        for (auto compIt : llvm::zip(funcTy.getInputs(), args)) {
            auto funcParamType = std::get<0>(compIt);
            auto argVal = std::get<1>(compIt);

            if (!argAndUDFParamCompatible(argVal.getType(), funcParamType)) {
                compatible = false;
                break;
            }
        }
        if (compatible) {
            return userDefinedFunc;
        }
    }
    // UDF with the provided name exists, but no version matches the argument
    // types
    if (range.second != range.first) {
        // FIXME: disallow user-defined function with same name as builtins,
        // otherwise this would be wrong behaviour
        std::stringstream s;
        s << "no definition of function `" << functionName << "` for argument types (";
        for (size_t i = 0; i < args.size(); i++) {
            s << args[i].getType();
            if (i < args.size() - 1)
                s << ", ";
        }
        // TODO For each available option, also say why it is not applicable
        // (which type isn't compatible).
        // TODO For each available option, also say where it is defined.
        s << "), available options: ";
        const size_t numOptions = functionsSymbolMap.count(functionName);
        size_t i = 0;
        for (auto it = range.first; it != range.second; ++it, ++i) {
            s << functionName << '(';
            auto userDefinedFunc = it->second;
            auto funcTy = userDefinedFunc.getFunctionType();
            for (size_t k = 0; k < funcTy.getNumInputs(); k++) {
                s << funcTy.getInput(k);
                if (k < funcTy.getNumInputs() - 1)
                    s << ", ";
            }
            s << ')';
            if (i < numOptions - 1)
                s << ", ";
        }
        throw ErrorHandler::compilerError(loc, "DSLVisitor", s.str());
    }

    // UDF with the provided name does not exist
    return std::nullopt;
}

std::optional<mlir::func::FuncOp>
DaphneDSLVisitor::findMatchingUnaryUDF(mlir::Location loc, const std::string &functionName, mlir::Type argType) const {
    // search user defined functions
    auto range = functionsSymbolMap.equal_range(functionName);

    // TODO: find not only a matching version, but the `most` specialized
    for (auto it = range.first; it != range.second; ++it) {
        auto userDefinedFunc = it->second;
        auto funcTy = userDefinedFunc.getFunctionType();

        if (funcTy.getNumInputs() != 1) {
            continue;
        }

        if (argAndUDFParamCompatible(argType, funcTy.getInput(0))) {
            return userDefinedFunc;
        }
    }
    // UDF with the provided name exists, but no version matches the argument
    // types
    if (range.second != range.first) {
        // FIXME: disallow user-defined function with same name as builtins,
        // otherwise this would be wrong behaviour
        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                          "No function definition of `" + functionName + "` found with matching types");
    }

    // UDF with the provided name does not exist
    return std::nullopt;
}

antlrcpp::Any DaphneDSLVisitor::handleMapOpCall(DaphneDSLGrammarParser::CallExprContext *ctx,
                                                const std::vector<CallArgInfo> &callArgs) {
    std::string func;
    const auto &identifierVec = ctx->IDENTIFIER();
    for (size_t s = 0; s < identifierVec.size(); s++)
        func += (s < identifierVec.size() - 1) ? identifierVec[s]->getText() + '.' : identifierVec[s]->getText();

    mlir::Location loc = utils.getLoc(ctx->start);

    if (func != "map")
        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                          "called 'handleMapOpCall' for function " + func + " instead of 'map'");

    std::vector<const CallArgInfo *> orderedSources;
    // Evaluate only the data argument, the UDF name is kept as string.
    if (!callArgs.empty()) {
        auto &firstArg = const_cast<CallArgInfo &>(callArgs[0]);
        firstArg.value = valueOrErrorOnVisit(firstArg.ctx->expr());
    }
    auto orderedArgs = orderCallArguments(func, loc, {"arg", "udf"}, 2, callArgs, &orderedSources);

    if (orderedArgs.size() != 2) {
        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                          "built-in function 'map' expects exactly 2 argument(s), but got " +
                                              std::to_string(callArgs.size()));
    }

    std::vector<mlir::Value> args;

    auto argVal = orderedArgs[0];
    args.push_back(argVal);

    auto argMatTy = llvm::dyn_cast<mlir::daphne::MatrixType>(argVal.getType());
    if (!argMatTy)
        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                          "built-in function 'map' expects argument of type matrix as its "
                                          "first parameter");

    std::string udfName = orderedSources[1]->ctx->expr()->getText();
    auto maybeUDF = findMatchingUnaryUDF(loc, udfName, argMatTy.getElementType());

    if (!maybeUDF)
        throw ErrorHandler::compilerError(loc, "DSLVisitor", "No function definition of `" + udfName + "` found");

    args.push_back(
        static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(loc, maybeUDF->getSymName().str())));

    // Create DaphneIR operation for the built-in function.
    return builtins.build(loc, func, args);
}

antlrcpp::Any DaphneDSLVisitor::visitCallExpr(DaphneDSLGrammarParser::CallExprContext *ctx) {
    std::string func;
    const auto &identifierVec = ctx->IDENTIFIER();
    bool hasKernelHint = ctx->kernel != nullptr;
    for (size_t s = 0; s < identifierVec.size() - 1 - hasKernelHint; s++)
        func += identifierVec[s]->getText() + '.';
    func += ctx->func->getText();
    mlir::Location loc = utils.getLoc(ctx->start);

    if (func == "map")
        return handleMapOpCall(ctx, collectCallArgs(ctx, /*evaluate*/ false));

    auto callArgs = collectCallArgs(ctx);

    // Try user-defined functions first.
    std::vector<mlir::Value> args_vec;
    if (functionsSymbolMap.count(func)) {
        auto paramNames = getUdfParamNames(func, callArgs.size());
        if (!paramNames) {
            if (hasNamedArguments(callArgs))
                throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                                  "missing parameter names for user-defined function `" + func + "`");
            // No metadata available but no keyword arguments: proceed with positional mapping.
            args_vec.reserve(callArgs.size());
            for (const auto &arg : callArgs)
                args_vec.push_back(arg.value);
        } else
            args_vec = orderCallArguments(func, loc, *paramNames, paramNames->size(), callArgs);
        auto maybeUDF = findMatchingUDF(func, args_vec, loc);
        if (maybeUDF) {
            if (hasKernelHint)
                throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                                  "kernel hints are not supported for calls to user-defined "
                                                  "functions");

            auto funcTy = maybeUDF->getFunctionType();
            auto co =
                builder.create<mlir::daphne::GenericCallOp>(loc, maybeUDF->getSymName(), args_vec, funcTy.getResults());
            if (funcTy.getNumResults() > 1)
                return co.getResults();
            else if (funcTy.getNumResults() == 1)
                return co.getResult(0);
            else
                return nullptr;
        }
    }

    // Parse arguments for built-ins (with keyword reordering).
    std::vector<std::string> builtinParamNames;
    size_t builtinRequired = 0;
    if (auto info = builtins.getParamNames(func)) {
        builtinParamNames = info->paramNames;
        builtinRequired = info->requiredParams;
    } else if (hasNamedArguments(callArgs))
        // If the built-in exposes no parameter metadata, we cannot map keyword
        // arguments to parameters.
        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                          "built-in function `" + func + "` does not support named arguments");

    args_vec = orderCallArguments(func, loc, builtinParamNames, builtinRequired, callArgs);

    // Create DaphneIR operation for the built-in function.
    antlrcpp::Any res = builtins.build(loc, func, args_vec);

    if (hasKernelHint) {
        std::string kernel = ctx->kernel->getText();

        // We deliberately don't check if the specified kernel
        // is registered for the created kind of operation,
        // since this is checked in RewriteToCallKernelOpPass.

        mlir::Operation *op;
        if (res.is<mlir::Operation *>()) // DaphneIR ops with exactly zero
                                         // results
            op = res.as<mlir::Operation *>();
        else if (res.is<mlir::Value>()) // DaphneIR ops with exactly one result
            op = res.as<mlir::Value>().getDefiningOp();
        else if (res.is<mlir::ResultRange>()) { // DaphneIR ops with more than
                                                // one results
            auto rr = res.as<mlir::ResultRange>();
            op = rr[0].getDefiningOp();
            // Normally, all values in the ResultRange should be results of
            // the same op, but we check it nevertheless, just to be sure.
            for (size_t i = 1; i < rr.size(); i++)
                if (rr[i].getDefiningOp() != op)
                    throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                                      "the given kernel hint `" + kernel +
                                                          "` cannot be applied since the DaphneIR operation "
                                                          "created for the built-in function `" +
                                                          func + "` is ambiguous");
        } else // unexpected case
            throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                              "the given kernel hint `" + kernel +
                                                  "` cannot be applied since the DaphneIR operation created "
                                                  "for the built-in function `" +
                                                  func + "` was not returned in a supported way");

        // TODO Don't hardcode the attribute name.
        op->setAttr("kernel_hint", builder.getStringAttr(kernel));
    }

    return res;
}

antlrcpp::Any DaphneDSLVisitor::visitCastExpr(DaphneDSLGrammarParser::CastExprContext *ctx) {
    mlir::Type resType;

    if (ctx->DATA_TYPE()) {
        std::string dtStr = ctx->DATA_TYPE()->getText();
        if (dtStr == "matrix") {
            mlir::Type vt;
            if (ctx->VALUE_TYPE())
                vt = utils.getValueTypeByName(ctx->VALUE_TYPE()->getText());
            else {
                vt = valueOrErrorOnVisit(ctx->expr()).getType();
                if (llvm::isa<mlir::daphne::FrameType>(vt))
                    // TODO Instead of using the value type of the first frame
                    // column as the value type of the matrix, we should better
                    // use the most general of all column types.
                    vt = llvm::dyn_cast<mlir::daphne::FrameType>(vt).getColumnTypes()[0];
                if (llvm::isa<mlir::daphne::MatrixType>(vt))
                    vt = llvm::dyn_cast<mlir::daphne::MatrixType>(vt).getElementType();
            }
            resType = utils.matrixOf(vt);
        } else if (dtStr == "frame") {
            // Currently does not support casts of type "Specify value type
            // only" (e.g., as.si64(x)) and "Specify data type and value type"
            // (e.g., as.frame<[si64, f64]>(x))
            std::vector<mlir::Type> colTypes;
            // TODO Take the number of columns into account.
            if (ctx->VALUE_TYPE())
                throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                                  "casting to a frame with particular column types is not "
                                                  "supported yet");
            // colTypes =
            // {utils.getValueTypeByName(ctx->VALUE_TYPE()->getText())};
            else {
                // TODO This fragment should be factored out, such that we can
                // reuse it for matrix/frame/scalar.
                mlir::Type argType = valueOrErrorOnVisit(ctx->expr()).getType();
                if (llvm::isa<mlir::daphne::MatrixType>(argType))
                    colTypes = {llvm::dyn_cast<mlir::daphne::MatrixType>(argType).getElementType()};
                else if (llvm::isa<mlir::daphne::FrameType>(argType))
                    // TODO Instead of using the value type of the first frame
                    // column as the value type of the matrix, we should better
                    // use the most general of all column types.
                    colTypes = {llvm::dyn_cast<mlir::daphne::FrameType>(argType).getColumnTypes()[0]};
                else
                    colTypes = {argType};
            }
            resType = mlir::daphne::FrameType::get(builder.getContext(), colTypes);
        } else if (dtStr == "scalar") {
            if (ctx->VALUE_TYPE())
                resType = utils.getValueTypeByName(ctx->VALUE_TYPE()->getText());
            else {
                // TODO This fragment should be factored out, such that we can
                // reuse it for matrix/frame/scalar.
                mlir::Type argType = valueOrErrorOnVisit(ctx->expr()).getType();
                if (llvm::isa<mlir::daphne::MatrixType>(argType))
                    resType = llvm::dyn_cast<mlir::daphne::MatrixType>(argType).getElementType();
                else if (llvm::isa<mlir::daphne::FrameType>(argType))
                    // TODO Instead of using the value type of the first frame
                    // column as the value type of the matrix, we should better
                    // use the most general of all column types.
                    resType = llvm::dyn_cast<mlir::daphne::FrameType>(argType).getColumnTypes()[0];
                else
                    resType = argType;
            }
        } else
            throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                              "unsupported data type in cast expression: " + dtStr);
    } else if (ctx->VALUE_TYPE()) { // Data type shall be retained
        mlir::Type vt = utils.getValueTypeByName(ctx->VALUE_TYPE()->getText());
        mlir::Type argTy = valueOrErrorOnVisit(ctx->expr()).getType();
        if (llvm::isa<mlir::daphne::MatrixType>(argTy))
            resType = utils.matrixOf(vt);
        else if (llvm::isa<mlir::daphne::FrameType>(argTy)) {
            throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                              "casting to a frame with particular column types is not "
                                              "supported yet");
            // size_t numCols =
            // llvm::dyn_cast<mlir::daphne::FrameType>(argTy).getColumnTypes().size();
            // std::vector<mlir::Type> colTypes(numCols, vt);
            // resType = mlir::daphne::FrameType::get(builder.getContext(),
            // colTypes);
        } else if (llvm::isa<mlir::daphne::UnknownType>(argTy))
            resType = utils.unknownType;
        else
            resType = vt;
    } else
        throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                          "casting requires the specification of the target data and/or "
                                          "value type");

    return static_cast<mlir::Value>(
        builder.create<mlir::daphne::CastOp>(utils.getLoc(ctx->start), resType, valueOrErrorOnVisit(ctx->expr())));
}

antlrcpp::Any DaphneDSLVisitor::visitRightIdxFilterExpr(DaphneDSLGrammarParser::RightIdxFilterExprContext *ctx) {
    mlir::Value obj = valueOrErrorOnVisit(ctx->obj);

    if (ctx->rows) // rows specified
        obj = builder.create<mlir::daphne::FilterRowOp>(utils.getLoc(ctx->rows->start), obj.getType(), obj,
                                                        valueOrErrorOnVisit(ctx->rows));
    if (ctx->cols) // cols specified
        obj = builder.create<mlir::daphne::FilterColOp>(utils.getLoc(ctx->cols->start),
                                                        obj.getType(), // TODO Not correct for frames, see #484.
                                                        obj, valueOrErrorOnVisit(ctx->cols));

    // Note: If rows and cols are specified, we create two filter steps.
    // This can be inefficient, but it is simpler for now.
    // TODO Create a combined FilterOp

    // Note: If neither rows nor cols are specified, we simply return the
    // object.

    return obj;
}

antlrcpp::Any DaphneDSLVisitor::visitRightIdxExtractExpr(DaphneDSLGrammarParser::RightIdxExtractExprContext *ctx) {
    mlir::Value obj = valueOrErrorOnVisit(ctx->obj);

    auto indexing = visit(ctx->idx).as<std::pair<std::pair<bool, antlrcpp::Any>, std::pair<bool, antlrcpp::Any>>>();
    auto rows = indexing.first;
    auto cols = indexing.second;

    // TODO Use location of rows/cols in utils.getLoc(...) for better
    // error messages.
    if (rows.first) // rows specified
        obj = applyRightIndexing<mlir::daphne::ExtractRowOp, mlir::daphne::SliceRowOp, mlir::daphne::NumRowsOp>(
            utils.getLoc(ctx->idx->start), obj, rows.second, false);
    if (cols.first) // cols specified
        obj = applyRightIndexing<mlir::daphne::ExtractColOp, mlir::daphne::SliceColOp, mlir::daphne::NumColsOp>(
            utils.getLoc(ctx->idx->start), obj, cols.second, llvm::isa<mlir::daphne::FrameType>(obj.getType()));

    // Note: If rows and cols are specified, we create two extraction steps.
    // This can be inefficient, but it is simpler for now.
    // TODO Create a combined ExtractOp/SliceOp.

    // Note: If neither rows nor cols are specified, we simply return the
    // object.

    return obj;
}

antlrcpp::Any DaphneDSLVisitor::visitMinusExpr(DaphneDSLGrammarParser::MinusExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value arg = valueOrErrorOnVisit(ctx->arg);

    if (op == "-")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwMinusOp>(loc, utils.unknownType, arg));
    if (op == "+")
        return arg;

    throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitMatmulExpr(DaphneDSLGrammarParser::MatmulExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
    mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);

    if (op == "@") {
        mlir::Value f = builder.create<mlir::daphne::ConstantOp>(loc, false);
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::MatMulOp>(loc, lhs.getType(), lhs, rhs, f, f));
    }

    throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitPowExpr(DaphneDSLGrammarParser::PowExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
    mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);

    if (op == "^")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwPowOp>(loc, utils.unknownType, lhs, rhs));

    throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitModExpr(DaphneDSLGrammarParser::ModExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
    mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);

    if (op == "%")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwModOp>(loc, utils.unknownType, lhs, rhs));

    throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitMulExpr(DaphneDSLGrammarParser::MulExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
    mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);
    bool hasKernelHint = ctx->kernel != nullptr;

    mlir::Value res = nullptr;
    if (op == "*")
        res = CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwMulOp>(loc, utils.unknownType, lhs, rhs));
    if (op == "/")
        res = CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwDivOp>(loc, utils.unknownType, lhs, rhs));

    if (hasKernelHint) {
        std::string kernel = ctx->kernel->getText();

        // We deliberately don't check if the specified kernel
        // is registered for the created kind of operation,
        // since this is checked in RewriteToCallKernelOpPass.

        mlir::Operation *op = res.getDefiningOp();
        // TODO Don't hardcode the attribute name.
        op->setAttr("kernel_hint", builder.getStringAttr(kernel));
    }

    if (res)
        return res;
    else
        throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitAddExpr(DaphneDSLGrammarParser::AddExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
    mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);
    bool hasKernelHint = ctx->kernel != nullptr;

    mlir::Value res = nullptr;
    if (op == "+")
        // Note that we use '+' for both addition (EwAddOp) and concatenation
        // (EwConcatOp). The choice is made based on the types of the operands
        // (if one operand is a string, we choose EwConcatOp). However, the
        // types might not be known at this point in time. Thus, we always
        // create an EwAddOp here. Note that EwAddOp has a canonicalize method
        // rewriting it to EwConcatOp if necessary.
        res = CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwAddOp>(loc, utils.unknownType, lhs, rhs));
    if (op == "-")
        res = CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwSubOp>(loc, utils.unknownType, lhs, rhs));

    if (hasKernelHint) {
        std::string kernel = ctx->kernel->getText();

        // We deliberately don't check if the specified kernel
        // is registered for the created kind of operation,
        // since this is checked in RewriteToCallKernelOpPass.

        mlir::Operation *op = res.getDefiningOp();
        // TODO Don't hardcode the attribute name.
        op->setAttr("kernel_hint", builder.getStringAttr(kernel));

        // TODO retain the attr in case EwAddOp is rewritten to EwConcatOp.
    }

    if (res)
        return res;
    else
        throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitCmpExpr(DaphneDSLGrammarParser::CmpExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
    mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);

    if (op == "==")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwEqOp>(loc, utils.unknownType, lhs, rhs));
    if (op == "!=")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwNeqOp>(loc, utils.unknownType, lhs, rhs));
    if (op == "<")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwLtOp>(loc, utils.unknownType, lhs, rhs));
    if (op == "<=")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwLeOp>(loc, utils.unknownType, lhs, rhs));
    if (op == ">")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwGtOp>(loc, utils.unknownType, lhs, rhs));
    if (op == ">=")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwGeOp>(loc, utils.unknownType, lhs, rhs));

    throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitConjExpr(DaphneDSLGrammarParser::ConjExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
    mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);

    if (op == "&&")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwAndOp>(loc, utils.unknownType, lhs, rhs));

    throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitDisjExpr(DaphneDSLGrammarParser::DisjExprContext *ctx) {
    std::string op = ctx->op->getText();
    mlir::Location loc = utils.getLoc(ctx->op);
    mlir::Value lhs = valueOrErrorOnVisit(ctx->lhs);
    mlir::Value rhs = valueOrErrorOnVisit(ctx->rhs);

    if (op == "||")
        return CompilerUtils::retValWithInferredType(
            builder.create<mlir::daphne::EwOrOp>(loc, utils.unknownType, lhs, rhs));

    throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected op symbol");
}

antlrcpp::Any DaphneDSLVisitor::visitCondExpr(DaphneDSLGrammarParser::CondExprContext *ctx) {
    return static_cast<mlir::Value>(builder.create<mlir::daphne::CondOp>(
        utils.getLoc(ctx->start), utils.unknownType, valueOrErrorOnVisit(ctx->cond), valueOrErrorOnVisit(ctx->thenExpr),
        valueOrErrorOnVisit(ctx->elseExpr)));
}

template <typename VT>
mlir::Value DaphneDSLVisitor::buildColMatrixFromValues(mlir::Location loc, const std::vector<mlir::Value> &values,
                                                       const std::vector<mlir::Type> &valueTypes, mlir::Type matrixVt) {
    std::shared_ptr<VT[]> constValues = std::shared_ptr<VT[]>(new VT[values.size()]);
    std::vector<int64_t> nonConstValsIdx;

    // convenience function
    auto fillRes = [&constValues, &nonConstValsIdx](int64_t i, std::pair<bool, auto> constValue) {
        if (constValue.first) {
            // currently supported types for matrix literals support conversions
            // to (most general) array's value type. if unsigned integers are
            // added, this can lead to conflicts
            constValues.get()[i] = constValue.second;
        } else {
            constValues.get()[i] = ValueTypeUtils::defaultValue<VT>;
            nonConstValsIdx.emplace_back(i);
        }
    };

    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        mlir::Value currentValue = values[i];
        mlir::Type currentType = valueTypes[i];

        if constexpr (std::is_same<VT, std::string>::value) {
            if (llvm::isa<mlir::daphne::StringType>(currentType))
                fillRes(i, CompilerUtils::isConstant<std::string>(currentValue));
            else
                throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
        } else {
            if (mlir::IntegerType valueIntType = llvm::dyn_cast<mlir::IntegerType>(currentType)) {
                if (currentType.isSignedInteger()) {
                    switch (valueIntType.getWidth()) {
                    case 64:
                        fillRes(i, CompilerUtils::isConstant<int64_t>(currentValue));
                        break;
                    case 32:
                        fillRes(i, CompilerUtils::isConstant<int32_t>(currentValue));
                        break;
                    case 8:
                        fillRes(i, CompilerUtils::isConstant<int8_t>(currentValue));
                        break;
                    default:
                        throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
                    }
                } else if (currentType.isUnsignedInteger()) {
                    switch (valueIntType.getWidth()) {
                    case 64:
                        fillRes(i, CompilerUtils::isConstant<uint64_t>(currentValue));
                        break;
                    case 32:
                        fillRes(i, CompilerUtils::isConstant<uint32_t>(currentValue));
                        break;
                    case 8:
                        fillRes(i, CompilerUtils::isConstant<uint8_t>(currentValue));
                        break;
                    default:
                        throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
                    }
                } else if (currentType.isSignlessInteger(1))
                    fillRes(i, CompilerUtils::isConstant<bool>(currentValue));
                else
                    throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
            } else if (currentType.isF64()) {
                fillRes(i, CompilerUtils::isConstant<double>(currentValue));
            } else if (currentType.isF32()) {
                fillRes(i, CompilerUtils::isConstant<float>(currentValue));
            } else {
                throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
            }
        }
    }

    auto mat = DataObjectFactory::create<DenseMatrix<VT>>(values.size(), 1, constValues);

    // Create a MatrixConstantOp backed by a DenseMatrix containing the
    // parse-time constant values from the DaphneDSL matrix literal (and zeros
    // for the remaining cells).
    mlir::Value result = static_cast<mlir::Value>(builder.create<mlir::daphne::MatrixConstantOp>(
        loc, utils.matrixOf(matrixVt), builder.create<mlir::daphne::ConstantOp>(loc, reinterpret_cast<uint64_t>(mat))));

    // Patch the cells corresponding to non-parse-time constant values from the
    // DaphneDSL matrix literal by creating InsertOps that insert the results of
    // the expressions.
    for (int64_t idx : nonConstValsIdx) {
        mlir::Value insValue = values[idx];

        // Cast the scalar expression result to the value type of the matrix, if
        // necessary.
        insValue = utils.castIf(matrixVt, insValue);

        // Cast the scalar expression result to a 1x1 matrix (required for
        // InsertOp).
        mlir::Value ins =
            static_cast<mlir::Value>(builder.create<mlir::daphne::CastOp>(loc, utils.matrixOf(matrixVt), insValue));

        // Maybe later these InsertOps can be fused into a single one
        // or replaced with InsertOps that support scalar input.
        result = static_cast<mlir::Value>(builder.create<mlir::daphne::InsertRowOp>(
            loc, utils.matrixOf(matrixVt), result, ins, builder.create<mlir::daphne::ConstantOp>(loc, idx),
            builder.create<mlir::daphne::ConstantOp>(loc, idx + 1)));
    }

    return result;
}

antlrcpp::Any DaphneDSLVisitor::visitMatrixLiteralExpr(DaphneDSLGrammarParser::MatrixLiteralExprContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);
    mlir::Value rows;
    mlir::Value cols;

    size_t numMatElems;

    // Validation of dimensions is left to reshape kernel.
    // Missing dimensions are inferred (defaults to column matrix).
    if (!ctx->rows && !ctx->cols) {
        numMatElems = ctx->expr().size();
        cols = builder.create<mlir::daphne::ConstantOp>(loc, static_cast<uint64_t>(1));
        rows = builder.create<mlir::daphne::ConstantOp>(loc, static_cast<uint64_t>(ctx->expr().size()));
    } else {
        numMatElems = (ctx->rows && ctx->cols) ? ctx->expr().size() - 2 : ctx->expr().size() - 1;
        if (ctx->cols && ctx->rows) {
            cols = valueOrErrorOnVisit(ctx->cols);
            rows = valueOrErrorOnVisit(ctx->rows);
        } else if (ctx->cols) {
            cols = valueOrErrorOnVisit(ctx->cols);
            rows =
                builder.create<mlir::daphne::EwDivOp>(loc, builder.getIntegerType(64, false),
                                                      builder.create<mlir::daphne::ConstantOp>(loc, static_cast<uint64_t>(numMatElems)), cols);
        } else {
            rows = valueOrErrorOnVisit(ctx->rows);
            cols =
                builder.create<mlir::daphne::EwDivOp>(loc, builder.getIntegerType(64, false),
                                                      builder.create<mlir::daphne::ConstantOp>(loc, static_cast<uint64_t>(numMatElems)), rows);
        }
    }
    cols = utils.castSizeIf(cols);
    rows = utils.castSizeIf(rows);

    if (numMatElems == 0)
        throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                          "empty matrix literals are not supported");

    std::vector<mlir::Value> values;
    std::vector<mlir::Type> valueTypes;
    values.reserve(numMatElems);
    valueTypes.reserve(numMatElems);
    for (size_t i = 0; i < numMatElems; ++i) {
        mlir::Value currentValue = valueOrErrorOnVisit(ctx->expr(i));
        values.emplace_back(currentValue);
        valueTypes.emplace_back(currentValue.getType());
    }

    mlir::Type valueType = mostGeneralVt(valueTypes);
    mlir::Value colMatrix;

    if (mlir::IntegerType valueIntType = llvm::dyn_cast<mlir::IntegerType>(valueType)) {
        if (valueType.isSignedInteger()) {
            switch (valueIntType.getWidth()) {
            case 64:
                colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<int64_t>(loc, values, valueTypes, valueType);
                break;
            case 32:
                colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<int32_t>(loc, values, valueTypes, valueType);
                break;
            case 8:
                colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<int8_t>(loc, values, valueTypes, valueType);
                break;
            default:
                throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
            }
        } else if (valueType.isUnsignedInteger()) {
            switch (valueIntType.getWidth()) {
            case 64:
                colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<uint64_t>(loc, values, valueTypes, valueType);
                break;
            case 32:
                colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<uint32_t>(loc, values, valueTypes, valueType);
                break;
            case 8:
                colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<uint8_t>(loc, values, valueTypes, valueType);
                break;
            default:
                throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
            }
        } else if (valueType.isSignlessInteger(1))
            colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<bool>(loc, values, valueTypes, valueType);
        else
            throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
    } else if (valueType.isF64())
        colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<double>(loc, values, valueTypes, valueType);
    else if (valueType.isF32())
        colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<float>(loc, values, valueTypes, valueType);
    else if (llvm::isa<mlir::daphne::StringType>(valueType))
        colMatrix = DaphneDSLVisitor::buildColMatrixFromValues<std::string>(loc, values, valueTypes, valueType);
    else {
        throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
    }

    // TODO: omit ReshapeOp if rows=1 (not always known at parse-time)
    mlir::Value result = static_cast<mlir::Value>(
        builder.create<mlir::daphne::ReshapeOp>(loc, utils.matrixOf(valueType), colMatrix, rows, cols));

    return result;
}

antlrcpp::Any
DaphneDSLVisitor::visitColMajorFrameLiteralExpr(DaphneDSLGrammarParser::ColMajorFrameLiteralExprContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);

    size_t labelCount = ctx->labels.size();
    size_t colCount = ctx->cols.size();

    if (labelCount != colCount)
        throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                          "frame literals must have an equal number of column labels and "
                                          "column matrices");

    std::vector<mlir::Value> parsedLabels;
    std::vector<mlir::Value> columnMatrices;
    std::vector<mlir::Type> columnMatElemType;
    parsedLabels.reserve(colCount);
    columnMatrices.reserve(colCount);
    columnMatElemType.reserve(colCount);

    for (size_t i = 0; i < colCount; ++i) {
        mlir::Value label = valueOrErrorOnVisit(ctx->labels[i]);
        mlir::Value mat = valueOrErrorOnVisit(ctx->cols[i]);

        if (label.getType() != utils.strType)
            throw ErrorHandler::compilerError(loc, "DSLVisitor", "labels for frame literals must be strings");
        if (!llvm::isa<mlir::daphne::MatrixType>(mat.getType()))
            throw ErrorHandler::compilerError(loc, "DSLVisitor", "columns for frame literals must be matrices");

        parsedLabels.emplace_back(label);
        columnMatrices.emplace_back(mat);
        auto matrixType = llvm::dyn_cast<mlir::daphne::MatrixType>(mat.getType());
        columnMatElemType.emplace_back(matrixType.getElementType());
    }

    mlir::Type frameColTypes = mlir::daphne::FrameType::get(builder.getContext(), columnMatElemType);

    mlir::Value result = static_cast<mlir::Value>(
        builder.create<mlir::daphne::CreateFrameOp>(loc, frameColTypes, columnMatrices, parsedLabels));

    return result;
}

antlrcpp::Any
DaphneDSLVisitor::visitRowMajorFrameLiteralExpr(DaphneDSLGrammarParser::RowMajorFrameLiteralExprContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);

    auto labelVectors = visit(ctx->labels).as<std::pair<std::vector<mlir::Value>, std::vector<mlir::Type>>>();
    auto parsedLabels = labelVectors.first;

    size_t cols = parsedLabels.size();
    size_t rows = ctx->rows.size();

    if (cols == 0 || rows == 0)
        throw ErrorHandler::compilerError(loc, "DSLVisitor", "empty frame literals are not supported");

    // validate label types
    for (mlir::Type labelType : labelVectors.second) {
        if (labelType != utils.strType)
            throw ErrorHandler::compilerError(loc, "DSLVisitor", "labels for frame literals must be strings");
    }

    // row-major matrices are converted to column-major format
    std::vector<std::vector<mlir::Value>> valuesVec;
    std::vector<std::vector<mlir::Type>> valueTypesVec;
    valuesVec.resize(cols);
    valueTypesVec.resize(cols);
    // reserve space for inner vectors
    for (size_t i = 0; i < cols; ++i) {
        valuesVec[i].reserve(rows);
        valueTypesVec[i].reserve(rows);
    }

    // build row vector and place values in the corresponding column
    for (size_t i = 0; i < rows; ++i) {
        auto rowVectors = visit(ctx->rows[i]).as<std::pair<std::vector<mlir::Value>, std::vector<mlir::Type>>>();

        if (rowVectors.first.size() != cols)
            throw ErrorHandler::compilerError(loc, "DSLVisitor", "size of row does not match the amount of labels");

        for (size_t j = 0; j < cols; ++j) {
            valuesVec[j].emplace_back(std::move(rowVectors.first[j]));
            valueTypesVec[j].emplace_back(std::move(rowVectors.second[j]));
        }
    }

    // determine most general value type in each column and
    // build column matrices from column vectors
    std::vector<mlir::Value> colValues;
    std::vector<mlir::Type> colTypes;
    colValues.reserve(cols);
    colTypes.reserve(cols);
    for (size_t i = 0; i < cols; ++i) {
        colTypes.emplace_back(mostGeneralVt(valueTypesVec[i]));

        if (mlir::IntegerType valueIntType = llvm::dyn_cast<mlir::IntegerType>(colTypes[i])) {
            if (colTypes[i].isSignedInteger()) {
                switch (valueIntType.getWidth()) {
                case 64:
                    colValues.emplace_back(DaphneDSLVisitor::buildColMatrixFromValues<int64_t>(
                        loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
                    break;
                case 32:
                    colValues.emplace_back(DaphneDSLVisitor::buildColMatrixFromValues<int32_t>(
                        loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
                    break;
                case 8:
                    colValues.emplace_back(DaphneDSLVisitor::buildColMatrixFromValues<int8_t>(
                        loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
                    break;
                default:
                    throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
                }
            } else if (colTypes[i].isUnsignedInteger()) {
                switch (valueIntType.getWidth()) {
                case 64:
                    colValues.emplace_back(DaphneDSLVisitor::buildColMatrixFromValues<uint64_t>(
                        loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
                    break;
                case 32:
                    colValues.emplace_back(DaphneDSLVisitor::buildColMatrixFromValues<uint32_t>(
                        loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
                    break;
                case 8:
                    colValues.emplace_back(DaphneDSLVisitor::buildColMatrixFromValues<uint8_t>(
                        loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
                    break;
                default:
                    throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
                }
            } else if (colTypes[i].isSignlessInteger(1))
                colValues.emplace_back(
                    DaphneDSLVisitor::buildColMatrixFromValues<bool>(loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
            else
                throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
        } else if (colTypes[i].isF64())
            colValues.emplace_back(
                DaphneDSLVisitor::buildColMatrixFromValues<double>(loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
        else if (colTypes[i].isF32())
            colValues.emplace_back(
                DaphneDSLVisitor::buildColMatrixFromValues<float>(loc, valuesVec[i], valueTypesVec[i], colTypes[i]));
        else {
            throw ErrorHandler::compilerError(loc, "DSLVisitor", "matrix literal of invalid value type");
        }
    }

    mlir::Type frameColTypes = mlir::daphne::FrameType::get(builder.getContext(), colTypes);

    mlir::Value result = static_cast<mlir::Value>(
        builder.create<mlir::daphne::CreateFrameOp>(loc, frameColTypes, colValues, parsedLabels));

    return result;
}

antlrcpp::Any DaphneDSLVisitor::visitFrameRow(DaphneDSLGrammarParser::FrameRowContext *ctx) {
    size_t elementCount = ctx->expr().size();
    std::vector<mlir::Value> values;
    std::vector<mlir::Type> types;
    values.reserve(elementCount);
    types.reserve(elementCount);
    for (size_t i = 0; i < elementCount; ++i) {
        mlir::Value currentValue = valueOrErrorOnVisit(ctx->expr(i));
        values.emplace_back(currentValue);
        types.emplace_back(currentValue.getType());
    }
    return std::make_pair(values, types);
}

antlrcpp::Any DaphneDSLVisitor::visitIndexing(DaphneDSLGrammarParser::IndexingContext *ctx) {
    auto rows = ctx->rows ? visit(ctx->rows).as<std::pair<bool, antlrcpp::Any>>()
                          : std::make_pair(false, antlrcpp::Any(nullptr));
    auto cols = ctx->cols ? visit(ctx->cols).as<std::pair<bool, antlrcpp::Any>>()
                          : std::make_pair(false, antlrcpp::Any(nullptr));
    return std::make_pair(rows, cols);
}

antlrcpp::Any DaphneDSLVisitor::visitRange(DaphneDSLGrammarParser::RangeContext *ctx) {
    if (ctx->pos)
        return std::make_pair(true, antlrcpp::Any(valueOrErrorOnVisit(ctx->pos)));
    else {
        mlir::Value posLowerIncl = ctx->posLowerIncl ? valueOrErrorOnVisit(ctx->posLowerIncl) : nullptr;
        mlir::Value posUpperExcl = ctx->posUpperExcl ? valueOrErrorOnVisit(ctx->posUpperExcl) : nullptr;
        return std::make_pair(posLowerIncl != nullptr || posUpperExcl != nullptr,
                              antlrcpp::Any(std::make_pair(posLowerIncl, posUpperExcl)));
    }
}

antlrcpp::Any DaphneDSLVisitor::visitLiteral(DaphneDSLGrammarParser::LiteralContext *ctx) {
    // TODO The creation of the ConstantOps could be simplified: We don't need
    // to create attributes here, since there are custom builder methods for
    // primitive C++ data types.
    mlir::Location loc = utils.getLoc(ctx->start);
    if (auto lit = ctx->INT_LITERAL()) {
        std::string litStr = lit->getText();

        // remove digit separators
        litStr = std::regex_replace(litStr, std::regex("_|'"), "");

        if (litStr.back() == 'u')
            return static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(loc, static_cast<uint64_t>(std::stoul(litStr))));
        else if ((litStr.length() > 2) && std::string_view(litStr).substr(litStr.length() - 3) == "ull") {
            // The suffix "ull" must be checked before the suffix "l", since "l"
            // is a suffix of "ull".
            return static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(loc, static_cast<uint64_t>(std::stoull(litStr))));
        } else if (litStr.back() == 'l')
            return static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(loc, static_cast<int64_t>(std::stol(litStr))));
        else if (litStr.back() == 'z') {
            return static_cast<mlir::Value>(
                builder.create<mlir::daphne::ConstantOp>(loc, static_cast<uint64_t>(std::stoll(litStr))));
        } else {
            // Note that a leading minus of a numeric literal is not parsed as
            // part of the literal itself, but handled separately as a unary
            // minus operator. Thus, this visitor actually sees the number
            // without the minus. This is problematic when a DaphneDSL script
            // contains the minimum int64 value -2^63, because without the
            // minus, 2^63 is beyond the range of int64, as the maximum int64
            // value is 2^63 - 1. Thus, we need a special case here.
            if (std::stoull(litStr) == (std::numeric_limits<int64_t>::max() + 1ull))
                return static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(
                    loc, static_cast<int64_t>(std::numeric_limits<int64_t>::min())));
            else
                return static_cast<mlir::Value>(
                    builder.create<mlir::daphne::ConstantOp>(loc, static_cast<int64_t>(std::stoll(litStr))));
        }
    }
    if (auto lit = ctx->FLOAT_LITERAL()) {
        std::string litStr = lit->getText();
        double val;
        if (litStr == "nan")
            val = std::numeric_limits<double>::quiet_NaN();
        else if (litStr == "nanf")
            val = std::numeric_limits<float>::quiet_NaN();
        else if (litStr == "inf")
            val = std::numeric_limits<double>::infinity();
        else if (litStr == "inff")
            val = std::numeric_limits<float>::infinity();
        else if (litStr == "-inf")
            val = -std::numeric_limits<double>::infinity();
        else if (litStr == "-inff")
            val = -std::numeric_limits<float>::infinity();
        else if (litStr.back() == 'f') {
            // remove digit separators
            litStr = std::regex_replace(litStr, std::regex("_|'"), "");
            auto fval = std::stof(litStr.c_str());
            return static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(loc, fval));
        } else {
            // remove digit separators
            litStr = std::regex_replace(litStr, std::regex("_|'"), "");
            val = std::atof(litStr.c_str());
        }
        return static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(loc, val));
    }
    if (ctx->bl)
        return visit(ctx->bl);
    if (auto lit = ctx->STRING_LITERAL()) {
        std::string val = lit->getText();

        // Remove quotation marks.
        val = val.substr(1, val.size() - 2);

        // Replace escape sequences.
        val = std::regex_replace(val, std::regex(R"(\\b)"), "\b");
        val = std::regex_replace(val, std::regex(R"(\\f)"), "\f");
        val = std::regex_replace(val, std::regex(R"(\\n)"), "\n");
        val = std::regex_replace(val, std::regex(R"(\\r)"), "\r");
        val = std::regex_replace(val, std::regex(R"(\\t)"), "\t");
        val = std::regex_replace(val, std::regex(R"(\\\")"), "\"");
        val = std::regex_replace(val, std::regex(R"(\\\\)"), "\\");

        return static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(loc, val));
    }
    throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected literal");
}

antlrcpp::Any DaphneDSLVisitor::visitBoolLiteral(DaphneDSLGrammarParser::BoolLiteralContext *ctx) {
    mlir::Location loc = utils.getLoc(ctx->start);
    bool val;
    if (ctx->KW_TRUE())
        val = true;
    else if (ctx->KW_FALSE())
        val = false;
    else
        throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor", "unexpected bool literal");

    return static_cast<mlir::Value>(builder.create<mlir::daphne::ConstantOp>(loc, val));
}

void removeTrailingOperationsAfterReturnOp(mlir::daphne::ReturnOp firstReturnOp, mlir::Block *block) {
    auto op = &block->getOperations().back();
    // erase in reverse order to ensure no uses will be left
    while (op != firstReturnOp) {
        auto prev = op->getPrevNode();
        op->emitWarning() << "Operation is ignored, as the function will return at " << firstReturnOp.getLoc();
        op->erase();
        op = prev;
    }
}

/**
 * @brief Ensures that the `caseBlock` has correct behaviour by appending
 * operations, as the other case has an early return.
 *
 * @param ifOpWithEarlyReturn The old `IfOp` with the early return
 * @param caseBlock The new block for the case without a `ReturnOp`
 */
void rectifyIfCaseWithoutReturnOp(mlir::scf::IfOp ifOpWithEarlyReturn, mlir::Block *caseBlock) {
    // ensure there is a `YieldOp` (for later removal of such)
    if (caseBlock->empty() || !llvm::isa<mlir::scf::YieldOp>(caseBlock->back())) {
        mlir::OpBuilder builder(ifOpWithEarlyReturn->getContext());
        builder.setInsertionPoint(caseBlock, caseBlock->end());
        builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc());
    }

    // As this if-case doesn't have an early return we need to move/clone
    // operations that should happen into this case.
    auto opsAfterIf = ifOpWithEarlyReturn->getNextNode();
    while (opsAfterIf) {
        auto next = opsAfterIf->getNextNode();
        if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(opsAfterIf)) {
            auto parentOp = llvm::dyn_cast<mlir::scf::IfOp>(yieldOp->getParentOp());
            if (!parentOp) {
                throw ErrorHandler::compilerError(yieldOp->getLoc(), "DSLVisitor",
                                                  "Early return not nested in `if`s not yet supported!");
            }
            next = parentOp->getNextNode();
        }
        if (opsAfterIf->getBlock() == ifOpWithEarlyReturn->getBlock()) {
            // can be moved inside if
            opsAfterIf->moveBefore(caseBlock, caseBlock->end());
        } else {
            // can't move them directly, need clone (operations will be needed
            // later)
            auto clonedOp = opsAfterIf->clone();
            mlir::OpBuilder builder(clonedOp->getContext());
            builder.setInsertionPoint(caseBlock, caseBlock->end());
            builder.insert(clonedOp);
        }
        opsAfterIf = next;
    }

    // Remove `YieldOp`s and replace the result values of `IfOp`s used by
    // operations that got moved in the previous loop with the correct values.
    auto currIfOp = ifOpWithEarlyReturn;
    auto currOp = &caseBlock->front();
    while (currOp) {
        auto nextOp = currOp->getNextNode();
        if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(currOp)) {
            // cast was checked in previous loop
            for (auto it : llvm::zip(currIfOp.getResults(), yieldOp.getOperands())) {
                auto ifResult = std::get<0>(it);
                auto yieldedVal = std::get<1>(it);
                ifResult.replaceUsesWithIf(yieldedVal, [&](mlir::OpOperand &opOperand) {
                    return opOperand.getOwner()->getBlock() == caseBlock;
                });
            }
            currIfOp = llvm::dyn_cast_or_null<mlir::scf::IfOp>(currIfOp->getParentOp());
            yieldOp->erase();
        }
        currOp = nextOp;
    }
}

mlir::scf::YieldOp replaceReturnWithYield(mlir::daphne::ReturnOp returnOp) {
    mlir::OpBuilder builder(returnOp);
    auto yieldOp = builder.create<mlir::scf::YieldOp>(returnOp.getLoc(), returnOp.getOperands());
    returnOp->erase();
    return yieldOp;
}

void rectifyEarlyReturn(mlir::scf::IfOp ifOp) {
    // FIXME: handle case where early return is in else block
    auto insertThenBlock = [&](mlir::OpBuilder &nested, mlir::Location loc) {
        auto newThenBlock = nested.getBlock();
        nested.getBlock()->getOperations().splice(nested.getBlock()->end(), ifOp.thenBlock()->getOperations());

        auto returnOps = newThenBlock->getOps<mlir::daphne::ReturnOp>();
        if (!returnOps.empty()) {
            // NOTE: we ignore operations after return, could also throw an
            // error
            removeTrailingOperationsAfterReturnOp(*returnOps.begin(), newThenBlock);
        } else {
            rectifyIfCaseWithoutReturnOp(ifOp, newThenBlock);
        }
        auto returnOp = llvm::dyn_cast<mlir::daphne::ReturnOp>(newThenBlock->back());
        if (!returnOp) {
            // this should never happen, if it does check the
            // `rectifyCaseByAppendingNecessaryOperations` function
            throw ErrorHandler::compilerError(ifOp->getLoc(), "DSLVisitor",
                                              "Final operation in then case has to be return op");
        }
        replaceReturnWithYield(returnOp);
    };
    auto insertElseBlock = [&](mlir::OpBuilder &nested, mlir::Location loc) {
        auto newElseBlock = nested.getBlock();
        if (!ifOp.getElseRegion().empty()) {
            newElseBlock->getOperations().splice(newElseBlock->end(), ifOp.elseBlock()->getOperations());
        }
        // TODO: check if already final operation is a return

        auto returnOps = newElseBlock->getOps<mlir::daphne::ReturnOp>();
        if (!returnOps.empty()) {
            // NOTE: we ignore operations after return, could also throw an
            // error
            removeTrailingOperationsAfterReturnOp(*returnOps.begin(), newElseBlock);
        } else {
            rectifyIfCaseWithoutReturnOp(ifOp, newElseBlock);
        }
        auto returnOp = llvm::dyn_cast<mlir::daphne::ReturnOp>(newElseBlock->back());
        if (!returnOp) {
            // this should never happen, if it does check the
            // `rectifyCaseByAppendingNecessaryOperations` function
            throw ErrorHandler::compilerError(ifOp->getLoc(), "DSLVisitor",
                                              "Final operation in else case has to be return op");
        }
        replaceReturnWithYield(returnOp);
    };
    mlir::OpBuilder builder(ifOp);

    auto newIfOp =
        builder.create<mlir::scf::IfOp>(builder.getUnknownLoc(), ifOp.getCondition(), insertThenBlock, insertElseBlock);
    builder.create<mlir::daphne::ReturnOp>(ifOp->getLoc(), newIfOp.getResults());
    ifOp.erase();
}

/**
 * @brief Adapts the block such that only a single return at the end of the
 * block is present, by moving early returns in SCF-Ops.
 *
 * General procedure is finding the most nested early return and then SCF-Op by
 * SCF-Op moves the return outside, putting the case without early return into
 * the other case. This is repeated until all SCF-Ops are valid and only a final
 * return exists. Might duplicate operations if we have more nested if ops like
 * this example:
 * ```
 * if (a > 5) {
 *   if (a > 10) {
 *     return SOMETHING_A;
 *   }
 *   print("a > 5");
 * }
 * else {
 *   print("a <= 5");
 * }
 * print("no early return");
 * return SOMETHING_B;
 * ```
 * would be converted to (MLIR pseudo code)
 * ```
 * return scf.if(a > 5) {
 *   yield scf.if(a > 10) {
 *     yield SOMETHING_A;
 *   } else {
 *     print("a > 5");
 *     print("no early return"); // duplicated
 *     yield SOMETHING_B; // duplicated
 *   }
 * } else {
 *   print("a <= 5");
 *   print("no early return");
 *   yield SOMETHING_B;
 * }
 * ```
 *
 * @param funcBlock The block of the function with possible early returns
 */
void rectifyEarlyReturns(mlir::Block *funcBlock) {
    if (funcBlock->empty())
        return;
    while (true) {
        size_t levelOfMostNested = 0;
        mlir::daphne::ReturnOp mostNestedReturn;
        funcBlock->walk([&](mlir::daphne::ReturnOp returnOp) {
            size_t nested = 1;
            auto op = returnOp.getOperation();
            while (op->getBlock() != funcBlock) {
                ++nested;
                op = op->getParentOp();
            }

            if (nested > levelOfMostNested) {
                mostNestedReturn = returnOp;
                levelOfMostNested = nested;
            }
        });
        if (!mostNestedReturn) {
            // no returns found -> nothing to do
            break;
        }
        if (mostNestedReturn->getBlock() == funcBlock) {
            // A top-level return with trailing ops: drop the unreachable ops and warn.
            if (mostNestedReturn != &funcBlock->back()) {
                removeTrailingOperationsAfterReturnOp(mostNestedReturn, funcBlock);
                continue;
            }
            break;
        }
        if (mostNestedReturn == &funcBlock->back())
            break;

        auto parentOp = mostNestedReturn->getParentOp();
        if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(parentOp)) {
            rectifyEarlyReturn(ifOp);
        } else {
            throw ErrorHandler::compilerError(parentOp->getLoc(), "DSLVisitor",
                                              "Early return in `" + parentOp->getName().getStringRef().str() +
                                                  "` is not supported.");
        }
    }
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionStatement(DaphneDSLGrammarParser::FunctionStatementContext *ctx) {
    auto loc = utils.getLoc(ctx->start);
    // TODO: check that the function does not shadow a builtin
    auto functionName = ctx->name->getText();
    // TODO: global variables support in functions
    auto globalSymbolTable = symbolTable;
    symbolTable = ScopedSymbolTable();

    // TODO: better check?
    if (globalSymbolTable.getNumScopes() > 1) {
        // TODO: create a function/class for throwing errors
        std::string s;
        llvm::raw_string_ostream stream(s);
        stream << loc << ": Functions can only be defined at top-level";
        throw ErrorHandler::compilerError(loc, "DSLVisitor", s.c_str());
    }

    std::vector<std::string> funcArgNames;
    std::vector<mlir::Type> funcArgTypes;
    if (ctx->args) {
        auto functionArguments = static_cast<std::vector<std::pair<std::string, mlir::Type>>>(
            (visit(ctx->args)).as<std::vector<std::pair<std::string, mlir::Type>>>());
        for (const auto &pair : functionArguments) {
            if (std::find(funcArgNames.begin(), funcArgNames.end(), pair.first) != funcArgNames.end()) {
                throw ErrorHandler::compilerError(loc, "DSLVisitor",
                                                  "Function argument name `" + pair.first + "` is used twice.");
            }
            funcArgNames.push_back(pair.first);
            funcArgTypes.push_back(pair.second);
        }
    }

    auto funcBlock = new mlir::Block();
    for (auto it : llvm::zip(funcArgNames, funcArgTypes)) {
        auto blockArg = funcBlock->addArgument(std::get<1>(it), builder.getUnknownLoc());
        handleAssignmentPart(utils.getLoc(ctx->start), std::get<0>(it), nullptr, symbolTable, blockArg);
    }

    std::vector<mlir::Type> returnTypes;
    mlir::func::FuncOp functionOperation;
    if (ctx->retTys) {
        // early creation of FuncOp for recursion
        returnTypes = visit(ctx->retTys).as<std::vector<mlir::Type>>();
        functionOperation =
            createUserDefinedFuncOp(loc, builder.getFunctionType(funcArgTypes, returnTypes), functionName);
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(funcBlock);
    visitBlockStatement(ctx->bodyStmt);

    rectifyEarlyReturns(funcBlock);

    if (funcBlock->getOperations().empty() ||
        !funcBlock->getOperations().back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder.create<mlir::daphne::ReturnOp>(utils.getLoc(ctx->stop));
    }
    auto terminator = funcBlock->getTerminator();
    auto returnOpTypes = terminator->getOperandTypes();
    if (!functionOperation) {
        // late creation if no return types defined
        functionOperation =
            createUserDefinedFuncOp(loc, builder.getFunctionType(funcArgTypes, returnOpTypes), functionName);
    } else {
        if (returnOpTypes.size() != returnTypes.size()) {
            std::stringstream s;
            s << "function `" << functionName << "` returns a different number of "
              << "values than specified in the definition (" << returnOpTypes.size() << " vs. " << returnTypes.size()
              << ')';
            throw ErrorHandler::compilerError(terminator->getLoc(), "DSLVisitor", s.str());
        }
        for (size_t i = 0; i < returnTypes.size(); i++)
            // TODO These checks should happen after type inference.
            if (!CompilerUtils::equalUnknownAware(returnOpTypes[i], returnTypes[i])) {
                std::stringstream s;
                s << "function `" << functionName << "` returns a different type for return value #" << i
                  << " than specified in the definition (" << returnOpTypes[i] << " vs. " << returnTypes[i] << ')';
                // TODO Should we use the location of the i-th argument of the
                // ReturnOp (more precise)?
                throw ErrorHandler::compilerError(terminator->getLoc(), "DSLVisitor", s.str());
            }
    }
    functionOperation.getBody().push_front(funcBlock);

    storeFunctionParamNames(functionOperation, funcArgNames);
    symbolTable = globalSymbolTable;
    return functionOperation;
}

mlir::func::FuncOp DaphneDSLVisitor::createUserDefinedFuncOp(const mlir::Location &loc,
                                                             const mlir::FunctionType &funcType,
                                                             const std::string &functionName) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *moduleBody = module.getBody();
    auto functionSymbolName = utils.getUniqueFunctionSymbol(functionName);

    builder.setInsertionPoint(moduleBody, moduleBody->begin());
    auto functionOperation = builder.create<mlir::func::FuncOp>(loc, functionSymbolName, funcType);
    functionsSymbolMap.insert({functionName, functionOperation});
    return functionOperation;
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionArgs(DaphneDSLGrammarParser::FunctionArgsContext *ctx) {
    std::vector<std::pair<std::string, mlir::Type>> functionArguments;
    for (auto funcArgCtx : ctx->functionArg()) {
        functionArguments.push_back(visitFunctionArg(funcArgCtx).as<std::pair<std::string, mlir::Type>>());
    }
    return functionArguments;
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionArg(DaphneDSLGrammarParser::FunctionArgContext *ctx) {
    auto ty = utils.unknownType;
    if (ctx->ty) {
        ty = utils.typeOrError(visitFuncTypeDef(ctx->ty));
    }
    return std::make_pair(ctx->var->getText(), ty);
}

antlrcpp::Any DaphneDSLVisitor::visitFunctionRetTypes(DaphneDSLGrammarParser::FunctionRetTypesContext *ctx) {
    std::vector<mlir::Type> retTys;
    for (auto ftdCtx : ctx->funcTypeDef())
        retTys.push_back(visitFuncTypeDef(ftdCtx).as<mlir::Type>());
    return retTys;
}

antlrcpp::Any DaphneDSLVisitor::visitFuncTypeDef(DaphneDSLGrammarParser::FuncTypeDefContext *ctx) {
    auto type = utils.unknownType;
    if (ctx->dataTy) {
        std::string dtStr = ctx->dataTy->getText();
        if (dtStr == "matrix") {
            mlir::Type vt;
            if (ctx->elTy)
                vt = utils.getValueTypeByName(ctx->elTy->getText());
            else
                vt = utils.unknownType;
            type = utils.matrixOf(vt);
        } else {
            // TODO: should we do this?
            // auto loc = utils.getLoc(ctx->start);
            // emitError(loc) << "unsupported data type for function argument: "
            // + dtStr;
            throw ErrorHandler::compilerError(utils.getLoc(ctx->start), "DSLVisitor",
                                              "unsupported data type for function argument: " + dtStr);
        }
    } else if (ctx->scalarTy)
        type = utils.getValueTypeByName(ctx->scalarTy->getText());
    return type;
}

antlrcpp::Any DaphneDSLVisitor::visitReturnStatement(DaphneDSLGrammarParser::ReturnStatementContext *ctx) {
    std::vector<mlir::Value> returns;
    for (auto expr : ctx->expr()) {
        returns.push_back(valueOrErrorOnVisit(expr));
    }
    return builder.create<mlir::daphne::ReturnOp>(utils.getLoc(ctx->start), returns);
}
