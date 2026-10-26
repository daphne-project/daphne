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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/DaphneTypeStorage.h>
#include <util/ErrorHandler.h>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
// #include "mlir/IR/FunctionImplementation.h" // Removed in newer LLVM
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <ir/daphneir/DaphneOpsEnums.cpp.inc>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>

#include <stdexcept>
#include <string>

#define GET_OP_CLASSES
#include <ir/daphneir/DaphneOps.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <ir/daphneir/DaphneOpsDialect.cpp.inc>
#include <ir/daphneir/DaphneOpsTypes.cpp.inc>

struct DaphneInlinerInterface : public mlir::DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable, bool wouldBeCloned) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation *, mlir::Region *, bool, mlir::IRMapping &) const final { return true; }

    bool isLegalToInline(mlir::Region *, mlir::Region *, bool, mlir::IRMapping &) const final { return true; }

    void handleTerminator(mlir::Operation *op, mlir::ValueRange valuesToRepl) const override {
        auto returnOp = mlir::dyn_cast<mlir::daphne::ReturnOp>(op);

        // Replace the values directly with the return operands.
        if (returnOp.getNumOperands() != valuesToRepl.size()) {
            throw ErrorHandler::compilerError(op, "DaphneInlinerInterface (handleTerminator)",
                                              "number of operands " + std::to_string(returnOp.getNumOperands()) +
                                                  " from " + op->getName().getStringRef().str() +
                                                  " do not match size " + std::to_string(valuesToRepl.size()));
        }

        for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
            // Need to use const_cast because valuesToRepl is const
            auto t = valuesToRepl[it.index()];
            const_cast<mlir::Value &>(t).replaceAllUsesWith(it.value());
        }
    }

    mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input, mlir::Type resultType,
                                               mlir::Location conversionLoc) const final {
        return builder.create<mlir::daphne::CastOp>(conversionLoc, resultType, input);
    }
};

void mlir::daphne::DaphneDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <ir/daphneir/DaphneOps.cpp.inc>
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include <ir/daphneir/DaphneOpsTypes.cpp.inc>
        >();
    addInterfaces<DaphneInlinerInterface>();
}

mlir::Operation *mlir::daphne::DaphneDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                                                  mlir::Location loc) {
    return builder.create<mlir::daphne::ConstantOp>(loc, type, value);
}

namespace {
mlir::LogicalResult parseOptionalDim(mlir::AsmParser &parser, ssize_t &value) {
    if (succeeded(parser.parseOptionalQuestion()))
        return mlir::success();
    return parser.parseInteger<ssize_t>(value);
}
} // namespace

std::string unknownStrIf(ssize_t val) { return (val == -1) ? "?" : std::to_string(val); }

std::string unknownStrIf(double val) { return (val == -1.0) ? "?" : std::to_string(val); }

mlir::Type mlir::daphne::MatrixType::parse(mlir::AsmParser &parser) {
    ssize_t numRows = -1;
    ssize_t numCols = -1;
    double sparsity = -1.0;
    MatrixRepresentation representation = MatrixRepresentation::Default;
    BoolOrUnknown symmetric = BoolOrUnknown::Unknown;
    mlir::Type elementType;

    if (parser.parseLess() || failed(parseOptionalDim(parser, numRows)) || parser.parseXInDimensionList() ||
        failed(parseOptionalDim(parser, numCols)) || parser.parseXInDimensionList() || parser.parseType(elementType))
        return mlir::Type();

    while (succeeded(parser.parseOptionalColon())) {
        if (succeeded(parser.parseOptionalKeyword("sp"))) {
            if (parser.parseLSquare() || parser.parseFloat(sparsity) || parser.parseRSquare())
                return mlir::Type();
        } else if (succeeded(parser.parseOptionalKeyword("rep"))) {
            llvm::StringRef repName;
            if (parser.parseLSquare() || parser.parseKeyword(&repName) || parser.parseRSquare())
                return mlir::Type();
            representation = stringToMatrixRepresentation(repName.str());
        } else if (succeeded(parser.parseOptionalKeyword("symmetric"))) {
            llvm::StringRef symmetricStr;
            if (parser.parseLSquare() || parser.parseKeyword(&symmetricStr) || parser.parseRSquare())
                return mlir::Type();
            symmetric = stringToBoolOrUnknown(symmetricStr.str());
        } else {
            return mlir::Type();
        }
    }

    if (parser.parseGreater())
        return mlir::Type();

    return MatrixType::get(parser.getBuilder().getContext(), elementType, numRows, numCols, sparsity, representation,
                           symmetric);
}

void mlir::daphne::MatrixType::print(mlir::AsmPrinter &os) const {
    os << "Matrix<" << unknownStrIf(getNumRows()) << 'x' << unknownStrIf(getNumCols()) << 'x' << getElementType();
    if (auto sparsity = getSparsity(); sparsity != -1.0)
        os << ":sp[" << sparsity << ']';
    if (auto representation = getRepresentation(); representation != MatrixRepresentation::Default)
        os << ":rep[" << matrixRepresentationToString(representation) << ']';
    if (auto symmetric = getSymmetric(); symmetric != BoolOrUnknown::Unknown)
        os << ":symmetric[" << boolOrUnknownToString(symmetric) << ']';
    os << '>';
}

mlir::Type mlir::daphne::FrameType::parse(mlir::AsmParser &parser) {
    ssize_t numRows = -1;
    ssize_t numCols = -1;
    llvm::SmallVector<mlir::Type> columnTypes;

    if (parser.parseLess() || failed(parseOptionalDim(parser, numRows)) || parser.parseKeyword("x") ||
        parser.parseLSquare() || failed(parseOptionalDim(parser, numCols)) || parser.parseColon())
        return mlir::Type();

    if (failed(parser.parseOptionalRSquare())) {
        mlir::Type type;
        do {
            if (parser.parseType(type))
                return mlir::Type();
            columnTypes.push_back(type);
        } while (succeeded(parser.parseOptionalComma()));
        if (parser.parseRSquare())
            return mlir::Type();
    }

    if (parser.parseComma())
        return mlir::Type();

    std::vector<std::string> *labels = nullptr;
    if (failed(parser.parseOptionalQuestion())) {
        std::string label;
        std::vector<std::string> parsedLabels;
        if (parser.parseLSquare())
            return mlir::Type();
        do {
            if (parser.parseString(&label))
                return mlir::Type();
            parsedLabels.push_back(label);
        } while (succeeded(parser.parseOptionalComma()));
        if (parser.parseRSquare())
            return mlir::Type();
        // Allocate labels to keep them alive for the lifetime of the context.
        labels = new std::vector<std::string>(parsedLabels);
    }

    if (parser.parseGreater())
        return mlir::Type();

    return FrameType::get(parser.getBuilder().getContext(),
                          std::vector<mlir::Type>(columnTypes.begin(), columnTypes.end()), numRows, numCols, labels);
}

void mlir::daphne::FrameType::print(mlir::AsmPrinter &os) const {
    os << "Frame<" << unknownStrIf(getNumRows()) << "x[" << unknownStrIf(getNumCols()) << ": ";
    std::vector<mlir::Type> cts = getColumnTypes();
    for (size_t i = 0; i < cts.size(); i++) {
        os << cts[i];
        if (i < cts.size() - 1)
            os << ", ";
    }
    os << "], ";
    std::vector<std::string> *labels = getLabels();
    if (labels) {
        os << '[';
        for (size_t i = 0; i < labels->size(); i++) {
            os << '"' << (*labels)[i] << '"';
            if (i < labels->size() - 1)
                os << ", ";
        }
        os << ']';
    } else
        os << '?';
    os << '>';
}

mlir::Type mlir::daphne::ColumnType::parse(mlir::AsmParser &parser) {
    ssize_t numRows = -1;
    mlir::Type valueType;

    if (parser.parseLess() || failed(parseOptionalDim(parser, numRows)) || parser.parseXInDimensionList() ||
        parser.parseType(valueType) || parser.parseGreater())
        return mlir::Type();

    return ColumnType::get(parser.getBuilder().getContext(), valueType, numRows);
}

void mlir::daphne::ColumnType::print(mlir::AsmPrinter &os) const {
    os << "Column<" << unknownStrIf(getNumRows()) << "x" << getValueType() << '>';
}

mlir::Type mlir::daphne::ListType::parse(mlir::AsmParser &parser) {
    mlir::Type elementType;
    if (parser.parseLess() || parser.parseType(elementType) || parser.parseGreater())
        return mlir::Type();
    return ListType::get(parser.getBuilder().getContext(), elementType);
}

void mlir::daphne::ListType::print(mlir::AsmPrinter &os) const { os << "List<" << getElementType() << '>'; }

mlir::Type mlir::daphne::HandleType::parse(mlir::AsmParser &parser) {
    mlir::Type dataType;
    if (parser.parseLess() || parser.parseType(dataType) || parser.parseGreater())
        return mlir::Type();
    return HandleType::get(parser.getBuilder().getContext(), dataType);
}

void mlir::daphne::HandleType::print(mlir::AsmPrinter &os) const { os << "Handle<" << getDataType() << ">"; }

mlir::Type mlir::daphne::VariadicPackType::parse(mlir::AsmParser &parser) {
    mlir::Type containedType;
    if (parser.parseLess() || parser.parseType(containedType) || parser.parseGreater())
        return mlir::Type();
    return VariadicPackType::get(parser.getBuilder().getContext(), containedType);
}

void mlir::daphne::VariadicPackType::print(mlir::AsmPrinter &os) const {
    os << "VariadicPack<" << getContainedType() << '>';
}

std::string mlir::daphne::matrixRepresentationToString(MatrixRepresentation rep) {
    switch (rep) {
    case MatrixRepresentation::Dense:
        return "dense";
    case MatrixRepresentation::Sparse:
        return "sparse";
    default:
        throw std::runtime_error("unknown mlir::daphne::MatrixRepresentation " + std::to_string(static_cast<int>(rep)));
    }
}

std::ostream &mlir::daphne::operator<<(std::ostream &os, MatrixRepresentation rep) {
    return os << matrixRepresentationToString(rep);
}

mlir::daphne::MatrixRepresentation mlir::daphne::stringToMatrixRepresentation(const std::string &str) {
    if (str == "dense")
        return MatrixRepresentation::Dense;
    else if (str == "sparse")
        return MatrixRepresentation::Sparse;
    else
        throw std::runtime_error("No matrix representation equals the string `" + str + "`");
}

std::string mlir::daphne::boolOrUnknownToString(BoolOrUnknown b) {
    switch (b) {
    case BoolOrUnknown::Unknown:
        return "?";
    case BoolOrUnknown::False:
        return "false";
    case BoolOrUnknown::True:
        return "true";
    default:
        throw std::runtime_error("unknown BoolOrUnknown " + std::to_string(static_cast<int>(b)));
    }
}

BoolOrUnknown mlir::daphne::stringToBoolOrUnknown(const std::string &str) {
    if (str == "?")
        return BoolOrUnknown::Unknown;
    else if (str == "false")
        return BoolOrUnknown::False;
    else if (str == "true")
        return BoolOrUnknown::True;
    else
        throw std::runtime_error("no BoolOrUnknown equals the string `" + str + "`");
}

namespace mlir::daphne {
namespace detail {

// Constructor implementation
MatrixTypeStorage::MatrixTypeStorage(::mlir::Type elementType, ssize_t numRows, ssize_t numCols, double sparsity,
                                     MatrixRepresentation representation, BoolOrUnknown symmetric)
    : elementType(elementType), numRows(numRows), numCols(numCols), sparsity(sparsity), representation(representation),
      symmetric(symmetric) {}

// Equality operator implementation
bool MatrixTypeStorage::operator==(const KeyTy &tblgenKey) const {
    if (!(elementType == std::get<0>(tblgenKey)))
        return false;
    if (numRows != std::get<1>(tblgenKey))
        return false;
    if (numCols != std::get<2>(tblgenKey))
        return false;
    if (std::fabs(sparsity - std::get<3>(tblgenKey)) >= epsilon)
        return false;
    if (representation != std::get<4>(tblgenKey))
        return false;
    if (symmetric != std::get<5>(tblgenKey))
        return false;
    return true;
}

// Hash key implementation
::llvm::hash_code MatrixTypeStorage::hashKey(const KeyTy &tblgenKey) {
    auto float_hashable = static_cast<ssize_t>(std::get<3>(tblgenKey) / epsilon);
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), float_hashable,
                                std::get<4>(tblgenKey), std::get<5>(tblgenKey));
}

// Construct implementation
MatrixTypeStorage *MatrixTypeStorage::construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    auto numRows = std::get<1>(tblgenKey);
    auto numCols = std::get<2>(tblgenKey);
    auto sparsity = std::get<3>(tblgenKey);
    auto representation = std::get<4>(tblgenKey);
    auto symmetric = std::get<5>(tblgenKey);

    return new (allocator.allocate<MatrixTypeStorage>())
        MatrixTypeStorage(elementType, numRows, numCols, sparsity, representation, symmetric);
}

} // namespace detail
::mlir::Type MatrixType::getElementType() const { return getImpl()->elementType; }
ssize_t MatrixType::getNumRows() const { return getImpl()->numRows; }
ssize_t MatrixType::getNumCols() const { return getImpl()->numCols; }
double MatrixType::getSparsity() const { return getImpl()->sparsity; }
MatrixRepresentation MatrixType::getRepresentation() const { return getImpl()->representation; }
BoolOrUnknown MatrixType::getSymmetric() const { return getImpl()->symmetric; }
} // namespace mlir::daphne

::mlir::LogicalResult mlir::daphne::MatrixType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                       Type elementType, ssize_t numRows, ssize_t numCols,
                                                       double sparsity, MatrixRepresentation rep,
                                                       BoolOrUnknown symmetric) {
    if ((
            // Value type is unknown.
            llvm::isa<mlir::daphne::UnknownType>(elementType)
            // Value type is known.
            || elementType.isSignedInteger(64) || elementType.isUnsignedInteger(8) ||
            elementType.isUnsignedInteger(64) || elementType.isF32() || elementType.isF64() || elementType.isIndex() ||
            elementType.isInteger(1) || llvm::isa<mlir::daphne::StringType>(elementType) ||
            elementType.isUnsignedInteger(64) || elementType.isUnsignedInteger(32) || elementType.isSignedInteger(32) ||
            elementType.isSignedInteger(8)) &&
        (
            // Number of rows and columns are valid (-1 for unknown).
            numRows >= -1 && numCols >= -1) &&
        (sparsity == -1 || (sparsity >= 0.0 && sparsity <= 1.0)) &&
        (
            // "symmetric is true" must imply that the matrix is square or the shape is unknown.
            symmetric != BoolOrUnknown::True || (numRows == numCols || numRows == -1 || numCols == -1)))
        return mlir::success();
    else
        return emitError() << "invalid matrix element type: " << elementType;
}

::mlir::LogicalResult mlir::daphne::FrameType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                      std::vector<Type> columnTypes, ssize_t numRows, ssize_t numCols,
                                                      std::vector<std::string> *labels) {
    // TODO Verify the individual column types.
    if (numRows < -1 || numCols < -1)
        return mlir::failure();
    if (numCols != -1) {
        // ToDo: ExtractColOp does not provide these columnTypes
        if (!columnTypes.empty()) {
            if (static_cast<ssize_t>(columnTypes.size()) != numCols)
                return mlir::failure();
            if (labels && static_cast<ssize_t>(labels->size()) != numCols)
                return mlir::failure();
        }
    }
    if (labels && labels->size() != columnTypes.size())
        return mlir::failure();
    return mlir::success();
}

::mlir::LogicalResult mlir::daphne::HandleType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                       Type dataType) {
    if (llvm::isa<MatrixType>(dataType)) {
        return mlir::success();
    } else
        return emitError() << "only matrix type is supported for handle atm, got: " << dataType;
}

::mlir::LogicalResult mlir::daphne::ColumnType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                       Type valueType, ssize_t numRows) {
    if (!CompilerUtils::isScaType(valueType) && !llvm::isa<mlir::daphne::UnknownType>(valueType))
        return mlir::failure();
    if (numRows < -1)
        return mlir::failure();
    return mlir::success();
}
