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

#include <ir/daphneir/DaphneOpsEnums.cpp.inc>

#include "mlir/Support/LogicalResult.h"
#define GET_OP_CLASSES
#include <ir/daphneir/DaphneOps.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <ir/daphneir/DaphneOpsDialect.cpp.inc>
#include <ir/daphneir/DaphneOpsTypes.cpp.inc>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
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

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>

#include <stdexcept>
#include <string>

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

mlir::Type mlir::daphne::DaphneDialect::parseType(mlir::DialectAsmParser &parser) const {
    llvm::StringRef keyword;
    mlir::ParseResult pr = parser.parseKeyword(&keyword);
    if (mlir::failed(pr))
        throw std::runtime_error("parsing a DaphneIR type failed");
    // `Matrix` `<` (`?` | \d+) `x` (`?` | \d+) `x` \type
    //      (`:` (
    //          `sp` `[` \float `]` |
    //          `rep` `[` (`dense` | `sparse`) `]`
    //      ))*
    if (keyword == "Matrix") {
        ssize_t numRows = -1;
        ssize_t numCols = -1;
        double sparsity = -1.0;
        MatrixRepresentation representation = MatrixRepresentation::Default; // default is dense
        BoolOrUnknown symmetric = BoolOrUnknown::Unknown;
        ssize_t mncSketchId = -1;
        mlir::Type elementType;
        if (parser.parseLess()) {
            return nullptr;
        }
        if (parser.parseOptionalQuestion()) {
            // Parse #rows if there was no '?'.
            if (parser.parseInteger<ssize_t>(numRows)) {
                return nullptr;
            }
        }
        if (parser.parseXInDimensionList()) {
            return nullptr;
        }
        if (parser.parseOptionalQuestion()) {
            // Parse #cols if there was no '?'.
            if (parser.parseInteger<ssize_t>(numCols)) {
                return nullptr;
            }
        }
        if (parser.parseXInDimensionList() || parser.parseType(elementType)) {
            return nullptr;
        }
        // additional properties (only print/read them when present, as this
        // will probably get more and more)
        while (succeeded(parser.parseOptionalColon())) {
            if (succeeded(parser.parseOptionalKeyword("sp"))) {
                if (sparsity != -1.0) {
                    // read sparsity twice
                    return nullptr;
                }
                if (parser.parseLSquare() || parser.parseFloat(sparsity) || parser.parseRSquare()) {
                    return nullptr;
                }
            } else if (succeeded(parser.parseOptionalKeyword("rep"))) {
                llvm::StringRef repName;
                if (parser.parseLSquare() || parser.parseKeyword(&repName) || parser.parseRSquare()) {
                    return nullptr;
                }
                representation = stringToMatrixRepresentation(repName.str());
            } else if (succeeded(parser.parseOptionalKeyword("symmetric"))) {
                llvm::StringRef symmetricStr;
                if (parser.parseLSquare() || parser.parseKeyword(&symmetricStr) || parser.parseRSquare()) {
                    return nullptr;
                }
                symmetric = stringToBoolOrUnknown(symmetricStr.str());
            } else if (succeeded(parser.parseOptionalKeyword("mncSketchId"))) {
                if (mncSketchId != -1) {
                    return nullptr;
                }
                if (parser.parseLSquare() || parser.parseInteger<ssize_t>(mncSketchId) || parser.parseRSquare()) {
                    return nullptr;
                }
            } else {
                return nullptr;
            }
        }
        if (parser.parseGreater()) {
            return nullptr;
        }

        return MatrixType::get(parser.getBuilder().getContext(), elementType, numRows, numCols, sparsity,
                               representation, symmetric, mncSketchId);
    } else if (keyword == "Frame") {
        ssize_t numRows = -1;
        ssize_t numCols = -1;
        if (parser.parseLess() || parser.parseOptionalQuestion() ||
            // TODO Parse #rows if there was no '?'.
            // parser.parseInteger<ssize_t>(numRows) ||
            parser.parseKeyword("x") || parser.parseLSquare() || parser.parseOptionalQuestion() ||
            // TODO Parse #cols if there was no '?'.
            // parser.parseInteger<ssize_t>(numCols) ||
            // TODO Parse sparsity
            parser.parseColon()) {
            return nullptr;
        }
        std::vector<mlir::Type> cts;
        mlir::Type type;
        do {
            if (parser.parseType(type))
                return nullptr;
            cts.push_back(type);
        } while (succeeded(parser.parseOptionalComma()));
        if (parser.parseRSquare() || parser.parseGreater()) {
            return nullptr;
        }
        return FrameType::get(parser.getBuilder().getContext(), cts, numRows, numCols, nullptr);
    } else if (keyword == "Handle") {
        mlir::Type dataType;
        if (parser.parseLess() || parser.parseType(dataType) || parser.parseGreater()) {
            return nullptr;
        }
        return mlir::daphne::HandleType::get(parser.getBuilder().getContext(), dataType);
    } else if (keyword == "String") {
        return StringType::get(parser.getBuilder().getContext());
    } else if (keyword == "Column") {
        if (parser.parseLess())
            return nullptr;
        ssize_t numRows = -1;
        if (parser.parseOptionalQuestion())
            // Parse #rows if there was no '?'.
            if (parser.parseInteger<ssize_t>(numRows))
                return nullptr;
        if (parser.parseXInDimensionList())
            return nullptr;
        mlir::Type vt;
        if (parser.parseType(vt))
            return nullptr;
        if (parser.parseGreater())
            return nullptr;
        return ColumnType::get(parser.getBuilder().getContext(), vt, numRows);
    } else if (keyword == "DaphneContext") {
        return mlir::daphne::DaphneContextType::get(parser.getBuilder().getContext());
    } else {
        parser.emitError(parser.getCurrentLocation()) << "Parsing failed, keyword `" << keyword << "` not recognized!";
        return nullptr;
    }
}

std::string unknownStrIf(ssize_t val) { return (val == -1) ? "?" : std::to_string(val); }

std::string unknownStrIf(double val) { return (val == -1.0) ? "?" : std::to_string(val); }

void mlir::daphne::DaphneDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &os) const {
    if (llvm::isa<mlir::daphne::StructureType>(type))
        os << "Structure";
    else if (auto t = llvm::dyn_cast<mlir::daphne::MatrixType>(type)) {
        os << "Matrix<" << unknownStrIf(t.getNumRows()) << 'x' << unknownStrIf(t.getNumCols()) << 'x'
           << t.getElementType();
        auto sparsity = t.getSparsity();
        auto representation = t.getRepresentation();
        auto symmetric = t.getSymmetric();
        auto mncSketchId = t.getMncSketchId();

        if (sparsity != -1.0) {
            os << ":sp[" << sparsity << ']';
        }
        if (representation != MatrixRepresentation::Default) {
            os << ":rep[" << matrixRepresentationToString(representation) << ']';
        }
        if (symmetric != BoolOrUnknown::Unknown) {
            os << ":symmetric[" << boolOrUnknownToString(symmetric) << ']';
        }
        if (mncSketchId != -1){
            os << ":mncSketchId[" << unknownStrIf(mncSketchId) << ']';
        }

        os << '>';
    } else if (auto t = llvm::dyn_cast<mlir::daphne::FrameType>(type)) {
        os << "Frame<" << unknownStrIf(t.getNumRows()) << "x[" << unknownStrIf(t.getNumCols()) << ": ";
        // Column types.
        std::vector<mlir::Type> cts = t.getColumnTypes();
        for (size_t i = 0; i < cts.size(); i++) {
            os << cts[i];
            if (i < cts.size() - 1)
                os << ", ";
        }
        os << "], ";
        // Column labels.
        std::vector<std::string> *labels = t.getLabels();
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
    } else if (auto t = llvm::dyn_cast<mlir::daphne::ColumnType>(type)) {
        os << "Column<" << unknownStrIf(t.getNumRows()) << "x" << t.getValueType() << '>';
    } else if (auto t = llvm::dyn_cast<mlir::daphne::ListType>(type)) {
        os << "List<" << t.getElementType() << '>';
    } else if (auto handle = llvm::dyn_cast<mlir::daphne::HandleType>(type)) {
        os << "Handle<" << handle.getDataType() << ">";
    } else if (llvm::isa<mlir::daphne::StringType>(type))
        os << "String";
    else if (auto t = llvm::dyn_cast<mlir::daphne::VariadicPackType>(type))
        os << "VariadicPack<" << t.getContainedType() << '>';
    else if (llvm::isa<mlir::daphne::DaphneContextType>(type))
        os << "DaphneContext";
    else if (llvm::isa<mlir::daphne::FileType>(type))
        os << "File";
    else if (llvm::isa<mlir::daphne::DescriptorType>(type))
        os << "Descriptor";
    else if (llvm::isa<mlir::daphne::TargetType>(type))
        os << "Target";
    else if (llvm::isa<mlir::daphne::UnknownType>(type))
        os << "Unknown";
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
                                     MatrixRepresentation representation, BoolOrUnknown symmetric, ssize_t mncSketchId)
    : elementType(elementType), numRows(numRows), numCols(numCols), sparsity(sparsity), representation(representation),
      symmetric(symmetric), mncSketchId(mncSketchId) {}

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
    if (mncSketchId != std::get<6>(tblgenKey))
        return false;
    return true;
}

// Hash key implementation
::llvm::hash_code MatrixTypeStorage::hashKey(const KeyTy &tblgenKey) {
    auto float_hashable = static_cast<ssize_t>(std::get<3>(tblgenKey) / epsilon);
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), float_hashable,
                                std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey));
}

// Construct implementation
MatrixTypeStorage *MatrixTypeStorage::construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto elementType = std::get<0>(tblgenKey);
    auto numRows = std::get<1>(tblgenKey);
    auto numCols = std::get<2>(tblgenKey);
    auto sparsity = std::get<3>(tblgenKey);
    auto representation = std::get<4>(tblgenKey);
    auto symmetric = std::get<5>(tblgenKey);
    auto mncSketchId = std::get<6>(tblgenKey);

    return new (allocator.allocate<MatrixTypeStorage>())
        MatrixTypeStorage(elementType, numRows, numCols, sparsity, representation, symmetric, mncSketchId);
}

} // namespace detail
::mlir::Type MatrixType::getElementType() const { return getImpl()->elementType; }
ssize_t MatrixType::getNumRows() const { return getImpl()->numRows; }
ssize_t MatrixType::getNumCols() const { return getImpl()->numCols; }
double MatrixType::getSparsity() const { return getImpl()->sparsity; }
MatrixRepresentation MatrixType::getRepresentation() const { return getImpl()->representation; }
BoolOrUnknown MatrixType::getSymmetric() const { return getImpl()->symmetric; }
ssize_t MatrixType::getMncSketchId() const {return getImpl()->mncSketchId; }
} // namespace mlir::daphne

::mlir::LogicalResult mlir::daphne::MatrixType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                                                       Type elementType, ssize_t numRows, ssize_t numCols,
                                                       double sparsity, MatrixRepresentation rep,
                                                       BoolOrUnknown symmetric, ssize_t mncSketchId) {
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
            symmetric != BoolOrUnknown::True || (numRows == numCols || numRows == -1 || numCols == -1))
        && (mncSketchId >= -1))
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
