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

#pragma once

// clang-format off
#include <compiler/utils/TypePrinting.h>
#include <ir/daphneir/Daphne.h>
#include <parser/metadata/MetaDataParser.h>
#include "util/ErrorHandler.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>

#include <stdexcept>
#include <string>
// clang-format on

struct CompilerUtils {

  private:
    template <typename ValT, typename AttrT>
    static std::pair<bool, ValT> isConstantHelper(mlir::Value v, const std::function<ValT(const AttrT &)> &func) {
        if (auto co = v.getDefiningOp<mlir::daphne::ConstantOp>())
            if (auto attr = llvm::dyn_cast<AttrT>(co.getValue()))
                return std::make_pair(true, func(attr));
        if (auto co = v.getDefiningOp<mlir::arith::ConstantOp>())
            if (auto attr = llvm::dyn_cast<AttrT>(co.getValue()))
                return std::make_pair(true, func(attr));
        return std::make_pair(false, ValT(0));
    }

    template <typename ValT, typename AttrT>
    static ValT constantOrThrowHelper(mlir::Value v, std::function<ValT(const AttrT &)> func,
                                      const std::string &errorMsg, const std::string &valTypeName) {
        auto p = isConstantHelper<ValT, AttrT>(v, func);
        if (p.first)
            return p.second;
        throw ErrorHandler::compilerError(
            v.getLoc(), "constantOrThrow",
            errorMsg.empty() ? ("the given value must be a constant of " + valTypeName + " type") : errorMsg);
    }

    template <typename ValT, typename AttrT>
    static ValT constantOrDefaultHelper(mlir::Value v, ValT d, std::function<ValT(const AttrT &)> func) {
        auto p = isConstantHelper<ValT, AttrT>(v, func);
        if (p.first)
            return p.second;
        return d;
    }

  public:
    /**
     * @brief The name of the attribute used to flag DaphneIR ops for vectorization if --vec-restricted is used.
     */
    static inline constexpr std::string ATTR_VEC = "vec";

    /* Optional attribute of CallKernelOp, which indicates that all results shall be combined into a single variadic
     * result.
     */
    static inline constexpr std::string_view ATTR_HAS_VARIADIC_RESULTS = "hasVariadicResults";

    /**
     * @brief Returns `true` if the given op has a boolean attribute with the given name and the value of this attribute
     * is `true`; or `false`, otherwise.
     *
     * @param op The operation to check
     * @param attr The name of the boolean attribute to check
     * @return `true` if the attribute is present and has the value `true`; or `false`, otherwise
     */
    static bool isAttrTrue(mlir::Operation *op, const std::string &attr) {
        return op->hasAttr(attr) ? llvm::dyn_cast<mlir::BoolAttr>(op->getAttr(attr)).getValue() : false;
    }

    /**
     * @brief If the given `Value` is defined by some constant operation, return
     * that constant operation; otherwise, return `nullptr`.
     *
     * @param v The `Value`.
     * @return The defining constant operation or `nullptr`.
     */
    static mlir::Operation *constantOfAnyType(mlir::Value v) {
        if (auto co = v.getDefiningOp<mlir::daphne::ConstantOp>())
            return co;
        if (auto co = v.getDefiningOp<mlir::arith::ConstantOp>())
            return co;
        return nullptr;
    }

    /**
     * @brief Returns if the given `Value` is a constant, and if so, also the
     * constant itself.
     *
     * @tparam T The C++ type of the constant to extract.
     * @param v The `Value`.
     * @return If the given value is a constant: a pair of the value `true` and
     * the constant value as type `T`; otherwise, a pair of the value `false`
     * and an unspecified value of type `T`.
     */
    template <typename T> static std::pair<bool, T> isConstant(mlir::Value v);

    /**
     * @brief Returns a constant extracted from the given `Value`, or throws an
     * exception if this is not possible.
     *
     * @tparam T The C++ type of the constant to extract.
     * @param v The `Value`.
     * @param errorMsg The message of the exception to throw. In case of an
     * empty string (default), the exception will have a generic error message.
     * @return The extracted constant as a value of type `T`, if the given value
     * is a constant.
     */
    template <typename T> static T constantOrThrow(mlir::Value v, const std::string &errorMsg = "");

    /**
     * @brief Returns a constant extracted from the given `Value`, or a default
     * value if this is not possible.
     *
     * @tparam T The C++ type of the constant to extract.
     * @param v The `Value`.
     * @param d The default value.
     * @return The extracted constant as a value of type `T`, if the given value
     * is a constant, or the given default value, otherwise.
     */
    template <typename T> static T constantOrDefault(mlir::Value v, T d);

    [[maybe_unused]] static FileMetaData getFileMetaData(mlir::Value filename);

    /**
     * @brief Produces a string containing the C++ type name of the
     * corresponding MLIR type. Mainly used to generate function names for
     * generated kernel libraries. This function is defined recursively to also
     * print the value types of templated containers (e.g., DenseMatrix<float>).
     * A pragma is added to silence clang-tidy which might complain about
     * recursion.
     *
     * @param t MLIR type name
     * @param angleBrackets If `true` (default), angle brackets are used for C++
     * template types (e.g., `DenseMatrix<float>`); Otherwise, underscores are
     * used (e.g., `DenseMatrix_float`).
     * @param generalizeToStructure If `true`, `Structure` is used instead of
     * derived types like `DenseMatrix` etc.
     * @return A string representation of the C++ type names
     */
    // TODO The parameter generalizeToStructure seems to be used only by some
    // remaining kernel name generation in LowerToLLVMPass. Once those
    // call-sites have been refactored to use the kernel catalog, this feature
    // can be removed here.
    static std::string mlirTypeToCppTypeName(mlir::Type t, bool angleBrackets = true,
                                             bool generalizeToStructure = false) { // NOLINT(misc-no-recursion)
        if (t.isF64())
            return "double";
        if (t.isF32())
            return "float";
        if (t.isSignedInteger(8))
            return "int8_t";
        if (t.isSignedInteger(32))
            return "int32_t";
        if (t.isSignedInteger(64))
            return "int64_t";
        if (t.isUnsignedInteger(8))
            return "uint8_t";
        if (t.isUnsignedInteger(32))
            return "uint32_t";
        if (t.isUnsignedInteger(64))
            return "uint64_t";
        if (t.isSignlessInteger(1))
            return "bool";
        if (t.isIndex())
            return "size_t";
        if (llvm::isa<mlir::daphne::StructureType>(t))
            return "Structure";
        if (auto matTy = llvm::dyn_cast<mlir::daphne::MatrixType>(t)) {
            if (generalizeToStructure)
                return "Structure";
            // For matrices of strings we use `std::string` as the value type, while for string scalars we use
            // `const char *` as the value type. Thus, we need this special case here. Maybe we can do it without a
            // special case in the future.
            std::string vtName;
            if (llvm::isa<mlir::daphne::StringType>(matTy.getElementType()))
                vtName = "std::string";
            else if (llvm::isa<mlir::daphne::FixedStr16Type>(matTy.getElementType()))
                vtName = "FixedStr16";
            else
                vtName = mlirTypeToCppTypeName(matTy.getElementType(), angleBrackets, false);
            switch (matTy.getRepresentation()) {
            case mlir::daphne::MatrixRepresentation::Dense:
                return angleBrackets ? ("DenseMatrix<" + vtName + ">") : ("DenseMatrix_" + vtName);
            case mlir::daphne::MatrixRepresentation::Sparse:
                return angleBrackets ? ("CSRMatrix<" + vtName + ">") : ("CSRMatrix_" + vtName);
            }
        } else if (llvm::isa<mlir::daphne::FrameType>(t)) {
            if (generalizeToStructure)
                return "Structure";
            return "Frame";
        } else if (auto colTy = llvm::dyn_cast<mlir::daphne::ColumnType>(t)) {
            if (generalizeToStructure)
                return "Structure";
            // For columns of strings we use `std::string` as the value type, while for string scalars we use
            // `const char *` as the value type. Thus, we need this special case here. Maybe we can do it without a
            // special case in the future.
            std::string vtName;
            if (llvm::isa<mlir::daphne::StringType>((colTy.getValueType())))
                vtName = "std::string";
            else if (llvm::isa<mlir::daphne::FixedStr16Type>(colTy.getValueType()))
                vtName = "FixedStr16";
            else
                vtName = mlirTypeToCppTypeName(colTy.getValueType(), angleBrackets, false);
            return angleBrackets ? ("Column<" + vtName + ">") : ("Column_" + vtName);
        } else if (auto lstTy = llvm::dyn_cast<mlir::daphne::ListType>(t)) {
            if (generalizeToStructure)
                return "Structure";
            const std::string dtName = mlirTypeToCppTypeName(lstTy.getElementType(), angleBrackets, false);
            return angleBrackets ? ("List<" + dtName + ">") : ("List_" + dtName);
        } else if (llvm::isa<mlir::daphne::StringType>(t))
            // This becomes "const char *" (which makes perfect sense for
            // strings) when inserted into the typical "const DT *" template of
            // kernel input parameters.
            return "char";
        else if (llvm::isa<mlir::daphne::FixedStr16Type>(t))
            return "FixedStr16";
        else if (llvm::isa<mlir::daphne::DaphneContextType>(t))
            return "DaphneContext";
        else if (auto handleTy = llvm::dyn_cast<mlir::daphne::HandleType>(t)) {
            const std::string tName =
                mlirTypeToCppTypeName(handleTy.getDataType(), angleBrackets, generalizeToStructure);
            return angleBrackets ? ("Handle<" + tName + ">") : ("Handle_" + tName);
        } else if (llvm::isa<mlir::daphne::FileType>(t))
            return "File";
        else if (llvm::isa<mlir::daphne::DescriptorType>(t))
            return "Descriptor";
        else if (llvm::isa<mlir::daphne::TargetType>(t))
            return "Target";
        else if (auto memRefType = llvm::dyn_cast<mlir::MemRefType>(t)) {
            const std::string vtName = mlirTypeToCppTypeName(memRefType.getElementType(), angleBrackets, false);
            const std::string rankStr = std::to_string(memRefType.getRank());
            return angleBrackets ? ("StridedMemRefType<" + vtName + "," + rankStr + ">")
                                 : ("StridedMemRefType_" + vtName + "_" + rankStr);
        }

        std::string typeName;
        llvm::raw_string_ostream rsos(typeName);
        t.print(rsos);
        throw std::runtime_error("no C++ type name known for the given MLIR type: " + typeName);
    }

    static bool isMatrixComputation(mlir::Operation *v);

    /**
     * @brief Returns the DAPHNE context used in the given function.
     *
     * Throws if there is not exactly one DAPHNE context.
     *
     * @param func
     * @return
     */
    [[maybe_unused]] mlir::Value static getDaphneContext(mlir::func::FuncOp &func) {
        mlir::Value dctx = nullptr;
        auto ops = func.getBody().front().getOps<mlir::daphne::CreateDaphneContextOp>();
        for (auto op : ops) {
            if (!dctx)
                dctx = op.getResult();
            else
                throw ErrorHandler::compilerError(op.getLoc(), "getDaphneContext",
                                                  "function body block contains more than one "
                                                  "CreateDaphneContextOp");
        }
        if (!dctx)
            throw ErrorHandler::compilerError(func.getLoc(), "getDaphneContext",
                                              "function body block contains no CreateDaphneContextOp");
        return dctx;
    }

    /**
     * @brief Returns `true` if the given type is a DAPHNE data object type; or `false`, otherwise.
     */
    [[maybe_unused]] static bool isObjType(mlir::Type t) {
        return llvm::isa<mlir::daphne::MatrixType, mlir::daphne::FrameType, mlir::daphne::ColumnType,
                         mlir::daphne::ListType>(t);
    }

    /**
     * @brief Returns `true` if the given value is a DAPHNE data object; or `false`, otherwise.
     */
    [[maybe_unused]] static bool hasObjType(mlir::Value v) { return isObjType(v.getType()); }

    /**
     * @brief Returns `true` if the given type is a DAPHNE matrix that has exactly one column; or `false`, otherwise.
     */
    static bool isMatTypeWithSingleCol(mlir::Type t) {
        auto mt = llvm::dyn_cast<mlir::daphne::MatrixType>(t);
        return mt && mt.getNumCols() == 1;
    }

    /**
     * @brief Returns `true` if the given type is a DAPHNE scalar type (or: value type); or `false`, otherwise.
     */
    static bool isScaType(mlir::Type t) {
        return
            // floating-point types
            t.isF64() || t.isF32() ||
            // signed integer types
            t.isSignedInteger(64) || t.isSignedInteger(32) || t.isSignedInteger(8) ||
            // unsigned integer types
            t.isUnsignedInteger(64) || t.isUnsignedInteger(32) || t.isUnsignedInteger(8) ||
            // index type
            t.isIndex() ||
            // boolean type
            t.isSignlessInteger(1) ||
            // string type
            llvm::isa<mlir::daphne::StringType>(t);
    }

    /**
     * @brief Returns `true` if the given value is a DAPHNE scalar; or `false`, otherwise.
     */
    static bool hasScaType(mlir::Value v) { return isScaType(v.getType()); }

    /**
     * @brief Returns the value type of the given type.
     *
     * - For the unknown type, the unknown type is returned.
     * - For matrices, the value type is extracted.
     * - For frames, an error is thrown.
     * - For columns, the value type is extracted.
     * - For lists, this function is called recursively on the element type.
     * - For scalars, the type itself is returned.
     * - For anything else, an error is thrown.
     *
     * @param t the given type
     * @return the value type of the given type
     */
    static mlir::Type getValueType(mlir::Type t) {
        if (llvm::isa<mlir::daphne::UnknownType>(t))
            return t;
        if (auto mt = llvm::dyn_cast<mlir::daphne::MatrixType>(t))
            return mt.getElementType();
        if (auto ft = llvm::dyn_cast<mlir::daphne::FrameType>(t))
            throw std::runtime_error(
                "getValueType() doesn't support frames yet"); // TODO maybe use the most general value type
        if (auto ct = llvm::dyn_cast<mlir::daphne::ColumnType>(t))
            return ct.getValueType();
        if (auto lt = llvm::dyn_cast<mlir::daphne::ListType>(t))
            return getValueType(lt.getElementType());
        if (isScaType(t))
            return t;

        std::stringstream s;
        s << "getValueType(): the given type is neither a supported data type nor a supported value type: `" << t
          << '`';
        throw std::runtime_error(s.str());
    }

    /**
     * @brief Returns the value types of the given type as a sequence.
     *
     * - For the unknown type, the unknown type is returned (single-element sequence).
     * - For matrices, the value type is extracted (single-element sequence).
     * - For frames, the sequence of column types is returned.
     * - For columns, the value type is extracted (single-element sequence).
     * - For lists, this function is called recursively on the element type.
     * - For scalars, the type itself is returned (single-element sequence).
     * - For anything else, an error is thrown.
     *
     * @param t the given type
     * @return the value types of the given type as a sequence
     */
    static std::vector<mlir::Type> getValueTypes(mlir::Type t) {
        if (llvm::isa<mlir::daphne::UnknownType>(t))
            return {t};
        if (auto mt = llvm::dyn_cast<mlir::daphne::MatrixType>(t))
            return {mt.getElementType()};
        if (auto ft = llvm::dyn_cast<mlir::daphne::FrameType>(t))
            return ft.getColumnTypes();
        if (auto ct = llvm::dyn_cast<mlir::daphne::ColumnType>(t))
            return {ct.getValueType()};
        if (auto lt = llvm::dyn_cast<mlir::daphne::ListType>(t))
            return getValueTypes(lt.getElementType());
        if (isScaType(t))
            return {t};

        std::stringstream s;
        s << "getValueTypes(): the given type is neither a supported data type nor a supported value type: `" << t
          << '`';
        throw std::runtime_error(s.str());
    }

    /**
     * @brief Sets the value type of the given type to the
     * given value type and returns this derived type.
     *
     * - For the unknown type, an error is thrown.
     * - For matrices, the value type is set to the given value type.
     * - For frames, an error is thrown.
     * - For columns, the value type is set to the given value type.
     * - For lists, this function is called recursively on the element type.
     * - For scalars, the given value type is returned.
     * - For anything else, an error is thrown.
     *
     * @param t the type whose value type shall be set
     * @param vt the value type to use
     * @return the derived type
     */
    static mlir::Type setValueType(mlir::Type t, mlir::Type vt) {
        if (llvm::isa<mlir::daphne::UnknownType>(t))
            throw std::runtime_error("setValueType(): cannot set the set the value type of an unknown type");
        if (auto mt = llvm::dyn_cast<mlir::daphne::MatrixType>(t))
            return mt.withElementType(vt);
        if (auto ft = llvm::dyn_cast<mlir::daphne::FrameType>(t))
            throw std::runtime_error("setValueType() doesn't support frames yet"); // TODO
        if (auto ct = llvm::dyn_cast<mlir::daphne::ColumnType>(t))
            return ct.withValueType(vt);
        if (auto lt = llvm::dyn_cast<mlir::daphne::ListType>(t))
            return setValueType(lt.getElementType(), vt);
        if (isScaType(t))
            return vt;

        std::stringstream s;
        s << "setValueType(): the given type is neither a supported data type nor a supported value type: `" << t
          << '`';
        throw std::runtime_error(s.str());
    }

    /**
     * @brief Checks if the two given types are the same, whereby
     * DaphneIR's unknown type acts as a wildcard.
     *
     * The two types are considered equal, iff they are exactly the same
     * type, or one of the following "excuses" holds:
     * - at least one of the types is unknown
     * - both types are matrices and at least one of them has an unknown
     *   value type
     *
     * @param t1 The first type
     * @param t2 The second type
     * @result `true` if the two types are considered equal, `false` otherwise
     */
    static bool equalUnknownAware(mlir::Type t1, mlir::Type t2) {
        using mlir::daphne::UnknownType;
        auto matT1 = llvm::dyn_cast<mlir::daphne::MatrixType>(t1);
        auto matT2 = llvm::dyn_cast<mlir::daphne::MatrixType>(t2);
        // The two given types are considered equal, iff:
        return (
            // The two types are exactly the same...
            t1 == t2
            // ...or one of the following "excuses" holds:
            ||
            (
                // at least one of the types is unknown
                llvm::isa<UnknownType>(t1) || llvm::isa<UnknownType>(t2) ||
                // both types are matrices and at least one of them
                // has an unknown value type
                (matT1 && matT2 &&
                 (llvm::isa<UnknownType>(matT1.getElementType()) || llvm::isa<UnknownType>(matT2.getElementType())))));
    }

    /**
     * @brief Infers and sets the result type of the given operation and returns the result as an `mlir::Value`.
     *
     * Works only for operations with exactly one result. For operations with more than one result, use
     * `retValsWithInferredTypes()`.
     */
    template <class Op> static mlir::Value retValWithInferredType(Op op) {
        mlir::daphne::setInferredTypes(op.getOperation());
        return static_cast<mlir::Value>(op);
    }

    /**
     * @brief Infers and sets the result types of the given operation and returns the results as an `mlir::ResultRange`.
     *
     * Works for operations with any number of results. For operations with exactly one result, using
     * `retValWithInferredType()` can be more convenient.
     */
    template <class Op> static mlir::ResultRange retValsWithInferredTypes(Op op) {
        mlir::daphne::setInferredTypes(op.getOperation());
        return op.getResults();
    }
};
