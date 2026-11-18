/*
 * Copyright 2024 The DAPHNE Consortium
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

#include "mlir/IR/OpDefinition.h"
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/DaphneOps.cpp.inc>
#include <util/ErrorHandler.h>

mlir::Attribute performCast(mlir::Attribute attr, mlir::Type targetType, mlir::Location loc) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
        auto apInt = intAttr.getValue();

        if (auto outTy = llvm::dyn_cast<mlir::IntegerType>(targetType)) {
            // Extend or truncate the integer value based on the target type
            if (outTy.isUnsignedInteger()) {
                apInt = apInt.zextOrTrunc(outTy.getWidth());
            } else if (outTy.isSignedInteger()) {
                apInt = (intAttr.getType().isSignedInteger()) ? apInt.sextOrTrunc(outTy.getWidth())
                                                              : apInt.zextOrTrunc(outTy.getWidth());
            }
            return mlir::IntegerAttr::getChecked(loc, outTy, apInt);
        }

        if (auto outTy = llvm::dyn_cast<mlir::IndexType>(targetType)) {
            return mlir::IntegerAttr::getChecked(loc, outTy, apInt);
        }

        if (targetType.isF64()) {
            if (intAttr.getType().isSignedInteger()) {
                return mlir::FloatAttr::getChecked(loc, targetType, llvm::APIntOps::RoundSignedAPIntToDouble(apInt));
            }
            if (intAttr.getType().isUnsignedInteger() || intAttr.getType().isIndex()) {
                return mlir::FloatAttr::getChecked(loc, targetType, llvm::APIntOps::RoundAPIntToDouble(apInt));
            }
        }

        if (targetType.isF32()) {
            if (intAttr.getType().isSignedInteger()) {
                return mlir::FloatAttr::getChecked(loc, targetType, llvm::APIntOps::RoundSignedAPIntToFloat(apInt));
            }
            if (intAttr.getType().isUnsignedInteger()) {
                return mlir::FloatAttr::get(targetType, llvm::APIntOps::RoundAPIntToFloat(apInt));
            }
        }
    } else if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
        auto val = floatAttr.getValueAsDouble();

        if (targetType.isF64()) {
            return mlir::FloatAttr::getChecked(loc, targetType, val);
        }
        if (targetType.isF32()) {
            return mlir::FloatAttr::getChecked(loc, targetType, static_cast<float>(val));
        }
        if (targetType.isIntOrIndex()) {
            auto num = static_cast<int64_t>(val);
            return mlir::IntegerAttr::getChecked(loc, targetType, num);
        }
    }

    // If casting is not possible, return the original attribute
    return {};
}

template <class AttrElementT, class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = std::function<ElementValueT(const ElementValueT &)>>
mlir::Attribute constFoldUnaryOp(mlir::Location loc, mlir::Type resultType, llvm::ArrayRef<mlir::Attribute> operands,
                                 const CalculationT &calculate) {
    if (operands.size() != 1)
        throw ErrorHandler::compilerError(loc, "CanonicalizerPass (constFoldUnaryOp)",
                                          "unary op takes one operand but " + std::to_string(operands.size()) +
                                              " were given");

    if (!operands[0])
        return {};

    if (llvm::isa<AttrElementT>(operands[0])) {
        auto operand = mlir::cast<AttrElementT>(operands[0]);

        return AttrElementT::get(resultType, calculate(operand.getValue()));
    }
    return {};
}

template <class ArgAttrElementT, class ResAttrElementT = ArgAttrElementT,
          class ArgElementValueT = typename ArgAttrElementT::ValueType,
          class ResElementValueT = typename ResAttrElementT::ValueType,
          class CalculationT = std::function<ResElementValueT(const ArgElementValueT &, const ArgElementValueT &)>>
mlir::Attribute constFoldBinaryOp(mlir::Location loc, mlir::Type resultType, llvm::ArrayRef<mlir::Attribute> operands,
                                  const CalculationT &calculate) {
    if (operands.size() != 2)
        throw ErrorHandler::compilerError(loc, "CanonicalizerPass (constFoldBinaryOp)",
                                          "binary op takes two operands but " + std::to_string(operands.size()) +
                                              " were given");

    if (!operands[0] || !operands[1])
        return {};

    if (llvm::isa<ArgAttrElementT>(operands[0]) && llvm::isa<ArgAttrElementT>(operands[1])) {
        auto lhs = mlir::cast<ArgAttrElementT>(operands[0]);
        auto rhs = mlir::cast<ArgAttrElementT>(operands[1]);

        // We need dedicated cases, as the parameters of ResAttrElementT::get()
        // depend on ResAttrElementT.
        if constexpr (std::is_same<ResAttrElementT, mlir::IntegerAttr>::value ||
                      std::is_same<ResAttrElementT, mlir::FloatAttr>::value) {
            mlir::Type l = lhs.getType();
            mlir::Type r = rhs.getType();
            if ((llvm::dyn_cast<mlir::IntegerType>(l) || llvm::dyn_cast<mlir::FloatType>(l)) &&
                (llvm::dyn_cast<mlir::IntegerType>(r) || llvm::dyn_cast<mlir::FloatType>(r))) {
                auto lhsBitWidth = lhs.getType().getIntOrFloatBitWidth();
                auto rhsBitWidth = rhs.getType().getIntOrFloatBitWidth();

                if (lhsBitWidth < rhsBitWidth) {
                    mlir::Attribute promotedLhs = performCast(lhs, rhs.getType(), loc);
                    lhs = mlir::cast<ArgAttrElementT>(promotedLhs);
                } else if (rhsBitWidth < lhsBitWidth) {
                    mlir::Attribute promotedRhs = performCast(rhs, lhs.getType(), loc);
                    rhs = mlir::cast<ArgAttrElementT>(promotedRhs);
                }
            }
            return ResAttrElementT::get(resultType, calculate(lhs.getValue(), rhs.getValue()));
        } else if constexpr (std::is_same<ResAttrElementT, mlir::BoolAttr>::value) {
            if (!resultType.isSignlessInteger(1))
                throw ErrorHandler::compilerError(loc, "CanonicalizerPass (constFoldBinaryOp)",
                                                  "expected boolean result type");
            return ResAttrElementT::get(lhs.getContext(), calculate(lhs.getValue(), rhs.getValue()));
        } else if constexpr (std::is_same<ResAttrElementT, mlir::StringAttr>::value) {
            if (!llvm::isa<mlir::daphne::StringType>(resultType))
                throw ErrorHandler::compilerError(loc, "CanonicalizerPass (constFoldBinaryOp)",
                                                  "expected string result type");
            return ResAttrElementT::get(calculate(lhs.getValue(), rhs.getValue()), resultType);
        }
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

mlir::OpFoldResult mlir::daphne::CastOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();

    if (isTrivialCast()) {
        if (!operands.empty() && operands[0])
            return {operands[0]};
        else
            return {getArg()};
    }

    if (!operands.empty() && operands[0]) {
        if (auto castedAttr = performCast(operands[0], getType(), getLoc())) {
            return castedAttr;
        }
    }

    return {};
}

mlir::OpFoldResult mlir::daphne::EwAddOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APFloat> mlir::daphne::EwAddOp::foldScalarFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a + b;
}

// Two's-complement addition produces the same bit pattern for signed and
// unsigned inputs, so both methods share a body. The split exists so the
// driver's signedness dispatch finds a handler for either result type.
std::optional<llvm::APInt> mlir::daphne::EwAddOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a + b;
}

std::optional<llvm::APInt> mlir::daphne::EwAddOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a + b;
}

mlir::OpFoldResult mlir::daphne::EwSubOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APFloat> mlir::daphne::EwSubOp::foldScalarFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a - b;
}

std::optional<llvm::APInt> mlir::daphne::EwSubOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a - b;
}

std::optional<llvm::APInt> mlir::daphne::EwSubOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a - b;
}

mlir::OpFoldResult mlir::daphne::EwMulOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APFloat> mlir::daphne::EwMulOp::foldScalarFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a * b;
}

std::optional<llvm::APInt> mlir::daphne::EwMulOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a * b;
}

std::optional<llvm::APInt> mlir::daphne::EwMulOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a * b;
}

mlir::OpFoldResult mlir::daphne::EwDivOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APFloat> mlir::daphne::EwDivOp::foldScalarFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a / b;
}

std::optional<llvm::APInt> mlir::daphne::EwDivOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    if (b == 0)
        throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwDivOp::fold)",
                                          "Can't divide by 0");
    return a.sdiv(b);
}

std::optional<llvm::APInt> mlir::daphne::EwDivOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    if (b == 0)
        throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwDivOp::fold)",
                                          "Can't divide by 0");
    return a.udiv(b);
}

mlir::OpFoldResult mlir::daphne::EwMinusOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APInt> mlir::daphne::EwMinusOp::foldScalarUnaryInt(const llvm::APInt &a) { return -a; }

std::optional<llvm::APFloat> mlir::daphne::EwMinusOp::foldScalarUnaryFloat(const llvm::APFloat &a) { return -a; }

mlir::OpFoldResult mlir::daphne::EwPowOp::fold(FoldAdaptor adaptor) {
    // TODO: EwPowOp integer constant folding
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APFloat> mlir::daphne::EwPowOp::foldScalarFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    // Compute in `double` because APFloat has no direct `pow`; convert the
    // result back into an APFloat matching the operands' semantics so the
    // driver can hand it to FloatAttr::getChecked without a semantics
    // mismatch.
    double r = std::pow(a.convertToDouble(), b.convertToDouble());
    llvm::APFloat out(r);
    bool losesInfo = false;
    out.convert(a.getSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return out;
}

mlir::OpFoldResult mlir::daphne::EwModOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APInt> mlir::daphne::EwModOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    if (b == 0)
        throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwModOp::fold)",
                                          "Can't compute mod 0");
    return a.srem(b);
}

std::optional<llvm::APInt> mlir::daphne::EwModOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    if (b == 0)
        throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwModOp::fold)",
                                          "Can't compute mod 0");
    return a.urem(b);
}

mlir::OpFoldResult mlir::daphne::EwLogOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APFloat> mlir::daphne::EwLogOp::foldScalarFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    // Element-wise log_b(a) via change-of-base on doubles, because APFloat has
    // no direct logarithm. Convert back into the operands' semantics for the
    // FloatAttr the driver will build.
    double r = std::log(a.convertToDouble()) / std::log(b.convertToDouble());
    llvm::APFloat out(r);
    bool losesInfo = false;
    out.convert(a.getSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return out;
}

mlir::OpFoldResult mlir::daphne::EwMinOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APFloat> mlir::daphne::EwMinOp::foldScalarFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return llvm::minimum(a, b);
}

std::optional<llvm::APInt> mlir::daphne::EwMinOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.slt(b) ? a : b;
}

std::optional<llvm::APInt> mlir::daphne::EwMinOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.ult(b) ? a : b;
}

mlir::OpFoldResult mlir::daphne::EwMaxOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<llvm::APFloat> mlir::daphne::EwMaxOp::foldScalarFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return llvm::maximum(a, b);
}

std::optional<llvm::APInt> mlir::daphne::EwMaxOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.sgt(b) ? a : b;
}

std::optional<llvm::APInt> mlir::daphne::EwMaxOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.ugt(b) ? a : b;
}

mlir::OpFoldResult mlir::daphne::EwAndOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<bool> mlir::daphne::EwAndOp::foldScalarBool(bool a, bool b) { return a && b; }

std::optional<llvm::APInt> mlir::daphne::EwAndOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    // TODO: should output bool
    unsigned w = getType().getIntOrFloatBitWidth();
    return llvm::APInt(w, (a != 0) && (b != 0));
}

std::optional<llvm::APInt> mlir::daphne::EwAndOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    unsigned w = getType().getIntOrFloatBitWidth();
    return llvm::APInt(w, (a != 0) && (b != 0));
}

mlir::OpFoldResult mlir::daphne::EwBitwiseAndOp::fold(FoldAdaptor adaptor) { return {}; }

mlir::OpFoldResult mlir::daphne::EwOrOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<bool> mlir::daphne::EwOrOp::foldScalarBool(bool a, bool b) { return a || b; }

std::optional<llvm::APInt> mlir::daphne::EwOrOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    // TODO: should output bool
    unsigned w = getType().getIntOrFloatBitWidth();
    return llvm::APInt(w, (a != 0) || (b != 0));
}

std::optional<llvm::APInt> mlir::daphne::EwOrOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    unsigned w = getType().getIntOrFloatBitWidth();
    return llvm::APInt(w, (a != 0) || (b != 0));
}

mlir::OpFoldResult mlir::daphne::EwXorOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<bool> mlir::daphne::EwXorOp::foldScalarBool(bool a, bool b) { return a ^ b; }

std::optional<llvm::APInt> mlir::daphne::EwXorOp::foldScalarSInt(const llvm::APInt &a, const llvm::APInt &b) {
    // TODO: should output bool
    unsigned w = getType().getIntOrFloatBitWidth();
    return llvm::APInt(w, (a != 0) ^ (b != 0));
}

std::optional<llvm::APInt> mlir::daphne::EwXorOp::foldScalarUInt(const llvm::APInt &a, const llvm::APInt &b) {
    unsigned w = getType().getIntOrFloatBitWidth();
    return llvm::APInt(w, (a != 0) ^ (b != 0));
}

mlir::OpFoldResult mlir::daphne::EwConcatOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();

    if (operands.size() != 2)
        throw ErrorHandler::compilerError(this->getLoc(), "CanonicalizerPass (mlir::daphne::EwConcatOp::fold)",
                                          "binary op takes two operands but " + std::to_string(operands.size()) +
                                              " were given");

    if (!operands[0] || !operands[1])
        return {};

    if (llvm::isa<StringAttr>(operands[0]) && isa<StringAttr>(operands[1])) {
        auto lhs = mlir::cast<StringAttr>(operands[0]);
        auto rhs = mlir::cast<StringAttr>(operands[1]);

        auto concated = lhs.getValue().str() + rhs.getValue().str();
        return StringAttr::get(concated, getType());
    }
    return {};
}

mlir::OpFoldResult mlir::daphne::EwEqOp::fold(FoldAdaptor adaptor) {
    // Try the shared numeric driver first; string equality is not part of
    // the ScalarConstantFoldable vocabulary, so fall back to the bespoke
    // StringAttr path if the driver returns nothing.
    if (auto res = foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc()))
        return res;
    auto strOp = [](const llvm::StringRef &a, const llvm::StringRef &b) { return a == b; };
    if (auto res = constFoldBinaryOp<StringAttr, IntegerAttr>(
            getLoc(), IntegerType::get(getContext(), 64, IntegerType::SignednessSemantics::Signed),
            adaptor.getOperands(), strOp))
        return res;
    return {};
}

// The legacy Eq folder was signedness-agnostic on integer inputs, and also
// folded bool pairs (a BoolAttr is an i1 IntegerAttr). The driver dispatches by
// kind, so we spell out signed, unsigned, and bool with the same body.
std::optional<bool> mlir::daphne::EwEqOp::foldScalarCmpFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a == b;
}

std::optional<bool> mlir::daphne::EwEqOp::foldScalarCmpSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a == b;
}

std::optional<bool> mlir::daphne::EwEqOp::foldScalarCmpUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a == b;
}

std::optional<bool> mlir::daphne::EwEqOp::foldScalarCmpBool(bool a, bool b) { return a == b; }

mlir::OpFoldResult mlir::daphne::EwNeqOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<bool> mlir::daphne::EwNeqOp::foldScalarCmpFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a != b;
}

std::optional<bool> mlir::daphne::EwNeqOp::foldScalarCmpSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a != b;
}

std::optional<bool> mlir::daphne::EwNeqOp::foldScalarCmpUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a != b;
}

std::optional<bool> mlir::daphne::EwNeqOp::foldScalarCmpBool(bool a, bool b) { return a != b; }

mlir::OpFoldResult mlir::daphne::EwLtOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<bool> mlir::daphne::EwLtOp::foldScalarCmpFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a < b;
}

std::optional<bool> mlir::daphne::EwLtOp::foldScalarCmpSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.slt(b);
}

std::optional<bool> mlir::daphne::EwLtOp::foldScalarCmpUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.ult(b);
}

mlir::OpFoldResult mlir::daphne::EwLeOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<bool> mlir::daphne::EwLeOp::foldScalarCmpFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a <= b;
}

std::optional<bool> mlir::daphne::EwLeOp::foldScalarCmpSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.sle(b);
}

std::optional<bool> mlir::daphne::EwLeOp::foldScalarCmpUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.ule(b);
}

mlir::OpFoldResult mlir::daphne::EwGtOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<bool> mlir::daphne::EwGtOp::foldScalarCmpFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a > b;
}

std::optional<bool> mlir::daphne::EwGtOp::foldScalarCmpSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.sgt(b);
}

std::optional<bool> mlir::daphne::EwGtOp::foldScalarCmpUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.ugt(b);
}

mlir::OpFoldResult mlir::daphne::EwGeOp::fold(FoldAdaptor adaptor) {
    return foldScalarOp(*this, adaptor.getOperands(), getType(), getLoc());
}

std::optional<bool> mlir::daphne::EwGeOp::foldScalarCmpFloat(const llvm::APFloat &a, const llvm::APFloat &b) {
    return a >= b;
}

std::optional<bool> mlir::daphne::EwGeOp::foldScalarCmpSInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.sge(b);
}

std::optional<bool> mlir::daphne::EwGeOp::foldScalarCmpUInt(const llvm::APInt &a, const llvm::APInt &b) {
    return a.uge(b);
}

// Shape queries fold to a single index constant whenever the operand's type
// carries a known extent

mlir::OpFoldResult mlir::daphne::NumRowsOp::fold(FoldAdaptor) {
    ssize_t numRows = -1;
    mlir::Type inTy = getArg().getType();
    if (auto t = llvm::dyn_cast<mlir::daphne::MatrixType>(inTy))
        numRows = t.getNumRows();
    else if (auto t = llvm::dyn_cast<mlir::daphne::FrameType>(inTy))
        numRows = t.getNumRows();
    if (numRows == -1)
        return {};
    return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), numRows);
}

mlir::OpFoldResult mlir::daphne::SparsityOp::fold(FoldAdaptor) {
    if (auto t = llvm::dyn_cast<mlir::daphne::MatrixType>(getArg().getType())) {
        double sparsity = t.getSparsity();
        if (sparsity != -1.0)
            return mlir::FloatAttr::get(mlir::Float64Type::get(getContext()), sparsity);
    }
    return {};
}
