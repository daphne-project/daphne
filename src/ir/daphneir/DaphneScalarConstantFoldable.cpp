/*
 * Copyright 2026 The DAPHNE Consortium
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

#include "ir/daphneir/DaphneScalarConstantFoldable.h"

#include "ir/daphneir/DaphneInferTypesOpInterface.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::daphne {
#include <ir/daphneir/DaphneScalarConstantFoldable.cpp.inc>
} // namespace mlir::daphne

using namespace mlir;

namespace {

// -------------------------------------------------------------------------
// Attribute-kind probes
// -------------------------------------------------------------------------
//
// Kept as free helpers so the driver reads as a sequence of guarded attempts
// rather than a nested if-else pyramid.

bool isFloatAttrPair(Attribute a, Attribute b) { return llvm::isa<FloatAttr>(a) && llvm::isa<FloatAttr>(b); }

bool isIntAttrPair(Attribute a, Attribute b) { return llvm::isa<IntegerAttr>(a) && llvm::isa<IntegerAttr>(b); }

bool isBoolAttrPair(Attribute a, Attribute b) { return llvm::isa<BoolAttr>(a) && llvm::isa<BoolAttr>(b); }

// The legacy templates promoted the narrower of two integer/float operands to
// the width of the wider one. We recreate that step here so op lambdas never
// have to worry about mixed-width inputs.
//
// Returns a (possibly new) attribute pair whose values are ready to feed into
// the op's fold method; the caller must cast back to the concrete Attr kind.
template <typename AttrKind>
std::pair<AttrKind, AttrKind> promoteWidth(AttrKind lhs, AttrKind rhs, Location loc);

template <> std::pair<FloatAttr, FloatAttr> promoteWidth(FloatAttr lhs, FloatAttr rhs, Location loc) {
    // Match the legacy templates: promote the narrower float operand to the
    // wider one's semantics before the op lambda runs. Without this an op
    // lambda computing e.g. `a + b` would feed APFloat::operator+ two values
    // with mismatched semantics (UB / debug assert). This can happen because
    // --canonicalize runs before type inference has unified operand widths.
    const llvm::fltSemantics &lSem = lhs.getValue().getSemantics();
    const llvm::fltSemantics &rSem = rhs.getValue().getSemantics();
    if (&lSem == &rSem)
        return {lhs, rhs};
    unsigned lw = llvm::APFloat::semanticsSizeInBits(lSem);
    unsigned rw = llvm::APFloat::semanticsSizeInBits(rSem);
    bool losesInfo = false;
    if (lw < rw) {
        llvm::APFloat ext = lhs.getValue();
        ext.convert(rSem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        return {llvm::cast<FloatAttr>(FloatAttr::getChecked(loc, rhs.getType(), ext)), rhs};
    }
    llvm::APFloat ext = rhs.getValue();
    ext.convert(lSem, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return {lhs, llvm::cast<FloatAttr>(FloatAttr::getChecked(loc, lhs.getType(), ext))};
}

template <> std::pair<IntegerAttr, IntegerAttr> promoteWidth(IntegerAttr lhs, IntegerAttr rhs, Location loc) {
    Type lTy = lhs.getType();
    Type rTy = rhs.getType();
    if (!llvm::isa<IntegerType>(lTy) || !llvm::isa<IntegerType>(rTy))
        return {lhs, rhs};
    unsigned lw = lTy.getIntOrFloatBitWidth();
    unsigned rw = rTy.getIntOrFloatBitWidth();
    if (lw == rw)
        return {lhs, rhs};
    // Extend by the source operand's signedness, like the legacy performCast:
    // unsigned zero-extends, signed sign-extends. Going by the wider type would
    // turn a high-bit unsigned value negative when widening into a signed type.
    if (lw < rw) {
        auto ext = llvm::cast<IntegerType>(lTy).isUnsigned() ? lhs.getValue().zextOrTrunc(rw)
                                                             : lhs.getValue().sextOrTrunc(rw);
        return {IntegerAttr::getChecked(loc, rTy, ext), rhs};
    }
    auto ext =
        llvm::cast<IntegerType>(rTy).isUnsigned() ? rhs.getValue().zextOrTrunc(lw) : rhs.getValue().sextOrTrunc(lw);
    return {lhs, IntegerAttr::getChecked(loc, lTy, ext)};
}

} // namespace

namespace mlir::daphne {

// -------------------------------------------------------------------------
// Unary driver
// -------------------------------------------------------------------------

static Attribute foldScalarUnary(ScalarConstantFoldable op, Attribute operand, Type resultType, Location loc) {
    if (!operand)
        return {};

    if (auto f = llvm::dyn_cast<FloatAttr>(operand))
        if (auto v = op.foldScalarUnaryFloat(f.getValue()))
            return FloatAttr::getChecked(loc, resultType, *v);

    if (auto i = llvm::dyn_cast<IntegerAttr>(operand))
        if (auto v = op.foldScalarUnaryInt(i.getValue()))
            return IntegerAttr::getChecked(loc, resultType, *v);

    return {};
}

// -------------------------------------------------------------------------
// Binary driver: arithmetic (numeric in, numeric out)
// -------------------------------------------------------------------------

static Attribute foldScalarArithBinary(ScalarConstantFoldable op, Attribute lhs, Attribute rhs, Type resultType,
                                       Location loc) {
    if (isFloatAttrPair(lhs, rhs)) {
        auto [l, r] = promoteWidth(llvm::cast<FloatAttr>(lhs), llvm::cast<FloatAttr>(rhs), loc);
        if (auto v = op.foldScalarFloat(l.getValue(), r.getValue()))
            return FloatAttr::getChecked(loc, resultType, *v);
    }
    // BoolAttr checked before IntegerAttr: MLIR's BoolAttr is-a IntegerAttr of
    // type i1, so `isa<IntegerAttr>` matches bool pairs too. Dispatch to the
    // bool method first to preserve the legacy fold order.
    if (isBoolAttrPair(lhs, rhs)) {
        auto l = llvm::cast<BoolAttr>(lhs).getValue();
        auto r = llvm::cast<BoolAttr>(rhs).getValue();
        if (auto v = op.foldScalarBool(l, r))
            return BoolAttr::get(op->getContext(), *v);
    }
    if (isIntAttrPair(lhs, rhs)) {
        auto [l, r] = promoteWidth(llvm::cast<IntegerAttr>(lhs), llvm::cast<IntegerAttr>(rhs), loc);
        // Precedence matches the legacy signedness dispatch in Fold.cpp.
        if (auto intTy = llvm::dyn_cast<IntegerType>(resultType); intTy && intTy.isSigned()) {
            if (auto v = op.foldScalarSInt(l.getValue(), r.getValue()))
                return IntegerAttr::getChecked(loc, resultType, *v);
        } else if (auto intTy = llvm::dyn_cast<IntegerType>(resultType); intTy && intTy.isUnsigned()) {
            if (auto v = op.foldScalarUInt(l.getValue(), r.getValue()))
                return IntegerAttr::getChecked(loc, resultType, *v);
        }
        // Signless / Index intentionally not folded, matching the legacy
        // Fold.cpp templates, which key off isSignedInteger()/isUnsignedInteger().
    }
    return {};
}

// -------------------------------------------------------------------------
// Binary driver: comparison (numeric in, bool out)
// -------------------------------------------------------------------------

static Attribute foldScalarCmpBinary(ScalarConstantFoldable op, Attribute lhs, Attribute rhs, Type resultType,
                                     Location loc) {
    // A comparison yields a truth value, but DAPHNE represents it in the op's
    // (numeric) result type rather than i1, matching the legacy Fold.cpp
    // comparison folders, which materialised the bool into `getType()`. So a
    // float-typed comparison result is a FloatAttr holding 0.0/1.0, and an
    // integer-typed one is an IntegerAttr holding 0/1.
    auto materialize = [&](bool v) -> Attribute {
        if (auto floatTy = llvm::dyn_cast<FloatType>(resultType)) {
            // Build the 0.0/1.0 truth value directly in the result type's
            // float semantics. A bare `APFloat(double)` is always IEEEdouble,
            // which `FloatAttr::verify` rejects for a narrower result type
            // (e.g. f32), silently dropping the fold.
            llvm::APFloat truth(floatTy.getFloatSemantics(), v ? "1.0" : "0.0");
            return FloatAttr::getChecked(loc, resultType, truth);
        }
        return IntegerAttr::getChecked(loc, resultType, v ? 1 : 0);
    };

    if (isFloatAttrPair(lhs, rhs)) {
        auto [l, r] = promoteWidth(llvm::cast<FloatAttr>(lhs), llvm::cast<FloatAttr>(rhs), loc);
        if (auto v = op.foldScalarCmpFloat(l.getValue(), r.getValue()))
            return materialize(*v);
    }
    // Bool checked before int: a BoolAttr is an i1 IntegerAttr, so isIntAttrPair
    // would match it, but i1 is signless and would fall through the signed /
    // unsigned branches below. Legacy Eq / Neq folded bool pairs, so route them
    // here first.
    if (isBoolAttrPair(lhs, rhs)) {
        auto l = llvm::cast<BoolAttr>(lhs).getValue();
        auto r = llvm::cast<BoolAttr>(rhs).getValue();
        if (auto v = op.foldScalarCmpBool(l, r))
            return materialize(*v);
    }
    if (isIntAttrPair(lhs, rhs)) {
        auto [l, r] = promoteWidth(llvm::cast<IntegerAttr>(lhs), llvm::cast<IntegerAttr>(rhs), loc);
        Type argTy = l.getType();
        // Comparison signedness follows the input type, unlike arithmetic
        // where the result type carries it. Signless / index inputs are not
        // folded, matching the legacy Fold.cpp templates which keyed off
        // isSignedInteger() / isUnsignedInteger() explicitly.
        if (auto intTy = llvm::dyn_cast<IntegerType>(argTy); intTy && intTy.isUnsigned()) {
            if (auto v = op.foldScalarCmpUInt(l.getValue(), r.getValue()))
                return materialize(*v);
        } else if (auto intTy = llvm::dyn_cast<IntegerType>(argTy); intTy && intTy.isSigned()) {
            if (auto v = op.foldScalarCmpSInt(l.getValue(), r.getValue()))
                return materialize(*v);
        }
    }
    return {};
}

// -------------------------------------------------------------------------
// Entry point
// -------------------------------------------------------------------------

Attribute foldScalarOp(ScalarConstantFoldable op, ArrayRef<Attribute> operands, Type resultType, Location loc) {
    if (operands.empty())
        return {};

    if (operands.size() == 1)
        return foldScalarUnary(op, operands[0], resultType, loc);

    if (operands.size() != 2)
        return {}; // Ternary and higher not modelled by this interface.

    if (!operands[0] || !operands[1])
        return {};

    // Comparison ops (numeric in, truth value out) carry the ValueTypeCmp
    // trait; every other binary op is arithmetic (numeric in, numeric out).
    // Dispatch on the trait rather than the result type: ValueTypeCmp gives a
    // comparison the most-general argument value type (e.g. si64/f64), never
    // i1, so an i1-result heuristic would never route comparisons here.
    const bool isCmp = op->hasTrait<OpTrait::ValueTypeCmp>();
    return isCmp ? foldScalarCmpBinary(op, operands[0], operands[1], resultType, loc)
                 : foldScalarArithBinary(op, operands[0], operands[1], resultType, loc);
}

} // namespace mlir::daphne
