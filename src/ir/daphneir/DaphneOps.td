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

#ifndef SRC_IR_DAPHNEIR_DAPHNEOPS_TD
#define SRC_IR_DAPHNEIR_DAPHNEOPS_TD

// TODO Get rid of those ScalarType params.

include "ir/daphneir/DaphneDialect.td"
include "ir/daphneir/DaphneTypes.td"
include "ir/daphneir/DaphneDistributableOpInterface.td"
include "ir/daphneir/DaphneInferFrameLabelsOpInterface.td"
include "ir/daphneir/DaphneInferSymmetricOpInterface.td"
include "ir/daphneir/DaphneInferShapeOpInterface.td"
include "ir/daphneir/DaphneInferShapeTraits.td"
include "ir/daphneir/DaphneInferSparsityOpInterface.td"
include "ir/daphneir/DaphneInferSparsityTraits.td"
include "ir/daphneir/DaphneInferMncSketchIdOpInterface.td"
include "ir/daphneir/DaphneInferTypesOpInterface.td"
include "ir/daphneir/DaphneInferTypesTraits.td"
include "ir/daphneir/DaphneVectorizableOpInterface.td"
include "ir/daphneir/DaphneTypeAdaptationTraits.td"
include "ir/daphneir/CUDASupport.td"
include "ir/daphneir/FPGAOPENCLSupport.td"

include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// ****************************************************************************
// Custom constraints
// ****************************************************************************

// The type of `mat` is a matrix whose value type is the type of `sca`.
class TypeIsMatrixOf<string mat, string sca> : And<[
    TypeIs<mat, Matrix>.predicate,
    TypesMatchWith<"", mat, sca, "llvm::dyn_cast<::mlir::daphne::MatrixType>($_self).getElementType()">.predicate
]>;

// The types of `a` and `b` must either be the same (e.g. both are matrices of
// the same value type, or both are scalars of the same type), or the type of
// `a` must be a matrix whose value type is the type of `b`, or vice-versa.
class TypesMatchOrOneIsMatrixOfOther<string a, string b> : PredOpTrait<
    a # " and " # b # " must either be of the same type, or " # a # " must be a matrix whose value type is the type of " # b # ", or vice-versa",
    Or<[
        TypesMatchWith<"", a, b, "$_self">.predicate,
        TypeIsMatrixOf<a, b>,
        TypeIsMatrixOf<b, a>
    ]>
>;

// ****************************************************************************
// Base Daphne operation definition
// ****************************************************************************

class Daphne_Op<string mnemonic, list<Trait> traits = []> :
        Op<Daphne_Dialect, mnemonic, traits>;

// ****************************************************************************
// DAPHNE Runtime Interoperability
// ****************************************************************************

def Daphne_ConvertSparseBuffersToMatrix
  : Daphne_Op<"convertSparseBuffersToMatrix", [Pure]> {
  let summary = "Wrap CSR buffers into a daphne.Matrix.";
  let description = [{ Constructs a Matrix from CSR layout buffers (values, column indices, row offsets) given as raw
    base pointer, offset, size, and stride metadata, plus the logical matrix sizes. }];
  let arguments = (ins Size:$valuesBase, Size:$valuesOffset, Size:$valuesSize, Size:$valuesStride,
                        Size:$colIdxsBase, Size:$colIdxsOffset, Size:$colIdxsSize, Size:$colIdxsStride,
                        Size:$rowOffsetsBase, Size:$rowOffsetsOffset, Size:$rowOffsetsSize, Size:$rowOffsetsStride,
                        Size:$numRows, Size:$numCols);
  let results   = (outs MatrixOrU:$res);
}

def Daphne_ConvertMemRefToDenseMatrix : Daphne_Op<"convertMemRefToDenseMatrix"> {
    let summary = "Return a DenseMatrix.";
    let description = [{ Constructs a DenseMatrix given a rank 2 StridedMemRefType. }];

    /* let arguments = (ins AnyMemRef:$arg); */
    let hasCanonicalizeMethod = 1;
    let arguments = (ins Size:$base, Size:$offset, Size:$size0, Size:$size1, Size:$stride0, Size:$stride1);
    let results = (outs MatrixOrU:$res);
}

def Daphne_ConvertDenseMatrixToMemRef : Daphne_Op<"convertDenseMatrixToMemRef", [Pure]> {
    let summary = "Given a DenseMatrix, return a StridedMemRefType.";
    let description = [{ Constructs a StridedMemRefType with rank 2 from a DenseMatrix* with already allocated memory. }];
    let hasCanonicalizeMethod = 1;
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs AnyMemRef:$output);
}

def Daphne_ConvertCSRMatrixToValuesMemRef : Daphne_Op<"convertCSRMatrixToValuesMemRef", [Pure]> {
    let summary = "Given a CSRMatrix, return a StridedMemRefType.";
    let description = [{ Constructs a StridedMemRefType with rank 1 containing the values of a CSRMatrix* with already allocated memory. }];
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs AnyMemRef:$resValues);
}

def Daphne_ConvertCSRMatrixToColIdxsMemRef : Daphne_Op<"convertCSRMatrixToColIdxsMemRef", [Pure]> {
    let summary = "Given a CSRMatrix, return a StridedMemRefType.";
    let description = [{ Constructs a StridedMemRefType with rank 1 containing the column indices of a CSRMatrix* with already allocated memory. }];
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs AnyMemRef:$resColIdxs);
}

def Daphne_ConvertCSRMatrixToRowOffsetsMemRef : Daphne_Op<"convertCSRMatrixToRowOffsetsMemRef", [Pure]> {
    let summary = "Given a CSRMatrix, return a StridedMemRefType.";
    let description = [{ Constructs a StridedMemRefType with rank 1 containing the row offsets of a CSRMatrix* with already allocated memory. }];
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs AnyMemRef:$resRowOffsets);
}

// ****************************************************************************
// Data generation
// ****************************************************************************

def Daphne_FillOp : Daphne_Op<"fill", [
    DataTypeMat, ValueTypeFromFirstArg,
    NumRowsFromIthScalar<1>, NumColsFromIthScalar<2>, CUDASupport,
    DeclareOpInterfaceMethods<InferSymmetricOpInterface>,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>
]> {
    let arguments = (ins AnyScalar:$arg, Size:$numRows, Size:$numCols);
    let results = (outs MatrixOrU:$res);
}

def Daphne_MatrixConstantOp : Daphne_Op<"matrixConstant", [
    DataTypeMat,
    DeclareOpInterfaceMethods<InferShapeOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>
]>{
    let arguments = (ins UI64:$matrixAddr);
    let results = (outs MatrixOrU:$res);
}

def Daphne_CreateFrameOp : Daphne_Op<"createFrame", [
    SameVariadicOperandSize,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>,
    Pure
]> {
    let arguments = (ins Variadic<MatrixOrU>:$cols, Variadic<StrScalar>:$labels);
    let results = (outs FrameOrU:$res);
}

def Daphne_DiagMatrixOp : Daphne_Op<"diagMatrix", [
    TypeFromFirstArg,
    NumRowsFromArg, NumColsFromArgNumRows,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>
]> {
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);
}

def Daphne_RandMatrixOp : Daphne_Op<"randMatrix", [
    TypesMatchWith<"min and max arguments must be of the same type", "min", "max", "$_self">,
    NumRowsFromIthScalar<0>, NumColsFromIthScalar<1>, DeclareOpInterfaceMethods<InferTypesOpInterface>,
    SparsityFromIthScalar<4>, CastArgsToResTypeRandMatrixOp, DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>,
]> {
    let arguments = (ins Size:$numRows, Size:$numCols, AnyScalar:$min, AnyScalar:$max, F64:$sparsity, IntScalar:$seed);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SampleOp : Daphne_Op<"sample", [
    DataTypeMat, ValueTypeFromFirstArg,
    NumRowsFromIthScalar<1>, OneCol
]> {
    let arguments = (ins AnyScalar:$range, Size:$size, BoolScalar:$withReplacement, Seed:$seed);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SeqOp : Daphne_Op<"seq", [
    DataTypeMat, ValueTypeFromArgs,
    OneCol, DeclareOpInterfaceMethods<InferNumRowsOpInterface>,
    CompletelyDense,
    CastArgsToResType,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>
]> {
    let arguments = (ins NumScalar:$from, NumScalar:$to, NumScalar:$inc);
    let results = (outs MatrixOf<[NumScalar]>);
}

// ****************************************************************************
// Matrix/frame meta data
// ****************************************************************************

def Daphne_TypeOfOp : Daphne_Op<"typeOf"> {
    let arguments = (ins AnyTypeOf<[AnyScalar, MatrixOrFrame]>:$arg);
    let results = (outs StrScalar:$res);
}

class Daphne_NumOp<string name, list<Trait> traits = []> : Daphne_Op<name, !listconcat(traits, [
    DataTypeSca, ValueTypeSize, Pure
])> {
    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs Size:$res);

    let hasCanonicalizeMethod = 1;
}

def Daphne_NumRowsOp : Daphne_NumOp<"numRows">;
def Daphne_NumColsOp : Daphne_NumOp<"numCols">;
def Daphne_NumCellsOp : Daphne_NumOp<"numCells">;

def Daphne_SparsityOp : Daphne_Op<"sparsity", [DataTypeSca]> {
  let arguments = (ins MatrixOf<[AnyScalar]>:$arg);
  let results = (outs FloatScalar:$res);

  let hasCanonicalizeMethod = 1;
}

def Daphne_IsSymmetricOp : Daphne_Op<"isSymmetric"> {
    let summary = "Checks if a matrix is symmetric";
    let description = [{
        This operation checks if the input matrix is symmetric.
    }];
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs BoolScalar:$res);
}

// ****************************************************************************
// Matrix multiplication
// ****************************************************************************

def Daphne_MatMulOp : Daphne_Op<"matMul", [
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DataTypeMat, ValueTypeFromArgs,
    DeclareOpInterfaceMethods<InferShapeOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>, 
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>, CUDASupport, FPGAOPENCLSupport,
    CastFirstTwoArgsToResType, NoMemoryEffect
]> {
    let arguments = (ins MatrixOf<[NumScalar]>:$lhs, MatrixOf<[NumScalar]>:$rhs, BoolScalar:$transa, BoolScalar:$transb);
    let results = (outs MatrixOf<[NumScalar]>:$res);
    let hasCanonicalizeMethod = 1;
}

// ****************************************************************************
// Elementwise unary
// ****************************************************************************

class Daphne_EwUnaryOp<string name, Type scalarType, list<Trait> traits = []> : Daphne_Op<name, !listconcat(traits, [
    DataTypeFromFirstArg,
    ShapeFromArg,
    CastArgsToResType,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NoMemoryEffect
])> {
    let arguments = (ins AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$arg);
    let results = (outs AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$res);
}

// ----------------------------------------------------------------------------
// Arithmetic/general math
// ----------------------------------------------------------------------------

// TODO EwMinusOp: Should an unsigned integer argument yield a signed integer result?
def Daphne_EwMinusOp : Daphne_EwUnaryOp<"ewMinus", NumScalar, [ValueTypeFromFirstArg]> {
   let hasFolder = 1;
   let hasCanonicalizeMethod = 1;
}
def Daphne_EwAbsOp : Daphne_EwUnaryOp<"ewAbs", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_EwSignOp : Daphne_EwUnaryOp<"ewSign", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_EwExpOp : Daphne_EwUnaryOp<"ewExp", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwLnOp : Daphne_EwUnaryOp<"ewLn", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwSqrtOp : Daphne_EwUnaryOp<"ewSqrt", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_EwNegOp : Daphne_EwUnaryOp<"ewNeg", NumScalar, [ValueTypeFromFirstArg]>;

// ----------------------------------------------------------------------------
// Rounding
// ----------------------------------------------------------------------------

def Daphne_EwRoundOp : Daphne_EwUnaryOp<"ewRound", NumScalar, [ValueTypeFromFirstArg]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwFloorOp : Daphne_EwUnaryOp<"ewFloor", NumScalar, [ValueTypeFromFirstArg]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwCeilOp : Daphne_EwUnaryOp<"ewCeil", NumScalar, [ValueTypeFromFirstArg]> {
    let hasCanonicalizeMethod = 1;
}

// ----------------------------------------------------------------------------
// Trigonometric
// ----------------------------------------------------------------------------

def Daphne_EwSinOp : Daphne_EwUnaryOp<"ewSin", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwCosOp : Daphne_EwUnaryOp<"ewCos", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwTanOp : Daphne_EwUnaryOp<"ewTan", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwSinhOp : Daphne_EwUnaryOp<"ewSinh", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwCoshOp : Daphne_EwUnaryOp<"ewCosh", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwTanhOp : Daphne_EwUnaryOp<"ewTanh", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwAsinOp : Daphne_EwUnaryOp<"ewAsin", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwAcosOp : Daphne_EwUnaryOp<"ewAcos", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_EwAtanOp : Daphne_EwUnaryOp<"ewAtan", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Comparison
// ----------------------------------------------------------------------------

def Daphne_EwIsNanOp : Daphne_EwUnaryOp<"ewIsnan", NumScalar, [ValueTypeFromFirstArg]>;

// ----------------------------------------------------------------------------
// Strings
// ----------------------------------------------------------------------------

def Daphne_EwLowerOp : Daphne_EwUnaryOp<"ewLower", StrScalar, [ValueTypeFromFirstArg]>;
def Daphne_EwUpperOp : Daphne_EwUnaryOp<"ewUpper", StrScalar, [ValueTypeFromFirstArg]>;

// ****************************************************************************
// Elementwise binary
// ****************************************************************************

class Daphne_EwBinaryOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeFromArgs,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    ShapeEwBinary,
    NoMemoryEffect,
])> {
    let arguments = (ins AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$lhs, AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$rhs);
    let results = (outs AnyTypeOf<[MatrixOf<[scalarType]>, scalarType, Unknown]>:$res);

    let hasFolder = 1;
}

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

// TODO Make EwAddOp Commutative again (see #449).
def Daphne_EwAddOp    : Daphne_EwBinaryOp<"ewAdd", AnyScalar, [ValueTypeFromArgs, 
                                                               CastArgsToResType/*, Commutative*/, 
                                                               EwSparseIfBoth, CUDASupport, 
                                                               DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>,
                                                               DeclareOpInterfaceMethods<InferSparsityOpInterface>]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwSubOp    : Daphne_EwBinaryOp<"ewSub", NumScalar, [ValueTypeFromArgs, CastArgsToResType, EwSparseIfBoth, CUDASupport, DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>, DeclareOpInterfaceMethods<InferSparsityOpInterface>]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwMulOp    : Daphne_EwBinaryOp<"ewMul", NumScalar, [ValueTypeFromArgs, CastArgsToResType, Commutative, EwSparseIfEither, CUDASupport, DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>, DeclareOpInterfaceMethods<InferSparsityOpInterface>]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwDivOp    : Daphne_EwBinaryOp<"ewDiv", NumScalar, [ValueTypeFromArgs, CastArgsToResType, CUDASupport, DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>, DeclareOpInterfaceMethods<InferSparsityOpInterface>]> {
    let hasCanonicalizeMethod = 1;
}
def Daphne_EwPowOp    : Daphne_EwBinaryOp<"ewPow", NumScalar, [ValueTypeFromArgs, CastArgsToResType, CUDASupport]>;
def Daphne_EwModOp    : Daphne_EwBinaryOp<"ewMod", NumScalar, [ValueTypeFromArgs, CastArgsToResType]>;
def Daphne_EwLogOp    : Daphne_EwBinaryOp<"ewLog", NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;

// ----------------------------------------------------------------------------
// Min/max
// ----------------------------------------------------------------------------

def Daphne_EwMinOp    : Daphne_EwBinaryOp<"ewMin", AnyScalar, [ValueTypeFromArgs, CastArgsToResType, Commutative]>;
def Daphne_EwMaxOp    : Daphne_EwBinaryOp<"ewMax", AnyScalar, [ValueTypeFromArgs, CastArgsToResType, Commutative, CUDASupport]>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_EwAndOp    : Daphne_EwBinaryOp<"ewAnd", NumScalar, [Commutative, ValueTypeFromArgsInt, CastArgsToResType]>;
def Daphne_EwOrOp     : Daphne_EwBinaryOp<"ewOr" , NumScalar, [Commutative, ValueTypeFromArgsInt, CastArgsToResType]>;
def Daphne_EwXorOp    : Daphne_EwBinaryOp<"ewXor", NumScalar, [Commutative, ValueTypeFromArgsInt, CastArgsToResType]>;

// ----------------------------------------------------------------------------
// Bitwise
// ----------------------------------------------------------------------------

def Daphne_EwBitwiseAndOp    : Daphne_EwBinaryOp<"ewBitwiseAnd", NumScalar, [Commutative, ValueTypeFromArgsInt, CastArgsToResType]>;

// ----------------------------------------------------------------------------
// Strings
// ----------------------------------------------------------------------------

def Daphne_EwConcatOp : Daphne_EwBinaryOp<"ewConcat", StrScalar, [ValueTypeStr, CastArgsToResType]>;

// ----------------------------------------------------------------------------
// Comparisons
// ----------------------------------------------------------------------------

class Daphne_EwCmpOp<string name, Type inputScalarType, list<Trait> traits = []>
: Daphne_EwBinaryOp<name, inputScalarType, !listconcat(traits, [ValueTypeCmp, CastArgsToMostGeneralArgType])> {
    // TODO: We do not enforce (matrix of) boolean output any more, but should
    // think about that again.
    //let results = (outs AnyTypeOf<[MatrixOf<[BoolScalar]>, BoolScalar, Unknown]>:$res);
}

def Daphne_EwEqOp  : Daphne_EwCmpOp<"ewEq" , AnyScalar, [Commutative]>;
def Daphne_EwNeqOp : Daphne_EwCmpOp<"ewNeq", AnyScalar, [Commutative, CUDASupport]>;
def Daphne_EwLtOp  : Daphne_EwCmpOp<"ewLt" , AnyScalar>;
def Daphne_EwLeOp  : Daphne_EwCmpOp<"ewLe" , AnyScalar>;
def Daphne_EwGtOp  : Daphne_EwCmpOp<"ewGt" , AnyScalar>;
def Daphne_EwGeOp  : Daphne_EwCmpOp<"ewGe" , AnyScalar>;

// ****************************************************************************
// Outer binary (generalized outer product)
// ****************************************************************************

class Daphne_OuterBinaryOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeMat,
    NumRowsFromIthArg<0>, NumColsFromIthArg<1>,
    CastArgsToResType,
    DeclareOpInterfaceMethods<VectorizableOpInterface>
])> {
    let arguments = (ins MatrixOf<[scalarType]>:$lhs, MatrixOf<[scalarType]>:$rhs);
    let results = (outs MatrixOf<[scalarType]>:$res);
}

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

def Daphne_OuterAddOp : Daphne_OuterBinaryOp<"outerAdd", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterSubOp : Daphne_OuterBinaryOp<"outerSub", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterMulOp : Daphne_OuterBinaryOp<"outerMul", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterDivOp : Daphne_OuterBinaryOp<"outerDiv", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterPowOp : Daphne_OuterBinaryOp<"outerPow", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterModOp : Daphne_OuterBinaryOp<"outerMod", NumScalar, [ValueTypeFromArgs]>;
def Daphne_OuterLogOp : Daphne_OuterBinaryOp<"outerLog", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Min/max
// ----------------------------------------------------------------------------

def Daphne_OuterMinOp : Daphne_OuterBinaryOp<"outerMin", AnyScalar, [ValueTypeFromArgs]>;
def Daphne_OuterMaxOp : Daphne_OuterBinaryOp<"outerMax", AnyScalar, [ValueTypeFromArgs]>;

// ----------------------------------------------------------------------------
// Logical
// ----------------------------------------------------------------------------

def Daphne_OuterAndOp : Daphne_OuterBinaryOp<"outerAnd", NumScalar, [ValueTypeFromArgsInt]>;
def Daphne_OuterOrOp  : Daphne_OuterBinaryOp<"outerOr" , NumScalar, [ValueTypeFromArgsInt]>;
def Daphne_OuterXorOp : Daphne_OuterBinaryOp<"outerXor", NumScalar, [ValueTypeFromArgsInt]>;

// ----------------------------------------------------------------------------
// Strings
// ----------------------------------------------------------------------------

def Daphne_OuterConcatOp : Daphne_OuterBinaryOp<"outerConcat", StrScalar>;

// ----------------------------------------------------------------------------
// Comparisons
// ----------------------------------------------------------------------------

class Daphne_OuterCmpOp<string name, Type inputScalarType, list<Trait> traits = []>
: Daphne_OuterBinaryOp<name, inputScalarType, !listconcat(traits, [ValueTypeFromArgs])> {
    // TODO: We do not enforce (matrix of) boolean output any more, but should
    // think about that again.
    //let results = (outs AnyTypeOf<[MatrixOf<[BoolScalar]>, BoolScalar, Unknown]>:$res);
}

def Daphne_OuterEqOp  : Daphne_OuterCmpOp<"outerEq" , AnyScalar>;
def Daphne_OuterNeqOp : Daphne_OuterCmpOp<"outerNeq", AnyScalar>;
def Daphne_OuterLtOp  : Daphne_OuterCmpOp<"outerLt" , AnyScalar>;
def Daphne_OuterLeOp  : Daphne_OuterCmpOp<"outerLe" , AnyScalar>;
def Daphne_OuterGtOp  : Daphne_OuterCmpOp<"outerGt" , AnyScalar>;
def Daphne_OuterGeOp  : Daphne_OuterCmpOp<"outerGe" , AnyScalar>;

// ****************************************************************************
// Elementwise ternary
// ****************************************************************************

def Daphne_CondOp : Daphne_Op<"cond", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let arguments = (ins AnyTypeOf<[Matrix, AnyScalar, Unknown]>:$cond, AnyType:$thenVal, AnyType:$elseVal);
    let results = (outs AnyType:$res);

    let hasCanonicalizeMethod = 1;
}

// ****************************************************************************
// Aggregation and statistical
// ****************************************************************************

class Daphne_AggOp<string name, Type inScalarType, list<Trait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins MatrixOf<[inScalarType]>:$arg);
}

// ----------------------------------------------------------------------------
// Full aggregation
// ----------------------------------------------------------------------------

class Daphne_AllAggOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_AggOp<name, scalarType, !listconcat(traits, [DataTypeSca, CastArgsToResType])> {
    let results = (outs scalarType:$res);
}

def Daphne_AllAggSumOp    : Daphne_AllAggOp<"sumAll", NumScalar, [ValueTypeFromFirstArg, DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_AllAggMinOp    : Daphne_AllAggOp<"minAll", NumScalar, [ValueTypeFromFirstArg, DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_AllAggMaxOp    : Daphne_AllAggOp<"maxAll", NumScalar, [ValueTypeFromFirstArg, DeclareOpInterfaceMethods<VectorizableOpInterface>]>;
def Daphne_AllAggMeanOp   : Daphne_AllAggOp<"meanAll", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_AllAggVarOp    : Daphne_AllAggOp<"varAll", NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_AllAggStddevOp : Daphne_AllAggOp<"stddevAll", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Row/column-wise aggregation
// ----------------------------------------------------------------------------

class Daphne_DimAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<Trait> traits = []>
: Daphne_AggOp<name, inScalarType, !listconcat(traits, [DataTypeMat])> {
    let results = (outs MatrixOf<[outScalarType]>:$res);
}

class Daphne_RowAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<Trait> traits = []>
: Daphne_DimAggOp<name, inScalarType, outScalarType, !listconcat(traits, [
    NumRowsFromArg, OneCol
])>;
class Daphne_ColAggOp<string name, Type inScalarType, Type outScalarType = inScalarType, list<Trait> traits = []>
: Daphne_DimAggOp<name, inScalarType, outScalarType, !listconcat(traits, [
    OneRow, NumColsFromArg
])>;

def Daphne_RowAggSumOp    : Daphne_RowAggOp<"sumRow"   , NumScalar, NumScalar, [
    ValueTypeFromFirstArg, CastArgsToResType, DeclareOpInterfaceMethods<VectorizableOpInterface>, CUDASupport
]>;
def Daphne_RowAggMinOp    : Daphne_RowAggOp<"minRow"   , AnyScalar, AnyScalar, [
    ValueTypeFromFirstArg, CastArgsToResType, DeclareOpInterfaceMethods<VectorizableOpInterface>
]>;
def Daphne_RowAggMaxOp    : Daphne_RowAggOp<"maxRow"   , AnyScalar, AnyScalar, [
    ValueTypeFromFirstArg, CastArgsToResType, DeclareOpInterfaceMethods<VectorizableOpInterface>, CUDASupport,
    DeclareOpInterfaceMethods<DistributableOpInterface>
]>;
def Daphne_RowAggIdxMinOp : Daphne_RowAggOp<"idxminRow", NumScalar, Size, [ValueTypeSize]>;
def Daphne_RowAggIdxMaxOp : Daphne_RowAggOp<"idxmaxRow", NumScalar, Size, [ValueTypeSize]>;
def Daphne_RowAggMeanOp   : Daphne_RowAggOp<"meanRow"  , NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;
def Daphne_RowAggVarOp    : Daphne_RowAggOp<"varRow"   , NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;
def Daphne_RowAggStddevOp : Daphne_RowAggOp<"stddevRow", NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;

def Daphne_ColAggSumOp    : Daphne_ColAggOp<"sumCol"   , NumScalar, NumScalar, [
    ValueTypeFromFirstArg, CastArgsToResType, DeclareOpInterfaceMethods<VectorizableOpInterface>, CUDASupport
]>;
def Daphne_ColAggMinOp    : Daphne_ColAggOp<"minCol"   , AnyScalar, AnyScalar, [
    ValueTypeFromFirstArg, CastArgsToResType, DeclareOpInterfaceMethods<VectorizableOpInterface>
]>;
def Daphne_ColAggMaxOp    : Daphne_ColAggOp<"maxCol"   , AnyScalar, AnyScalar, [
    ValueTypeFromFirstArg, CastArgsToResType, DeclareOpInterfaceMethods<VectorizableOpInterface>
]>;
def Daphne_ColAggIdxMinOp : Daphne_ColAggOp<"idxminCol", NumScalar, Size, [ValueTypeSize]>;
def Daphne_ColAggIdxMaxOp : Daphne_ColAggOp<"idxmaxCol", NumScalar, Size, [ValueTypeSize]>;
def Daphne_ColAggMeanOp   : Daphne_ColAggOp<"meanCol"  , NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;
def Daphne_ColAggVarOp    : Daphne_ColAggOp<"varCol"   , NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;
def Daphne_ColAggStddevOp : Daphne_ColAggOp<"stddevCol", NumScalar, NumScalar, [ValueTypeFromArgsFP, CastArgsToResType]>;

// ----------------------------------------------------------------------------
// Cumulative aggregation
// ----------------------------------------------------------------------------

class Daphne_CumAggOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_AggOp<name, scalarType, !listconcat(traits, [
    DataTypeMat, ShapeFromArg
])> {
    let results = (outs MatrixOf<[scalarType]>);
}

def Daphne_CumAggSumOp  : Daphne_CumAggOp<"sumCum" , NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_CumAggProdOp : Daphne_CumAggOp<"prodCum", NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_CumAggMinOp  : Daphne_CumAggOp<"minCum" , AnyScalar, [ValueTypeFromFirstArg]>;
def Daphne_CumAggMaxOp  : Daphne_CumAggOp<"maxCum" , AnyScalar, [ValueTypeFromFirstArg]>;

// ----------------------------------------------------------------------------
// Grouped aggregation
// ----------------------------------------------------------------------------

class Daphne_GrpAggOp<string name, Type scalarType, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeMat,
    NumRowsFromIthScalar<2>, NumColsFromArg
])> {
    let arguments = (ins MatrixOf<[scalarType]>:$arg, MatrixOf<[Size]>:$groupIds, Size:$numGroups/*, Optional<MatrixOrU>:$weights*/);
    let results = (outs MatrixOf<[scalarType]>:$groupAggs);
}

def Daphne_GrpAggCountOp  : Daphne_GrpAggOp<"countGrp" , AnyScalar>;
def Daphne_GrpAggSumOp    : Daphne_GrpAggOp<"sumGrp"   , NumScalar, [ValueTypeFromFirstArg]>;
def Daphne_GrpAggMinOp    : Daphne_GrpAggOp<"minGrp"   , AnyScalar, [ValueTypeFromFirstArg]>;
def Daphne_GrpAggMaxOp    : Daphne_GrpAggOp<"maxGrp"   , AnyScalar, [ValueTypeFromFirstArg]>;
def Daphne_GrpAggMeanOp   : Daphne_GrpAggOp<"meanGrp"  , NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_GrpAggVarOp    : Daphne_GrpAggOp<"varGrp"   , NumScalar, [ValueTypeFromArgsFP]>;
def Daphne_GrpAggStddevOp : Daphne_GrpAggOp<"stddevGrp", NumScalar, [ValueTypeFromArgsFP]>;

// ----------------------------------------------------------------------------
// Statistical for column matrices
// ----------------------------------------------------------------------------

def Daphne_MedianOp : Daphne_Op<"median"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg, Optional<MatrixOf<[FloatScalar]>>:$weights);
    let results = (outs FloatScalar:$res);
}

def Daphne_QuantileOp : Daphne_Op<"quantile"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg, MatrixOf<[FloatScalar]>:$ps, Optional<MatrixOf<[FloatScalar]>>:$weights);
    let results = (outs MatrixOf<[FloatScalar]>:$res);
}

def Daphne_MomentOp : Daphne_Op<"moment"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg, Size:$k, Optional<MatrixOf<[FloatScalar]>>:$weights);
    let results = (outs FloatScalar:$res);
}

def Daphne_CovOp : Daphne_Op<"cov"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$lhs, MatrixOf<[FloatScalar]>:$rhs, Optional<MatrixOf<[FloatScalar]>>:$weights);
    let results = (outs FloatScalar:$res);
}

// ****************************************************************************
// Left and right indexing
// ****************************************************************************

def Daphne_ExtractOp : Daphne_Op<"extract", [
    NumRowsFromIthArg<1>, NumColsFromIthArgNumRows<2>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>
]> {
    let summary = "Copies the specified rows and columns from the argument to the result.";

    let description = [{
        This operation is effectively similar to a combination of `extractRow`
        and `extractCol`. Please see the documentation of those operations.
    }];

    let arguments = (ins MatrixOrFrame:$source, Selection:$selectedRows, AnyTypeOf<[Selection, StrScalar, Unknown]>:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ExtractRowOp : Daphne_Op<"extractRow", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    NumRowsFromIthArg<1>, NumColsFromArg
]> {
    let summary = "Copies the specified rows from the argument to the result.";

    let description = [{
        Copies the rows of a data object `source` (matrix or frame) at the
        positions specified by `selectedRows` to the result. `selectedRows`
        must be either a scalar or a single-column matrix of positions (row
        indexes) into `source`, i.e., the entries must be between zero
        (inclusive) and the number of rows in `source` (exclusive). Beyond that,
        there are no restrictions.

        Note that `FilterRowOp` is similar, but works with bit vectors instead
        of positions lists.
    }];

    let arguments = (ins MatrixOrFrame:$source, Selection:$selectedRows);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_SliceRowOp : Daphne_Op<"sliceRow", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let summary = "Copies the specified rows from the argument to the result.";

    let description = [{
        Copies the rows of a data object `source` (matrix or frame) at the
        positions in `[lowerIncl, upperExcl)` to the result.
    }];

    let arguments = (ins MatrixOrFrame:$source, SI64:$lowerIncl, SI64:$upperExcl);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ExtractColOp : Daphne_Op<"extractCol", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>, CUDASupport, Pure
]> {
    let summary = "Copies the specified columns from the argument to the result.";

    let description = [{
        Copies the columns of a data object `source` (matrix or frame) at the
        positions specified by `selectedCols` to the result. `selectedCols`
        must be either a scalar or a single-column matrix of positions (column
        indexes) into `source`, i.e., the entries must be between zero
        (inclusive) and the number of columns in `source` (exclusive). Beyond
        that, there are no restrictions.

        Alternatively, `selectedCols` can be a string, in which case the column
        with that label is extracted from `source` if it is a frame.

        Note that `FilterRowOp` is similar, but works with bit vectors instead
        of positions lists.
    }];

    let arguments = (ins MatrixOrFrame:$source, AnyTypeOf<[Selection, StrScalar, Unknown]>:$selectedCols);
    let results = (outs MatrixOrFrame:$res);

    let hasCanonicalizeMethod = 1;
}

def Daphne_SliceColOp : Daphne_Op<"sliceCol", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let summary = "Copies the specified columns from the argument to the result.";

    let description = [{
        Copies the columns of a data object `source` (matrix or frame) at the
        positions in `[lowerIncl, upperExcl)` to the result.
    }];

    let arguments = (ins MatrixOrFrame:$source, SI64:$lowerIncl, SI64:$upperExcl);
    let results = (outs MatrixOrFrame:$res);
}

// TODO Create combined InsertOp (see #238).

def Daphne_InsertRowOp : Daphne_Op<"insertRow", [
    TypeFromFirstArg, // this is debatable
    ShapeFromArg
]> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOrFrame:$ins, SI64:$rowLowerIncl, SI64:$rowUpperExcl);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_InsertColOp : Daphne_Op<"insertCol", [
    TypeFromFirstArg, // this is debatable
    ShapeFromArg
]> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOrFrame:$ins, SI64:$colLowerIncl, SI64:$colUpperExcl);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Reorganization
// ****************************************************************************

def Daphne_ReshapeOp : Daphne_Op<"reshape", [
    TypeFromFirstArg,
    NumRowsFromIthScalar<1>, NumColsFromIthScalar<2>, SparsityFromArg,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>
]> {
    let arguments = (ins MatrixOrU:$arg, Size:$numRows, Size:$numCols);
    let results = (outs MatrixOrU:$res);
}

def Daphne_TransposeOp : Daphne_Op<"transpose", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromArgNumCols, NumColsFromArgNumRows, SparsityFromArg, CUDASupport,
    DeclareOpInterfaceMethods<InferSymmetricOpInterface>,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>,
    Pure
]> {
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);

    let builders = [
        OpBuilder<(ins "Value":$input), [{
            auto ty = input.getType();
            build($_builder, $_state, ty, input);
        }]>,
    ];
}

class Daphne_BindOp<string name, list<Trait> traits = []> : Daphne_Op<name, !listconcat(traits, [
    DataTypeFromArgs,
    CastArgsToResType
])> {
    let arguments = (ins MatrixOrFrame:$lhs, MatrixOrFrame:$rhs);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_ColBindOp : Daphne_BindOp<"colBind", [
    ValueTypesConcat,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromAllArgs, NumColsFromSumOfAllArgs, CUDASupport,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>,
    Pure
]>;

def Daphne_RowBindOp : Daphne_BindOp<"rowBind", [
    ValueTypeFromArgs,
    NumRowsFromSumOfAllArgs, NumColsFromAllArgs,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>
]>;

def Daphne_ReverseOp : Daphne_Op<"reverse", [
    TypeFromFirstArg, ShapeFromArg
]> {
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);
}

def Daphne_OrderOp : Daphne_Op<"order", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>, // due to possibility of returning indexes
    SameVariadicOperandSize,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    // TODO Maybe colIdxs and ascs should be attributes.
    let arguments = (ins MatrixOrFrame:$arg, Variadic<Size>:$colIdxs, Variadic<BoolScalar>:$ascs, BoolScalar:$returnIdxs);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Matrix decompositions & co
// ****************************************************************************

def Daphne_EigenOp : Daphne_Op<"eigenCal", [TypeFromFirstArg, DeclareOpInterfaceMethods<InferTypesOpInterface>,
        DeclareOpInterfaceMethods<InferShapeOpInterface>]> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$eigenValues, MatrixOf<[FloatScalar]>:$eigenVectors);
}

def Daphne_LuOp : Daphne_Op<"lu"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$p, MatrixOf<[FloatScalar]>:$l, MatrixOf<[FloatScalar]>:$u);
}

def Daphne_QrOp : Daphne_Op<"qr"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$h, MatrixOf<[FloatScalar]>:$r);
}

def Daphne_SvdOp : Daphne_Op<"svd"> {
    let arguments = (ins MatrixOf<[FloatScalar]>:$arg);
    let results = (outs MatrixOf<[FloatScalar]>:$u, MatrixOf<[FloatScalar]>:$s, MatrixOf<[FloatScalar]>:$v);
}

// ****************************************************************************
// Deep neural network
// ****************************************************************************

// TODO The DNN-related operations share several arguments. Is there an elegant
// way to specify the *common* arguments just once?

// ----------------------------------------------------------------------------
// Activation
// ----------------------------------------------------------------------------

class Daphne_ActivationForwardOp<string name, list<Trait> traits = []> :
        Daphne_Op<name, traits> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

def Daphne_ReluForwardOp : Daphne_ActivationForwardOp<"NN_Relu_Forward", [ ValueTypeFromFirstArg, DataTypeFromFirstArg, CUDASupport ]>;

// ----------------------------------------------------------------------------
// Affine
// ----------------------------------------------------------------------------

def Daphne_AffineForwardOp : Daphne_Op<"NN_Affine_Forward", [ ValueTypeFromFirstArg, DataTypeFromFirstArg, CUDASupport ]> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$weights,
            MatrixOf<[FloatScalar]>:$bias);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ----------------------------------------------------------------------------
// Batch Normalization
// ----------------------------------------------------------------------------

def Daphne_BatchNorm2DInferenceForwardOp : Daphne_Op<"BatchNormInferenceForward", [ ValueTypeFromFirstArg, DataTypeFromFirstArg, CUDASupport ]> {
    let arguments = (ins
            MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$gamma, MatrixOf<[FloatScalar]>:$beta,
            MatrixOf<[FloatScalar]>:$emaMean, MatrixOf<[FloatScalar]>:$emaVar, FloatScalar:$eps);
    let results = (outs MatrixOf<[FloatScalar]>:$output);
}

def Daphne_BatchNorm2DTrainForwardOp : Daphne_Op<"BatchNorm_Train_Forward", [ ValueTypeFromFirstArg, DataTypeFromFirstArg, CUDASupport ]> {
    let arguments = (ins
            MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$gamma, MatrixOf<[FloatScalar]>:$beta,
            MatrixOf<[FloatScalar]>:$emaMean, MatrixOf<[FloatScalar]>:$emaVar, FloatScalar:$eps, FloatScalar:$mu);
    let results = (outs MatrixOf<[FloatScalar]>:$output,
                        MatrixOf<[FloatScalar]>:$new_emaMean,
                        MatrixOf<[FloatScalar]>:$new_emaVar,
                        MatrixOf<[FloatScalar]>:$Mean,
                        MatrixOf<[FloatScalar]>:$invVar);
}

def Daphne_BatchNorm2DBackwardOp : Daphne_Op<"BatchNorm_Backward", [ ValueTypeFromFirstArg, DataTypeFromFirstArg, CUDASupport ]> {
    let arguments = (ins
            MatrixOf<[FloatScalar]>:$mean, MatrixOf<[FloatScalar]>:$invVar, MatrixOf<[FloatScalar]>:$input,
            MatrixOf<[FloatScalar]>:$dout, MatrixOf<[FloatScalar]>:$gamma, FloatScalar:$eps);
    let results = (outs MatrixOf<[FloatScalar]>:$dX,
                        MatrixOf<[FloatScalar]>:$dGamma,
                        MatrixOf<[FloatScalar]>:$dBeta);
}

// ----------------------------------------------------------------------------
// Bias Addition
// ----------------------------------------------------------------------------

def Daphne_BiasAddOp : Daphne_Op<"BiasAdd", [ ValueTypeFromFirstArg, DataTypeFromFirstArg, CUDASupport ]> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$bias);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ----------------------------------------------------------------------------
// Convolution
// ----------------------------------------------------------------------------

def Daphne_Conv2DForwardOp : Daphne_Op<"Convolution_Forward",
    [ DeclareOpInterfaceMethods<InferTypesOpInterface>, DeclareOpInterfaceMethods<InferShapeOpInterface>,
      CUDASupport ]> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$filter, MatrixOf<[FloatScalar]>:$bias,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // filter shape (num filters & channels is encoded in dimensions of filter matrix (-> [F,C*H*W]))
        Size:$filterHeight, Size:$filterWidth, Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth);

    let results = (outs MatrixOf<[FloatScalar]>:$output, Size:$outHeight, Size:$outWidth);
}

def Daphne_Conv2DBackwardFilterOp : Daphne_Op<"conv2DBackwardFilter"> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$output,
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // filter shape
        Size:$filterNumFilters, Size:$filterNumChannels, Size:$filterHeight, Size:$filterWidth
    );
    let results = (outs MatrixOf<[FloatScalar]>:$filter);
}

def Daphne_Conv2DBackwardDataOp : Daphne_Op<"conv2DBackwardData"> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$filter, MatrixOf<[FloatScalar]>:$output,
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth,
        // input shape
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // filter shape
        Size:$filterNumFilters, Size:$filterNumChannels, Size:$filterHeight, Size:$filterWidth
    );
    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ----------------------------------------------------------------------------
// Pooling
// ----------------------------------------------------------------------------

class Daphne_PoolForwardOp<string name, list<Trait> traits = []> : Daphne_Op<name, traits> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input,
        // input shape (NCHW)
        Size:$inputBatchSize, Size:$inputNumChannels, Size:$inputHeight, Size:$inputWidth,
        // pool shape (ToDo: optional, default filter size 2x2)
        Size:$poolHeight, Size:$poolWidth,
        // filter application properties (ToDo: optional, default stride=1x1, padding=0x0)
        Size:$strideHeight, Size:$strideWidth, Size:$padHeight, Size:$padWidth
    );

    let results = (outs MatrixOf<[FloatScalar]>:$data, Size:$outHeight, Size:$outWidth);
}

def Daphne_MaxPoolForwardOp : Daphne_PoolForwardOp<"NN_Max_Forward", [ DeclareOpInterfaceMethods<InferTypesOpInterface>,
                                                                       DeclareOpInterfaceMethods<InferShapeOpInterface>,
                                                                       CUDASupport ]>;
def Daphne_AvgPoolForwardOp : Daphne_PoolForwardOp<"NN_Avg_Forward", [ DeclareOpInterfaceMethods<InferTypesOpInterface>,
                                                                       DeclareOpInterfaceMethods<InferShapeOpInterface>,
                                                                       CUDASupport ]>;

def Daphne_AvgPoolBackwardOp : Daphne_Op<"avgPoolBackward", [ CUDASupport ]> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$dOut,
        Size:$batch_size, Size:$num_channels, Size:$img_h, Size:$img_w,
        Size:$pool_h, Size:$pool_w,
        Size:$stride_h, Size:$stride_w, Size:$pad_h, Size:$pad_w
    );
    let results = (outs MatrixOf<[FloatScalar]>:$res);
}

def Daphne_MaxPoolBackwardOp : Daphne_Op<"maxPoolBackward", [ CUDASupport ]> {
    let arguments = (ins
        MatrixOf<[FloatScalar]>:$input, MatrixOf<[FloatScalar]>:$dOut,
        Size:$batch_size, Size:$num_channels, Size:$img_h, Size:$img_w,
        Size:$pool_h, Size:$pool_w,
        Size:$stride_h, Size:$stride_w, Size:$pad_h, Size:$pad_w
    );
    let results = (outs MatrixOf<[FloatScalar]>:$res);
}

// ----------------------------------------------------------------------------
// Softmax
// ----------------------------------------------------------------------------
def Daphne_SoftmaxOp : Daphne_Op<"Softmax", [ DataTypeFromFirstArg, ValueTypeFromFirstArg, CUDASupport ]> {
    let arguments = (ins  MatrixOf<[FloatScalar]>:$input);

    let results = (outs MatrixOf<[FloatScalar]>:$data);
}

// ****************************************************************************
// Other matrix operations
// ****************************************************************************

def Daphne_DiagVectorOp : Daphne_Op<"diagVector", [
    TypeFromFirstArg, NumRowsFromArg, OneCol
]> {
    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);
}

def Daphne_TriOp : Daphne_Op<"tri", [
    TypeFromFirstArg, ShapeFromArg, DeclareOpInterfaceMethods<InferSparsityOpInterface>
]> {
    let arguments = (ins MatrixOrU:$arg, BoolScalar:$upper, BoolScalar:$diag, BoolScalar:$values);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SolveOp : Daphne_Op<"solve", [
    DataTypeMat, ValueTypeFromArgs, NumRowsFromArg, OneCol, CUDASupport, CastArgsToResType
]> {
    let arguments = (ins MatrixOf<[NumScalar]>:$a, MatrixOf<[NumScalar]>:$b);
    let results = (outs MatrixOf<[NumScalar]>:$x);
}

def Daphne_ReplaceOp : Daphne_Op<"replace", [
    DataTypeFromFirstArg, ValueTypeFromArgs, ShapeFromArg, CastArgsToResType
]> {
    let arguments = (ins MatrixOrU:$arg, AnyScalar:$pattern, AnyScalar:$replacement);
    let results = (outs MatrixOrU:$res);
}

def Daphne_CTableOp : Daphne_Op<"ctable", [
    DataTypeMat, ValueTypeFromThirdArg,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let arguments = (ins MatrixOrU:$lhs, MatrixOrU:$rhs, NumScalar:$weight, SI64:$resNumRows, SI64:$resNumCols);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SyrkOp : Daphne_Op<"syrk", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    NumRowsFromArgNumCols, NumColsFromArg, CUDASupport,FPGAOPENCLSupport, 
    CastFirstArgToResType
]> {
    let summary = [{Performs the operation `t(A) @ A` if `transLeft` is `true`; or `A @ t(A)`, otherwise.}];
    let arguments = (ins MatrixOrU:$arg, BoolScalar:$transLeft);
    let results = (outs MatrixOrU:$res);
}

def Daphne_GemvOp : Daphne_Op<"gemv", [
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DataTypeMat, ValueTypeFromArgs,
    NumRowsFromArgNumCols, OneCol, CUDASupport,
    CastArgsToResType
]> {
    // TODO: support `A @ x` operation
    let summary = [{Performs the operation `t(A) @ x`}];
    let arguments = (ins MatrixOrU:$mat, MatrixOrU:$vec);
    let results = (outs MatrixOrU:$res);
}

// ****************************************************************************
// Extended relational algebra
// ****************************************************************************

// ----------------------------------------------------------------------------
// Entire SQL query
// ----------------------------------------------------------------------------

def Daphne_SqlOp : Daphne_Op<"sql"> {
    let arguments = (ins StrAttr:$sql);
    let results = (outs FrameOrU:$res);
}

def Daphne_RegisterViewOp : Daphne_Op<"registerView"> {
    let arguments = (ins StrAttr:$view, MatrixOrFrame:$arg);
    let results = (outs); // no results
}

// ----------------------------------------------------------------------------
// Set operations
// ----------------------------------------------------------------------------

class Daphne_SetOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeFrm, ValueTypeFromArgs,
    NumColsFromAllArgs
])> {
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs);
    let results = (outs FrameOrU:$res);
}

def Daphne_IntersectOp : Daphne_SetOp<"intersect">;
def Daphne_MergeOp : Daphne_SetOp<"merge">;
def Daphne_ExceptOp : Daphne_SetOp<"except">;

// ----------------------------------------------------------------------------
// Cartesian product and joins
// ----------------------------------------------------------------------------

def Daphne_CartesianOp : Daphne_Op<"cartesian", [
    DataTypeFrm, ValueTypesConcat,
    NumColsFromSumOfAllArgs,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferNumRowsOpInterface>
]> {
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs); //let arguments = (ins Variadic<FrameOrU>:$args);
    let results = (outs FrameOrU:$res);
}

class Daphne_JoinOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeFrm, ValueTypesConcat,
    SameVariadicOperandSize,
    NumColsFromSumOfAllArgs
])> {
    // TODO Maybe leftOn and rightOn should be attributes.
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs, Variadic<Size>:$leftOn, Variadic<Size>:$rightOn);
    let results = (outs FrameOrU:$res);
}

def Daphne_InnerJoinOp : Daphne_Op<"innerJoin", [
    DataTypeFrm, ValueTypesConcat,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
]> {
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn, Size:$numRowRes);
    let results = (outs FrameOrU:$res);
}


def Daphne_CompareOperation_Equal : I32EnumAttrCase<"Equal", 1>;
def Daphne_CompareOperation_LessThan : I32EnumAttrCase<"LessThan", 2>;
def Daphne_CompareOperation_LessEqual : I32EnumAttrCase<"LessEqual", 3>;
def Daphne_CompareOperation_GreaterThan : I32EnumAttrCase<"GreaterThan", 4>;
def Daphne_CompareOperation_GreaterEqual : I32EnumAttrCase<"GreaterEqual", 5>;
def Daphne_CompareOperation_NotEqual : I32EnumAttrCase<"NotEqual", 6>;

def Daphne_CompareEnum : I32EnumAttr<"CompareOperation", "", [
   Daphne_CompareOperation_Equal, Daphne_CompareOperation_LessThan, Daphne_CompareOperation_LessEqual,
   Daphne_CompareOperation_GreaterThan, Daphne_CompareOperation_GreaterEqual, Daphne_CompareOperation_NotEqual
   ]>{
    let cppNamespace = "::mlir::daphne";
}

def Daphne_ThetaJoinOp : Daphne_Op<"thetaJoin", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DataTypeFrm, ValueTypesConcat
]> {
    let arguments = (
        ins Frame:$lhs,
        Frame:$rhs,
        Variadic<StrScalar>:$lhsOn,
        Variadic<StrScalar>:$rhsOn,
        TypedArrayAttrBase<Daphne_CompareEnum, "enum">:$cmp);
    let results = (outs Frame:$res);
}


def Daphne_FullOuterJoinOp : Daphne_JoinOp<"fullOuterJoin">;
def Daphne_LeftOuterJoinOp : Daphne_JoinOp<"leftOuterJoin">;
def Daphne_AntiJoinOp : Daphne_JoinOp<"antiJoin">;

// TODO Reconcile this with the other join ops, but we need it to work quickly now.
def Daphne_SemiJoinOp : Daphne_Op<"semiJoin", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    NumColsFromArg
]> {
    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn, Size:$numRowRes);
    let results = (outs FrameOrU:$res, MatrixOf<[Size]>:$lhsTids);
}

def Daphne_GroupJoinOp : Daphne_Op<"groupJoin", [
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    // TODO Support arbitrary aggregation functions.
    // TODO Support an arbitrary number of aggregates.
    // TODO Support an arbitrary number of join/group columns.

    let summary = [{
        Group-join of `lhs` and `rhs` on `lhs.lhsOn == rhs.rhsOn` with
        summation of `rhs.rhsAgg`.
    }];

    let description = [{
        Performs a group-join of the input frames `lhs` and `rhs` on
        `lhs.lhsOn == rhs.rhsOn`, where `lhsOn` and `rhsOn` are column labels
        in the respective frame, including a summation on `rhs.rhsAgg`, where
        `rhsAgg` is a column label in `rhs`.

        This is equivalent to an inner join of `lhs` and `rhs` on
        `lhs.lhsOn == rhs.rhsOn` followed by a grouping of the result on
        `lhsOn` including a summation on `rhsAgg`.

        `lhs.lhsOn` is assumed to be unique.

        The results are:
        - A two-column frame consisting of (1) the distinct values in
          `lhsOn`/`rhsOn` that pass the join and (2) the grouped aggregates in
          `rhsAgg`. The column labels are `lhsOn` and `rhsAgg`.
        - A column-matrix containing the positions of the tuples in `lhs`
          corresponding to the pairs in the first result.
    }];

    let arguments = (ins FrameOrU:$lhs, FrameOrU:$rhs, StrScalar:$lhsOn, StrScalar:$rhsOn, StrScalar:$rhsAgg);
    // TODO Result `lhsTids` could be made optional.
    let results = (outs FrameOrU:$res, MatrixOf<[Size]>:$lhsTids);
}

// ----------------------------------------------------------------------------
// Selection
// ----------------------------------------------------------------------------

def Daphne_QueryOp : Daphne_Op<"query", [
    TypeFromFirstArg, NumColsFromArg
]> {
    let arguments = (ins FrameOrU:$arg, StrAttr:$cond);
    let results = (outs FrameOrU:$res);
}

def Daphne_FilterRowOp : Daphne_Op<"filterRow", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    NumColsFromArg, Pure
]> {
    let summary = "Filters the rows of a data object according to a bit vector";

    let description = [{
        Filters the rows of a data object (matrix or frame) according to a bit
        vector. The argument `selectedRows` must be a single-column matrix. All
        its entries must be zero or one. The value type of `selectedRows` does
        not need to be boolean, it could also be an integer or floating-point
        type. `selectedRows` and `source` must have the same number of rows.
        The *i*-th row of `source` is contained in the result, if the *i*-th
        entry of `selectedRows` is a one (in its respective value type).

        Note that `ExtractRowOp` and `SliceRowOp` are similar, but work with
        positions instead of bit vectors.
    }];

    let arguments = (ins MatrixOrFrame:$source, MatrixOrU:$selectedRows);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_FilterColOp : Daphne_Op<"filterCol", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    // TODO Support frame label inference, (see #484).
    /*DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,*/
    NumRowsFromArg
]> {
    let summary = "Filters the columns of a data object according to a bit vector";

    let description = [{
        Filters the columns of a data object (matrix or frame) according to a bit
        vector. The argument `selectedCols` must be a single-column matrix. All
        its entries must be zero or one. The value type of `selectedCols` does
        not need to be boolean, it could also be an integer or floating-point
        type. The number of entries (rows) in `selectedCols` and the number of
        columns in `source` must be the same.
        The *i*-th column of `source` is contained in the result, if the *i*-th
        entry of `selectedCols` is a one (in its respective value type).

        Note that `ExtractColOp` and `SliceColOp` are similar, but work with
        positions instead of bit vectors.
    }];

    let arguments = (ins MatrixOrFrame:$source, MatrixOrU:$selectedCols);
    let results = (outs MatrixOrFrame:$res);
}

// Note that ExtractOp can be used to filter rows by a column of bool or a
// column of positions.

// ----------------------------------------------------------------------------
// Grouping
// ----------------------------------------------------------------------------

def Daphne_GroupAggCount : I32EnumAttrCase<"COUNT", 1>;
def Daphne_GroupAggSum : I32EnumAttrCase<"SUM", 2>;
def Daphne_GroupAggMin : I32EnumAttrCase<"MIN", 3>;
def Daphne_GroupAggMax : I32EnumAttrCase<"MAX", 4>;
def Daphne_GroupAggAvg : I32EnumAttrCase<"AVG", 5>;

def Daphne_GroupAggEnum : I32EnumAttr<"GroupEnum", "", [Daphne_GroupAggCount, Daphne_GroupAggSum, Daphne_GroupAggMin, Daphne_GroupAggMax, Daphne_GroupAggAvg]>{
    let cppNamespace = "::mlir::daphne";
}

def Daphne_GroupOp : Daphne_Op<"group", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>]>{
    let arguments = (
        ins FrameOrU:$frame,
        Variadic<StrScalar>:$keyCol,
        Variadic<StrScalar>:$aggCol,
        TypedArrayAttrBase<Daphne_GroupAggEnum, "enum">:$aggFuncs
    );
    let results = (outs FrameOrU:$res);
}

// ****************************************************************************
// Columnar algebra
// ****************************************************************************
// TODO These columnar operations (plus the Column type) could become a separate dialect.

class Daphne_ColSelectCmpOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeCol, ValueTypeSize
])> {
    let summary = "Selection on columnar data; input: data; output: sorted positions.";
    let description = [{
        Compares each value of the left-hand-side input column to the right-hand-side input scalar using the concrete
        comparison operation. The output column contains the positions (counting starts at zero) of the matching
        values in the left-hand-side input column. The output column is sorted.

        Note that this operation can only compare a column to a scalar. To compare two columns, consider using
        `ColCalcBinaryOp`.
    }];

    let arguments = (ins ColumnOrU:$lhsData, AnyTypeOf<[AnyScalar, Unknown]>:$rhsData);
    let results = (outs ColumnOrU:$resPos);
}
def Daphne_ColSelectEqOp : Daphne_ColSelectCmpOp<"colSelectEq">;
def Daphne_ColSelectNeqOp : Daphne_ColSelectCmpOp<"colSelectNeq">;
def Daphne_ColSelectGtOp : Daphne_ColSelectCmpOp<"colSelectGt">;
def Daphne_ColSelectGeOp : Daphne_ColSelectCmpOp<"colSelectGe">;
def Daphne_ColSelectLtOp : Daphne_ColSelectCmpOp<"colSelectLt">;
def Daphne_ColSelectLeOp : Daphne_ColSelectCmpOp<"colSelectLe">;

def Daphne_ColProjectOp : Daphne_Op<"colProject", [
    DataTypeCol, ValueTypeFromFirstArg
]> {
    let summary = "Projection on columnar data; input: data, positions; output: data.";
    let description = [{
        Extracts the data elements addressed by the positions in the right-hand-side input column from the
        left-hand-side input column. The input positions must be valid positions in the input data column (not out of
        bounds); other than that, there are no restrictions, i.e., the positions do not need to be sorted and may
        contain duplicates. The elements in the output data column correspond to the input positions, i.e., are in the
        same order.
    }];

    let arguments = (ins ColumnOrU:$lhsData, ColumnOrU:$rhsPos);
    let results = (outs ColumnOrU:$resData);
}

class Daphne_ColSetOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeCol, ValueTypeSize
])> {
    let arguments = (ins ColumnOrU:$lhsPos, ColumnOrU:$rhsPos);
    let results = (outs ColumnOrU:$resPos);
}
def Daphne_ColIntersectOp : Daphne_ColSetOp<"colIntersect"> {
    let summary = "Set intersection of columnar positions lists; input: sorted and unique positions; output: sorted and unique positions.";
    let description = [{
        Intersects the two given position list columns, i.e., returns a column containing the positions present in both
        inputs. Both inputs must be sorted and unique (i.e., must not contain duplicates). The output positions are
        sorted and unique, too.
    }];
}
def Daphne_ColMergeOp : Daphne_ColSetOp<"colMerge"> {
    let summary = "Set union of columnar positions lists; input: sorted and unique positions; output: sorted and unique positions.";
    let description = [{
        Merges the two given position list columns, i.e., returns a column containing the positions present in any
        input. Both inputs must be sorted and unique (i.e., must not contain duplicates). The output positions are
        sorted and unique, too.
    }];
}

def Daphne_ColJoinOp : Daphne_Op<"colJoin", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>
]> {
    let summary = "N:1 equi-join on columnar data; input: data, unique data; output: sorted positions, positions.";
    let description = [{
        Compares each element in the left-hand-side input column to each element in the right-hand-side input column.
        For each match, outputs the pair of positions of the matching elements in the left-hand-side input and
        right-hand-side input. The values the in right-hand-side input must be unique (no duplicates). The two output
        positions lists have the same length. When used for a primary-key foreign-key join, the left-hand-side input
        is the foreign key column and the right-hand-side input is the primary key column.
    }];

    let arguments = (ins ColumnOrU:$lhsData, ColumnOrU:$rhsData, Size:$numRes);
    let results = (outs ColumnOrU:$resLhsPos, ColumnOrU:$resRhsPos);
}

def Daphne_ColSemiJoinOp : Daphne_Op<"colSemiJoin", [
    DataTypeCol, ValueTypeSize
]> {
    let summary = "Semi-join on columnar data; input: data, data, output: sorted positions.";
    let description = [{
        Outputs the position of each element in the left-hand-side input column for which there is a matching element
        in the right-hand-side input column. The output positions are sorted.
    }];

    let arguments = (ins ColumnOrU:$lhsData, ColumnOrU:$rhsData, Size:$numRes);
    let results = (outs ColumnOrU:$resLhsPos);
}

def Daphne_ColGroupFirstOp : Daphne_Op<"colGroupFirst", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>
]> {
    let summary = "Initial grouping step on columnar data; input: data; output: positions (group ids), positions (of representatives).";
    let description = [{
        Maps each distinct value in the input data column to a unique group id. Group ids are consecutive integers
        starting at zero. Outputs (1) a column of group ids for each element in the input data column (same length as
        the input data column), and (2) a column of positions of one representative data element per group (the i-th
        element is the position of any input data element belonging to group id i). This operation is the first step
        when grouping by one or multiple columns.
    }];

    let arguments = (ins ColumnOrU:$argData);
    let results = (outs ColumnOrU:$resGrpIds, ColumnOrU:$resReprPos);
}

def Daphne_ColGroupNextOp : Daphne_Op<"colGroupNext", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>
]> {
    let summary = "Subsequent grouping step on columnar data; input: data, positions (group ids); output: positions (group ids), positions (of representatives).";
    let description = [{
        Maps each distinct combination of a value in the left-hand-side input data column and the right-hand-side group
        id column to a unique group id. The two input columns must have the same length. Group ids are consecutive
        integers starting at zero. Outputs (1) a column of group ids for each element in the input data column (same
        length as the input data column), and (2) a column of positions of one representative data element per group
        (the i-th element is the position of any input data element belonging to group id i). When grouping by multiple
        columns, this operation is applied after `colGroupFirst` to add one more key column to the grouping.
    }];

    let arguments = (ins ColumnOrU:$argData, ColumnOrU:$argGrpIds);
    let results = (outs ColumnOrU:$resGrpIds, ColumnOrU:$resReprPos);
}

class Daphne_ColCalcBinaryOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeCol, ValueTypeFromArgs
])> {
    let summary = "Elementwise binary operation on columnar data; input: data; output: data.";
    let description = [{
        Applies the given binary operation to all corresponding pairs in the two input data columns. The two input data
        columns must have the same length. The output data column has the same length as the inputs.
    }];

    // TODO Support scalars as arguments.
    let arguments = (ins ColumnOrU:$lhsData, ColumnOrU:$rhsData);
    let results = (outs ColumnOrU:$resData);
}
def Daphne_ColCalcSubOp : Daphne_ColCalcBinaryOp<"colCalcSub">;
def Daphne_ColCalcMulOp : Daphne_ColCalcBinaryOp<"colCalcMul">;

class Daphne_ColAllAggOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeCol, ValueTypeFromFirstArg
])> {
    let summary = "Full aggregation of columnar data; input: data; output: data.";
    let description = [{
        Aggregates all elements in the input data column using the given aggregation function. The result is a column
        containing a single data element.
    }];

    let arguments = (ins ColumnOrU:$arg);
    let results = (outs ColumnOrU:$res);
}
def Daphne_ColAllAggSumOp : Daphne_ColAllAggOp<"colSumAll">;

class Daphne_ColGrpAggOp<string name, list<Trait> traits = []>
: Daphne_Op<name, !listconcat(traits, [
    DataTypeCol, ValueTypeFromFirstArg
])> {
    let summary = "Grouped aggregation of columnar data; input: data, positions (group ids); output: data.";
    let description = [{
        Aggregates all elements in the input data column according to the input group ids using the specified
        aggregation function. The two input columns must have the same length. The input group ids are typically the
        output of the `colGroupFirst` or `colGroupNext` operation. The output column has one element per input group
        (the i-th element is the aggregate for group id i).
    }];

    let arguments = (ins ColumnOrU:$data, ColumnOrU:$groupIds, Size:$numDistinct, BoolScalar:$allowOptimisticSplitting);
    let results = (outs ColumnOrU:$res);
}
def Daphne_ColGrpAggSumOp : Daphne_ColGrpAggOp<"colSumGrp">;

// ****************************************************************************
// Frame label manipulation
// ****************************************************************************

def Daphne_SetColLabelsOp : Daphne_Op<"setColLabels", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    ShapeFromArg,
    Pure
]> {
    let arguments = (ins FrameOrU:$arg, Variadic<StrScalar>:$labels);
    let results = (outs FrameOrU:$res);

    let hasCanonicalizeMethod = 1;
}

def Daphne_SetColLabelsPrefixOp : Daphne_Op<"setColLabelsPrefix", [
    TypeFromFirstArg,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    ShapeFromArg
]> {
    let arguments = (ins FrameOrU:$arg, StrScalar:$prefix);
    let results = (outs FrameOrU:$res);
}

// ****************************************************************************
// Conversions and casts
// ****************************************************************************

def Daphne_ToStringOp : Daphne_Op<"toString", [DataTypeSca, ValueTypeStr]> {
    // TODO In the future, we might introduce more arguments, e.g. for
    // formatting a single value.
    let arguments = (ins MatrixOrFrame:$arg, StrScalar:$rowSep, StrScalar:$colSep);
    let results = (outs StrScalar:$res);
}

def Daphne_CastOp : Daphne_Op<"cast", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    ShapeFromArg,
    Pure
]> {
    // Note that the requested result type is not an argument, but should be
    // specified as the output type when creating a CastOp.
    let arguments = (ins AnyTypeOf<[MatrixOrFrame, ColumnOrU, AnyScalar, Unknown]>:$arg);
    let results = (outs AnyTypeOf<[MatrixOrFrame, ColumnOrU, AnyScalar, Unknown]>:$res);
    let hasFolder = 1;

    let extraClassDeclaration = [{
        bool isTrivialCast() {
            Type argTy = getArg().getType();
            Type resTy = getRes().getType();
            auto argFrmTy = llvm::dyn_cast<FrameType>(argTy);
            auto resFrmTy = llvm::dyn_cast<FrameType>(resTy);
            if(argFrmTy && resFrmTy) {
                // TODO avoid such a special case for frames, represent the optionality of labels in a different
                // way than a pointer (maybe llvm::Option/std::optional), sth that supports op== to be true on different instances
                if(argFrmTy.getColumnTypes() != resFrmTy.getColumnTypes())
                    return false;
                if(argFrmTy.getNumRows() != resFrmTy.getNumRows())
                    return false;
                if(argFrmTy.getNumCols() != resFrmTy.getNumCols())
                    return false;
                std::vector<std::string> *argLabels = argFrmTy.getLabels();
                std::vector<std::string> *resLabels = resFrmTy.getLabels();
                if((argLabels == nullptr) != (resLabels == nullptr))
                    return false;
                if(argLabels && *argLabels != *resLabels)
                    return false;
                return true;
            } else
                return argTy == resTy;
        }

        /**
         * @brief Checks if this cast just removes detailed properties from a matrix/frame type,
         * to make them unknown instead.
         *
         * Usually casts like this need to exist to make the IR correct for SCF operations.
         * @return true if this cast just tags some known information unknown, false otherwise
         */
        bool isRemovePropertyCast() {
            Type argTy = getArg().getType();
            Type resTy = getRes().getType();
            auto argMatTy = llvm::dyn_cast<daphne::MatrixType>(argTy);
            auto resMatTy = llvm::dyn_cast<daphne::MatrixType>(resTy);
            auto argFrmTy = llvm::dyn_cast<daphne::FrameType>(argTy);
            auto resFrmTy = llvm::dyn_cast<daphne::FrameType>(resTy);
            return (argMatTy && resMatTy && argMatTy.isSpecializationOf(resMatTy))
                || (argFrmTy && resFrmTy && argFrmTy.isSpecializationOf(resFrmTy));
        }

        bool mightLoseInformation() {
            Type argTy = getArg().getType();
            Type resTy = getRes().getType();

            auto argMatTy = llvm::dyn_cast<daphne::MatrixType>(argTy);
            auto argFrmTy = llvm::dyn_cast<daphne::FrameType>(argTy);
            auto argColTy = llvm::dyn_cast<daphne::ColumnType>(argTy);

            auto resMatTy = llvm::dyn_cast<daphne::MatrixType>(resTy);
            auto resFrmTy = llvm::dyn_cast<daphne::FrameType>(resTy);
            auto resColTy = llvm::dyn_cast<daphne::ColumnType>(resTy);

            // Note: We only consider data/value types, but not the properties.
            // TODO What if "b" has more property info that would be important for further inference (is that possible)?

            // If we cast from a matrix to a frame and all columns of the frame have the value type of the matrix, then we certainly do not lose information.
            if(argMatTy && resFrmTy) {
                Type argMatValTy = argMatTy.getElementType();
                std::vector resFrmColTys = resFrmTy.getColumnTypes();
                for (Type resFrmColTy : resFrmColTys)
                    // TODO It would be okay if resFrmColTy can represent all values of argMatValTy.
                    if (argMatValTy != resFrmColTy)
                        return true;
                return false;
            }
            // If we cast from a frame to a matrix and all columns of the frame have the value type of the matrix, then we certainly do not lose information.
            if(argFrmTy && resMatTy) {
                std::vector argFrmColTys = argFrmTy.getColumnTypes();
                Type resMatValTy = resMatTy.getElementType();
                for (Type argFrmColTy : argFrmColTys)
                    // TODO It would be okay if resMatValTy can represent all values of argFrmColTy.
                    if (argFrmColTy != resMatValTy)
                        return true;
                return false;
            }
            // If we cast from a matrix to a column and both have the same value type, then we certainly do not lose information.
            if (argMatTy && resColTy && argMatTy.getElementType() == resColTy.getValueType())
                return false;
            // If we cast from a column to a matrix and both have the same value type, then we certainly do not lose information.
            if (argColTy && resMatTy && argColTy.getValueType() == resMatTy.getElementType())
                return false;
            // If we cast from a column to a single-column frame and both have the same value type, then we certainly do not lose information.
            if (argColTy && resFrmTy && argColTy.getValueType() == resFrmTy.getColumnTypes()[0])
                return false;
            // If we cast from a single-column frame to a column and both have the same value type, then we certainly do not lose information.
            if (argFrmTy && resColTy && argFrmTy.getColumnTypes()[0] == resColTy.getValueType())
                return false;

            return true;
        }
    }];

    let hasCanonicalizeMethod = 1;
}

def Daphne_RenameOp : Daphne_Op<"rename", [
    TypeFromFirstArg
]> {
    let arguments = (ins AnyType:$arg);
    let results = (outs AnyType:$res);
    let hasCanonicalizeMethod = 1;
}

def Daphne_QuantizeOp : Daphne_Op<"quantize", [
    ShapeFromArg
]> {
    let arguments = (ins MatrixOrU:$arg, NumScalar:$min, NumScalar:$max);
    let results = (outs MatrixOrU:$res);
}

def Daphne_GetColIdxOp : Daphne_Op<"getColIdx", [DataTypeSca, ValueTypeSize]>{
    let arguments = (ins Frame:$frame, StrScalar:$columnName);
    let results = (outs Size:$res);
}

def Daphne_ConvertPosListToBitmapOp : Daphne_Op<"convertPosListToBitmap", [
    TypeFromFirstArg,
    ShapeFromArg,
    Pure
]> {
    let summary = "Creates a single-column matrix of zero/one entries (bit vector), where the entries addressed by the numbers in the single-column input matrix (position list) are set to one.";

    let arguments = (ins MatrixOrU:$arg, Size:$numRowsRes);
    let results = (outs MatrixOrU:$res);
}

def Daphne_ConvertBitmapToPosListOp : Daphne_Op<"convertBitmapToPosList", [
    DataTypeMat, ValueTypeSize,
    ShapeFromArg,
    Pure
]> {
    let summary = "Given a single-column matrix with only zero/one entries (bit vector), creates a sorted single-column matrix of the positions of ones in the input (position list).";

    let arguments = (ins MatrixOrU:$arg);
    let results = (outs MatrixOrU:$res);

    let hasCanonicalizeMethod = 1;
}

// ****************************************************************************
// Distributed Operations
// ****************************************************************************

def Daphne_DistributedReadOp : Daphne_Op<"distributedRead", [Pure]> {
    let arguments = (ins StrScalar:$fileName);
    let results = (outs Handle:$res);
}

def Daphne_DistributeOp : Daphne_Op<"distribute", [Pure]> {
    let arguments = (ins MatrixOrU:$mat);
    let results = (outs Handle:$res);

    let hasCanonicalizer = 1;
}

def Daphne_BroadcastOp : Daphne_Op<"broadcast", [Pure]> {
    let arguments = (ins MatrixOrU:$mat);
    let results = (outs Handle:$res);
}

def Daphne_DistributedComputeOp : Daphne_Op<"distributedCompute", [Pure, IsolatedFromAbove]> {
    let arguments = (ins Variadic<Handle>:$args);
    let results = (outs Variadic<Handle>:$res);
    let regions = (region SizedRegion<1>:$body);
}

// TODO: Map Keys Operation

def Daphne_DistributedCollectOp : Daphne_Op<"distributedCollect", [Pure]> {
    let arguments = (ins Handle:$arg);
    let results = (outs AnyType:$res);

    let builders = [
        OpBuilder<(ins "Value":$value), [{
            build($_builder, $_state, llvm::cast<HandleType>(value.getType()).getDataType(), value);
        }]>,
    ];
}

// ****************************************************************************
// Input/output
// ****************************************************************************

// ----------------------------------------------------------------------------
// High-level
// ----------------------------------------------------------------------------

def Daphne_PrintOp : Daphne_Op<"print"> {
    // TODO We might change it to only accept scalars here and enforce toString
    // for matrices and frames. But currently, we need it like that for the
    // rest of the program.
    let arguments = (ins AnyTypeOf<[AnyScalar, MatrixOrFrame, List, AnyMemRef, Unknown]>:$arg, BoolScalar:$newline, BoolScalar:$err);
    let results = (outs); // no results
}

// TODO Take asynchronous read into account.
def Daphne_ReadOp : Daphne_Op<"read", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>,
    DeclareOpInterfaceMethods<InferFrameLabelsOpInterface>,
    DeclareOpInterfaceMethods<InferSparsityOpInterface>,
    DeclareOpInterfaceMethods<InferMncSketchIdOpInterface>
]> {
    // TODO We might add arguments for a UDF later.
    let arguments = (ins StrScalar:$fileName);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_WriteOp : Daphne_Op<"write"> {
    let arguments = (ins MatrixOrFrame:$arg, StrScalar:$fileName);
    let results = (outs); // no results
}

def Daphne_ReceiveFromNumpyOp: Daphne_Op<"receiveFromNumpy">{
    let arguments = (ins UI64: $address, SI64:$rows, SI64:$cols);
    let results = (outs MatrixOrU:$res);
}

def Daphne_SaveDaphneLibResultOp : Daphne_Op<"saveDaphneLibResult"> {
    let arguments = (ins MatrixOrFrame:$arg);
    let results = (outs); // no results
}

def Daphne_StopOp : Daphne_Op<"stop"> {
    let arguments = (ins StrScalar:$message);
    let results = (outs); // no results
}

// ----------------------------------------------------------------------------
// Low-level
// ----------------------------------------------------------------------------

def Daphne_OpenFileOp : Daphne_Op<"openFile"> {
    let arguments = (ins StrScalar:$filename);
    let results = (outs File:$res);
}

def Daphne_OpenDeviceOp : Daphne_Op<"openDevice"> {
    let arguments = (ins StrScalar:$device);
    let results = (outs Target:$res);
}

def Daphne_OpenFileOnTargetOp : Daphne_Op<"openFileOnTarget"> {
    let arguments = (ins Target:$target, StrScalar:$filename);
    let results = (outs Descriptor:$res);
}

def Daphne_CloseOp : Daphne_Op<"close"> {
    let arguments = (ins AnyTypeOf<[File, Target, Unknown]>);
    let results = (outs); // no results
}

def Daphne_ReadCsvOp : Daphne_Op<"readCsv"> {
    let arguments = (ins
        AnyTypeOf<[File, Descriptor, Unknown]>:$input,
        Size:$numRows, Size:$numCols,
        StrScalar:$delim
    );
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Data preprocessing
// ****************************************************************************

def Daphne_OneHotOp : Daphne_Op<"oneHot", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    // TODO We can know the exact number of columns at compile-time if $info is
    // known at compile-time (e.g. if it is a matrix literal).
    NumRowsFromArg, // DeclareOpInterfaceMethods<InferNumColsOpInterface>
]> {
    let arguments = (ins MatrixOrFrame:$arg, MatrixOf<[SI64]>:$info);
    let results = (outs MatrixOrFrame:$res);
}

def Daphne_RecodeOp : Daphne_Op<"recode", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>,
    DeclareOpInterfaceMethods<InferShapeOpInterface>
]> {
    let arguments = (ins MatrixOrFrame:$arg, BoolScalar:$orderPreserving);
    let results = (outs MatrixOrFrame:$res, MatrixOrFrame:$dict);
}

def Daphne_BinOp : Daphne_Op<"bin", [
    TypeFromFirstArg,
    ShapeFromArg
]> {
    let arguments = (ins MatrixOrFrame:$arg, NumScalar:$numBins, NumScalar:$min, NumScalar:$max);
    let results = (outs MatrixOrFrame:$res);
}

// ****************************************************************************
// Measurements
// ****************************************************************************

def Daphne_NowOp : Daphne_Op<"now", [DataTypeSca, ValueTypeSI64]> {
    let summary = "Returns the current time since the epoch in nano seconds.";
    let description = [{
        Take care if you use this for time measurements! This is currently just
        a quick and somewhat dirty approach. Once we have compiler passes that
        could reorder operations, you might not measure the time you expect
        anymore!
    }];

    let arguments = (ins); // no arguments
    let results = (outs SI64);
}

// ****************************************************************************
// Context handling
// ****************************************************************************

def Daphne_CreateDaphneContextOp : Daphne_Op<"createDaphneContext"> {
    let summary = "Creates an instance of a Daphne run-time context.";

    let description = [{
        The newly created instance might be partially initialized. The
        initialization of certain parts/aspects of the context (which are not
        always relevant) might require additional DaphneIR operations.
    }];

    let arguments = (ins
                    UI64 : $userConfigPtr,
                    UI64 : $dispatchMappingPtr,
                    UI64 : $statisticsPtr,
                    UI64 : $propertyLoggerPtr,
                    UI64 : $stringRefCountPtr
                    );
    let results = (outs DaphneContext : $ctx);
}

def Daphne_DestroyDaphneContextOp : Daphne_Op<"destroyDaphneContext"> {
    let summary = "Destroys an instance of a Daphne run-time context.";

    // Note that the DaphneContext is automatically injected during the
    // lowering to kernel calls.
    let arguments = (ins); // no arguments
    let results = (outs); // no results
}


def Daphne_CreateCUDAContextOp : Daphne_Op<"createCUDAContext", [CUDASupport]> {
    let arguments = (ins);
    let results = (outs);
}

def Daphne_CreateDistributedContextOp : Daphne_Op<"createDistributedContext", []> {
    let arguments = (ins);
    let results = (outs);
}

def Daphne_CreateHDFSContextOp : Daphne_Op<"createHDFSContext", []> {
    let arguments = (ins);
    let results = (outs);
}

def Daphne_CreateFPGAContextOp : Daphne_Op<"createFPGAContext", [FPGAOPENCLSupport]> {
    let arguments = (ins);
    let results = (outs);
}


// ****************************************************************************
// Vectorized operations
// ****************************************************************************

// TODO: should be recursive side effects
// TODO: use symbol handling (should be isolated from above, but needs reference to daphne context)
def Daphne_VectorizedPipelineOp : Daphne_Op<"vectorizedPipeline", [AttrSizedOperandSegments]> {
    let summary = "Performs multi-threaded vector-wise computations of the input.";

    // Unlike most other DaphneIR ops, this op explicitly needs the DaphneContext as an argument, because it is not
    // lowered to CallKernelOp in RewriteToCallKernelOpsPass but in LowerToLLVMPass, where an extraction of the
    // DaphneContext from the IR is not possible/convenient anymore.
    let arguments = (ins Variadic<AnyTypeOf<[AnyScalar, MatrixOrFrame, Unknown]>>:$inputs,
            // TODO: both types should be SIZE, but we require `-1` for dynamic/unknown
            Variadic<AnyTypeOf<[SIntScalar, Size, Unknown]>>:$out_rows,
            Variadic<AnyTypeOf<[SIntScalar, Size, Unknown]>>:$out_cols,
            TypedArrayAttrBase<VectorSplitAttr, "Vector-Splits">:$splits,
            TypedArrayAttrBase<VectorCombineAttr, "Vector-Combines">:$combines,
            Optional<DaphneContext>:$ctx);
    let results = (outs Variadic<MatrixOrFrame>:$outputs);
    let regions = (region SizedRegion<1>:$body, AnyRegion:$cuda);

    let hasCanonicalizeMethod = 1;
}

def Daphne_DistributedPipelineOp : Daphne_Op<"distributedPipeline", [AttrSizedOperandSegments]> {
    let summary = "...";

    let arguments = (ins
            StrScalar:$ir,
            Variadic<AnyTypeOf<[AnyScalar, MatrixOrFrame]>>:$inputs,
            // TODO: both types should be SIZE, but we require `-1` for dynamic/unknown
            Variadic<AnyTypeOf<[SIntScalar, Size]>>:$out_rows,
            Variadic<AnyTypeOf<[SIntScalar, Size]>>:$out_cols,
            TypedArrayAttrBase<VectorSplitAttr, "Vector-Splits">:$splits,
            TypedArrayAttrBase<VectorCombineAttr, "Vector-Combines">:$combines
    );
    let results = (outs Variadic<MatrixOrFrame>:$outputs);
}

// ****************************************************************************
// Higher-order operations
// ****************************************************************************

def Daphne_MapOp : Daphne_Op<"map", [ShapeFromArg]> {
    let summary = "Applies a user-defined function to elements of a matrix.";
    // Unlike most other DaphneIR ops, this op explicitly needs the DaphneContext as an argument, because it is not
    // lowered to CallKernelOp in RewriteToCallKernelOpsPass but in LowerToLLVMPass, where an extraction of the
    // DaphneContext from the IR is not possible/convenient anymore.
    let arguments = (ins MatrixOrU:$arg, SymbolNameAttr:$func, Optional<DaphneContext>:$ctx);
    let results = (outs MatrixOrU:$res);
}


// ****************************************************************************
// Low-level auxiliary operations
// ****************************************************************************

def Daphne_CreateVariadicPackOp : Daphne_Op<"createVariadicPack"> {
    let arguments = (ins I64Attr:$numElements);
    let results = (outs VariadicPack:$res);
}

def Daphne_StoreVariadicPackOp : Daphne_Op<"storeVariadicPack"> {
    let arguments = (ins VariadicPack:$pack, AnyType:$item, I64Attr:$pos);
    let results = (outs); // no results
}

def Daphne_IncRefOp : Daphne_Op<"incRef"> {
    let summary = "Increases the reference counter of the underlying runtime data object.";

    let arguments = (ins AnyTypeOf<[MatrixOrFrameOrString, List]>:$arg);
    let results = (outs); // no results
}

def Daphne_DecRefOp : Daphne_Op<"decRef"> {
    let summary = "Decreases the reference counter of the underlying runtime data object and frees it if the reference counter becomes zero.";

    let arguments = (ins AnyTypeOf<[MatrixOrFrameOrString, List]>:$arg);
    let results = (outs); // no results
}

// ****************************************************************************
// Profiling
// ****************************************************************************

def Daphne_StartProfilingOp : Daphne_Op<"startProfiling"> {
    let summary = "Starts profiling for the given region name";

    let arguments = (ins StrScalar:$regionName);
    let results = (outs); // no results
}

def Daphne_StopProfilingOp : Daphne_Op<"stopProfiling"> {
    let summary = "Stops profiling for the given region name";

    let arguments = (ins StrScalar:$regionName);
    let results = (outs); // no results
}

// ****************************************************************************
// List operations
// ****************************************************************************

def Daphne_CreateListOp : Daphne_Op<"createList", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>
]> {
    let summary = "Creates a new list from the given elements";

    let arguments = (ins Variadic<MatrixOrU>:$elems);
    let results = (outs ListOrU:$res);
}

def Daphne_LengthOp : Daphne_Op<"length", [
    DataTypeSca, ValueTypeSize
]> {
    let summary = "Returns the number of elements in the given list";
    
    let arguments = (ins ListOrU:$arg);
    let results = (outs Size:$res);
}

def Daphne_AppendOp : Daphne_Op<"append", [
    TypeFromFirstArg
]> {
    let summary = "Appends the given element to the end of the given list";
    
    let arguments = (ins ListOrU:$argList, MatrixOrU:$elem);
    let results = (outs ListOrU:$resList);
}

def Daphne_RemoveOp : Daphne_Op<"remove", [
    DeclareOpInterfaceMethods<InferTypesOpInterface>
]> {
    let summary = "Removes and returns the element at the specified index from the given list";
    
    let arguments = (ins ListOrU:$argList, Size:$idx);
    let results = (outs ListOrU:$resList, MatrixOrU:$elem);
}

// ****************************************************************************
// Handling of data properties
// ****************************************************************************

def Daphne_RecordPropertiesOp : Daphne_Op<"recordProperties"> {
    let arguments = (ins MatrixOrU:$arg, UI32:$valueId);
    let results = (outs);

    let summary = "Records the data properties of the given matrix.";
}

def Daphne_TransferPropertiesOp : Daphne_Op<"transferProperties"> {
    let summary = "Transfers compile-time information on data properties to the runtime data objects";

    let arguments = (ins MatrixOrU:$arg, F64:$sparsity, SI64:$symmetric, SI32:$mncSketchId);
    let results = (outs);

    let hasCanonicalizeMethod = 1;
}

// ****************************************************************************
// Old operations
// ****************************************************************************
// These are currently still need for the prototype to work. They will be
// removed/replaced/changed in the future.

def Daphne_ConstantOp : Daphne_Op<"constant", [ConstantLike, Pure]> {
    let summary = "constant operation";

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyTypeOf<[AnyScalar, Matrix, Unknown]>:$result);

    let builders = [
        OpBuilder<(ins "double":$value), [{
            build($_builder, $_state, $_builder.getF64Type(), $_builder.getF64FloatAttr(value));
        }]>,
        OpBuilder<(ins "float":$value), [{
            build($_builder, $_state, $_builder.getF32Type(), $_builder.getF32FloatAttr(value));
        }]>,
        OpBuilder<(ins "int64_t":$value), [{
            Type t = $_builder.getIntegerType(64, true);
            build($_builder, $_state, t, $_builder.getIntegerAttr(t, value));
        }]>,
        OpBuilder<(ins "uint32_t":$value), [{
            Type t = $_builder.getIntegerType(32, false);
            build($_builder, $_state, t, $_builder.getIntegerAttr(t, value));
        }]>,
        OpBuilder<(ins "uint64_t":$value), [{
            Type t = $_builder.getIntegerType(64, false);
            build($_builder, $_state, t, $_builder.getIntegerAttr(t, value));
        }]>,
        OpBuilder<(ins "bool":$value), [{
            Type t = $_builder.getI1Type();
            // TODO Why not getBoolAttr?
            build($_builder, $_state, t, $_builder.getIntegerAttr(t, value));
        }]>,
        //OpBuilder<(ins "int":$value), [{
        //    build($_builder, $_state, $_builder.getI32IntegerAttr(value));
        //}]>,
        OpBuilder<(ins "std::string":$value), [{
            build($_builder, $_state, mlir::daphne::StringType::get($_builder.getContext()), $_builder.getStringAttr(value));
        }]>,
    ];

    let hasFolder = 1;
}

def Daphne_ReturnOp : Daphne_Op<"return", [Pure, Terminator, ReturnLike, ParentOneOf<["func::FuncOp", "DistributedComputeOp", "VectorizedPipelineOp"]>]> {
    let summary = "return operation";

    let arguments = (ins Variadic<AnyType>:$operands);

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, std::nullopt);
        }]>
    ];
}

def Daphne_CallKernelOp : Daphne_Op<"call_kernel"> {
    let summary = "kernel operation for C";

    let arguments = (ins StrAttr:$callee, Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>);

    let builders = [
        OpBuilder<(ins "StringAttr":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(results);
        }]>,
        OpBuilder<(ins "StringRef":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            build($_builder, $_state, $_builder.getStringAttr(callee), operands, results);
        }]>,
    ];
}

def Daphne_GenericCallOp : Daphne_Op<"generic_call", [DeclareOpInterfaceMethods<InferTypesOpInterface>]> {
    let summary = "User defined function call";

    let arguments = (ins SymbolNameAttr:$callee, Variadic<AnyType>:$inputs);
    let results = (outs Variadic<AnyType>);

    let builders = [
        OpBuilder<(ins "StringAttr":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(results);
        }]>,
        OpBuilder<(ins "StringRef":$callee, CArg<"ValueRange", "{}">:$operands, CArg<"TypeRange", "{}">:$results), [{
            build($_builder, $_state, $_builder.getStringAttr(callee), operands, results);
        }]>,
    ];
}

class Daphne_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
        Daphne_Op<mnemonic, !listconcat(traits, [Pure, TypesMatchOrOneIsMatrixOfOther<"lhs", "rhs">])> {
    let arguments = (ins AnyTypeOf<[AnyScalar, Matrix, Unknown]>:$lhs, AnyTypeOf<[AnyScalar, Matrix, Unknown]>:$rhs);
    let results = (outs AnyTypeOf<[AnyScalar, Matrix, Unknown]>:$result);

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            $_state.addTypes(lhs.getType());
            $_state.addOperands({lhs, rhs});
        }]>
    ];
}

def Daphne_AddOp : Daphne_ElementwiseBinaryOp<"addOld", [SameOperandsAndResultShape]> {
    let summary = "add operation";
}

def Daphne_SubOp : Daphne_ElementwiseBinaryOp<"subOld", [SameOperandsAndResultShape]> {
    let summary = "subtraction operation";
}

def Daphne_MulOp : Daphne_ElementwiseBinaryOp<"mulOld", [SameOperandsAndResultShape]> {
    let summary = "mul operation";
}

def Daphne_SetCellOp : Daphne_Op<"setCell",  []> {
    let summary = "set cell operation";

    let arguments = (ins MatrixOrU:$mat, Index:$row, Index:$col, AnyScalar:$val);
}

#endif //SRC_IR_DAPHNEIR_DAPHNEOPS_TD
