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

#ifndef SRC_IR_DAPHNEIR_DAPHNETYPES_TD
#define SRC_IR_DAPHNEIR_DAPHNETYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

// ****************************************************************************
// General
// ****************************************************************************

// The base for all DaphneIR-specific types.
class Daphne_Type<string name> : TypeDef<Daphne_Dialect, name> {
    let typeName = name;
    let mnemonic = name;
}

// ****************************************************************************
// Unknown type
// ****************************************************************************

def Unknown : Daphne_Type<"Unknown"> {
    let summary = "placeholder for an unknown type";
    let description = [{
        This type is meant to be used in situations when a type is not (yet)
        known. In some cases, it could be replaced by a known type in a
        compiler pass. In other cases, it might not be known at compile-time at
        all and the run-time needs to deal with it.
    }];
}

// ****************************************************************************
// Data types
// ****************************************************************************

def Structure : Daphne_Type<"Structure"> {
    // Don't use this in TableGen!
    let summary = "structure";
}

// A matrix type.
def Matrix : Daphne_Type<"Matrix"> {
    let summary = "matrix";
    let hasCustomAssemblyFormat = 1;

    // TODO Maybe we should rename "element type" to "value type" everywhere.
    let parameters = (ins
        "::mlir::Type":$elementType,
        "ssize_t":$numRows, "ssize_t":$numCols, "double":$sparsity, "MatrixRepresentation":$representation, "BoolOrUnknown":$symmetric
    );
    let genVerifyDecl = 1;
    // NOTE: float needs special treatment for equality check and hashing (epsilon)
    let genStorageClass = 0;

    let builders = [
        // Creates a MatrixType from mere element type information, with all
        // other parameters reset.
        TypeBuilder<(ins "::mlir::Type":$elementType), [{
            return Base::get($_ctxt, elementType, -1, -1, -1.0, ::mlir::daphne::MatrixRepresentation::Default, ::mlir::daphne::BoolOrUnknown::Unknown);
        }]>,
    ];

    let extraClassDeclaration = [{
        // The following methods return a MatrixType with the same parameters
        // as this MatrixType, except for one parameter, which is replaced by a
        // new value.

        ::mlir::daphne::MatrixType withElementType(::mlir::Type elementType) {
            return get(getContext(), elementType, getNumRows(), getNumCols(), getSparsity(), getRepresentation(), getSymmetric());
        }

        ::mlir::daphne::MatrixType withShape(ssize_t numRows, ssize_t numCols) {
            return get(getContext(), getElementType(), numRows, numCols, getSparsity(), getRepresentation(), getSymmetric());
        }

        ::mlir::daphne::MatrixType withSparsity(double sparsity) {
            return get(getContext(), getElementType(), getNumRows(), getNumCols(), sparsity, getRepresentation(), getSymmetric());
        }

        ::mlir::daphne::MatrixType withRepresentation(::mlir::daphne::MatrixRepresentation rep) {
            return get(getContext(), getElementType(), getNumRows(), getNumCols(), getSparsity(), rep, getSymmetric());
        }

        ::mlir::daphne::MatrixType withSymmetric(BoolOrUnknown symmetric) {
            return get(getContext(), getElementType(), getNumRows(), getNumCols(), getSparsity(), getRepresentation(), symmetric);
        }

        // The following methods return a MatrixType preserving the specified
        // properties and resetting all others.

        ::mlir::daphne::MatrixType withSameElementType() {
            return get(getContext(), getElementType());
        }

        ::mlir::daphne::MatrixType withSameElementTypeAndRepr() {
            return get(getContext(), getElementType(), -1, -1, -1.0, getRepresentation(), ::mlir::daphne::BoolOrUnknown::Unknown);
        }

        /**
         * @brief Check if `other` has strictly less information than this matrix type.
         * @param other the other matrix type to compare to
         * @return true if this type has more information, false otherwise
         */
        bool isSpecializationOf(::mlir::daphne::MatrixType other) {
            if (!llvm::isa<mlir::daphne::UnknownType>(other.getElementType()) && getElementType() != other.getElementType())
                return false;

            if (other.getNumRows() != -1 && getNumRows() != other.getNumRows())
                return false;

            if (other.getNumCols() != -1 && getNumCols() != other.getNumCols())
                return false;

            if (other.getSparsity() != -1.0 && getSparsity() != other.getSparsity())
                return false;

            if(other.getRepresentation() != ::mlir::daphne::MatrixRepresentation::Default
                && getRepresentation() != other.getRepresentation())
                return false;

            if (other.getSymmetric() != ::mlir::daphne::BoolOrUnknown::Unknown
                && getSymmetric() != other.getSymmetric())
                return false;

            return true;
        }
    }];
}

// TODO We could omit the "Of" and have it default to ScalarType.
// A type constraint checking if a type is a matrix whose value type is one of
// the given types (whereby Unknown is always allowed automatically).
// Reuses MLIR's ContainerType constraint in a way inspired by MLIR's VectorOf.
class MatrixOf<list<Type> allowedTypes> : AnyTypeOf<[
        ContainerType<
                AnyTypeOf<!listconcat(allowedTypes, [Unknown])>,
                Matrix.predicate,
                "mlir::dyn_cast<::mlir::daphne::MatrixType>($_self).getElementType()",
                "matrix"
        >,
        Unknown
]>;

def Frame : Daphne_Type<"Frame"> {
    let summary = "frame";
    let hasCustomAssemblyFormat = 1;

    let parameters = (ins
        "std::vector<::mlir::Type>":$columnTypes,
        "ssize_t":$numRows, "ssize_t":$numCols,
        "std::vector<std::string> *":$labels
    );

    let genVerifyDecl = 1;

    let builders = [
        // Creates a FrameType from mere column type information, with all
        // other parameters reset.
        TypeBuilder<(ins "std::vector<::mlir::Type>":$columnTypes), [{
            return Base::get($_ctxt, columnTypes, -1, -1, nullptr);
        }]>,
    ];

    let extraClassDeclaration =  [{
        // The following methods return a FrameType with the same parameters
        // as this FrameType, except for one parameter, which is replaced by a
        // new value.

        ::mlir::daphne::FrameType withColumnTypes(std::vector<::mlir::Type> columnTypes) {
            return get(getContext(), columnTypes, getNumRows(), getNumCols(), getLabels());
        }

        ::mlir::daphne::FrameType withShape(ssize_t numRows, ssize_t numCols) {
            return get(getContext(), getColumnTypes(), numRows, numCols, getLabels());
        }

        ::mlir::daphne::FrameType withLabels(std::vector<std::string> * labels) {
            return get(getContext(), getColumnTypes(), getNumRows(), getNumCols(), labels);
        }

        // The following method returns a FrameType with the same column types
        // as this FrameType, but all other parameters reset.

        ::mlir::daphne::FrameType withSameColumnTypes() {
            return get(getContext(), getColumnTypes());
        }

        /**
         * @brief Check if `other` has strictly less information than this frame type.
         * @param other the other frame type to compare to
         * @return true if this type has more information, false otherwise
         */
        bool isSpecializationOf(::mlir::daphne::FrameType other) {
            // TODO #421 needs to be taken into account here, once we address it.
            auto columnTypes = getColumnTypes();
            auto otherColumnTypes = other.getColumnTypes();
            // Normally, this size() should always be the same as numCols.
            if (columnTypes.size() != otherColumnTypes.size())
                return false;
            for (size_t i = 0; i < columnTypes.size(); i++)
                if(!llvm::isa<mlir::daphne::UnknownType>(otherColumnTypes[i]) && columnTypes[i] != otherColumnTypes[i])
                    return false;

            if (other.getNumRows() != -1 && getNumRows() != other.getNumRows())
                return false;

            if (other.getNumCols() != -1 && getNumCols() != other.getNumCols())
                return false;

            // Note that this compares the labels pointers, not the labels themselves.
            // This is to ensure consistency with other places, especially when
            // comparing two mlir::Type/mlir::daphne::FrameType instances using operator==.
            if (other.getLabels() != nullptr && getLabels() != other.getLabels())
                return false;

            return true;
        }
    }];
}

def MatrixOrFrame : AnyTypeOf<[Matrix, Frame, Unknown]>;
// TODO It would be great to get rid of the "unknown" in the names. Ideally, we
// would like to allow unknown without explicitly specifying it.
def MatrixOrU : AnyTypeOf<[Matrix, Unknown]>;
def FrameOrU : AnyTypeOf<[Frame, Unknown]>;

def Column : Daphne_Type<"Column"> {
    let summary = "column";
    let hasCustomAssemblyFormat = 1;

    let parameters = (ins "::mlir::Type":$valueType, "ssize_t":$numRows);

    let genVerifyDecl = 1;

    let builders = [
        // Creates a ColumnType from mere value type information, with all other parameters reset.
        TypeBuilder<(ins "::mlir::Type":$valueType), [{
            return Base::get($_ctxt, valueType, -1);
        }]>,
    ];

    let extraClassDeclaration =  [{
        // The following methods return a ColumnType with the same parameters
        // as this ColumnType, except for one parameter, which is replaced by a
        // new value.

        ::mlir::daphne::ColumnType withValueType(::mlir::Type valueType) {
            return get(getContext(), valueType, getNumRows());
        }

        ::mlir::daphne::ColumnType withShape(ssize_t numRows) {
            return get(getContext(), getValueType(), numRows);
        }

        // The following method returns a ColumnType with the same value type
        // as this ColumnType, but all other parameters reset.

        ::mlir::daphne::ColumnType withSameValueType() {
            return get(getContext(), getValueType());
        }

        /**
         * @brief Check if `other` has strictly less information than this column type.
         * @param other the other column type to compare to
         * @return true if this type has more information, false otherwise
         */
        bool isSpecializationOf(::mlir::daphne::ColumnType other) {
            auto valueType = getValueType();
            auto otherValueType = other.getValueType();
            if(!llvm::isa<mlir::daphne::UnknownType>(otherValueType) && valueType != otherValueType)
                return false;

            if (other.getNumRows() != -1 && getNumRows() != other.getNumRows())
                return false;

            return true;
        }
    }];
}
def ColumnOrU : AnyTypeOf<[Column, Unknown]>;

// TODO We could omit the "Of" and have it default to ScalarType.
// A type constraint checking if a type is a column whose value type is one of
// the given types (whereby Unknown is always allowed automatically).
// Reuses MLIR's ContainerType constraint in a way inspired by MLIR's VectorOf.
class ColumnOf<list<Type> allowedTypes> : AnyTypeOf<[
        ContainerType<
                AnyTypeOf<!listconcat(allowedTypes, [Unknown])>,
                Column.predicate,
                "$_self.dyn_cast<::mlir::daphne::ColumnType>().getValueType()",
                "column"
        >,
        Unknown
]>;

def List : Daphne_Type<"List"> {
    let summary = "list";
    let hasCustomAssemblyFormat = 1;

    let parameters = (ins
        "::mlir::Type":$elementType
    );
}

def ListOrU : AnyTypeOf<[List, Unknown]>;

// ****************************************************************************
// Distributed types
// ****************************************************************************

def Handle : Daphne_Type<"Handle"> {
    let summary = "handle to distributed data";
    let hasCustomAssemblyFormat = 1;

    let parameters = (ins "::mlir::Type":$dataType);
    let genVerifyDecl = 1;
}

// ****************************************************************************
// Value types
// ****************************************************************************

// A character string type.
def StrScalar : Daphne_Type<"String"> {
    let summary = "string";
}
def MatrixOrFrameOrString : AnyTypeOf<[Matrix, Frame, StrScalar, Unknown]>;

def SIntScalar : AnyTypeOf<[SI8, SI32, SI64, Unknown], "signed integer">;
def UIntScalar : AnyTypeOf<[UI8, UI32, UI64, Unknown], "unsigned integer">;
def IntScalar : AnyTypeOf<[SIntScalar, UIntScalar], "integer">;
def FloatScalar : AnyTypeOf<[F32, F64, Unknown], "float">;
def NumScalar : AnyTypeOf<[IntScalar, FloatScalar, Index], "numeric">;
def BoolScalar : TypeAlias<I1, "boolean">;

// TODO Exclude Index again.
def AnyScalar : AnyTypeOf<[NumScalar, BoolScalar, StrScalar, Index]>;

// ****************************************************************************
// IO types
// ****************************************************************************

def File : Daphne_Type<"File"> {
    let summary = "An open file.";
}

def Descriptor : Daphne_Type<"Descriptor"> {
    let summary = "An open file on a device.";
}

def Target : Daphne_Type<"Target"> {
    let summary = "An open device.";
}

// ****************************************************************************
// Auxiliary types
// ****************************************************************************

// TODO Could also be Index or whatever, but it might be wise to define the
// type we use to express dimensions and numeric row/col indexes just once.
// TODO Rename, it's misleading. Sounds like a pair of height/width.
def Size : TypeAlias<Index, "size">;

def Seed : TypeAlias<SI64, "seed">;

// TODO Remove SI64 again, it was only added temporarily to make things work.
def Selection : AnyTypeOf<[Size, MatrixOf<[Size, SI64]>, Unknown]>;

def VariadicPack : Daphne_Type<"VariadicPack"> {
    let summary = "pack of variadic operands";
    let description = [{
        This type is used to represent a variadic operand of an operation as a
        single SSA value irrespective of the number of concrete operands. It
        is meant to exist only in the low levels of the Daphne compilation
        stack. It is introduced during the lowering to kernel calls and
        converted when lowering to the LLVM dialect.
    }];
    let hasCustomAssemblyFormat = 1;
    let parameters = (ins "::mlir::Type":$containedType);
}

def DaphneContext : Daphne_Type<"DaphneContext"> {
    let summary = "daphne context";
    let description = [{
        This type represents all kinds of run-time context information in one
        SSA value. This could be things like available hardware resources and
        global run-time facilities.
    }];
}

#endif //SRC_IR_DAPHNEIR_DAPHNETYPES_TD
