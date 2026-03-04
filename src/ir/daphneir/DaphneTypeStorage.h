/*
 * Copyright 2025 The DAPHNE Consortium
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

#ifndef SRC_IR_DAPHNEIR_DAPHNETYPESTORAGE_H
#define SRC_IR_DAPHNEIR_DAPHNETYPESTORAGE_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir::daphne {

// Forward declarations of enums
enum class MatrixRepresentation;
enum class BoolOrUnknown;
namespace detail {

struct MatrixTypeStorage : public ::mlir::TypeStorage {
    // TODO: adapt epsilon for equality check (I think the only use is saving
    // memory for the MLIR-IR representation of this type)
    //  the choosen epsilon directly defines how accurate our sparsity inference
    //  can be
    constexpr static const double epsilon = 1e-6;
    MatrixTypeStorage(::mlir::Type elementType, ssize_t numRows, ssize_t numCols, double sparsity,
                      MatrixRepresentation representation, BoolOrUnknown symmetric, ssize_t mncSketchId);

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<::mlir::Type, ssize_t, ssize_t, double, MatrixRepresentation, BoolOrUnknown, ssize_t>;
    bool operator==(const KeyTy &tblgenKey) const;
    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey);

    /// Define a construction method for creating a new instance of this
    /// storage.
    static MatrixTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey);

    // Parameters
    ::mlir::Type elementType;
    ssize_t numRows;
    ssize_t numCols;
    double sparsity;
    MatrixRepresentation representation;
    BoolOrUnknown symmetric;
    ssize_t mncSketchId;
};

} // namespace detail
} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNETYPESTORAGE_H
