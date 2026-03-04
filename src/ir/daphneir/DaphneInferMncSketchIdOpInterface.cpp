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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/MncSketch.h>
#include <runtime/local/datastructures/MncSketchPropagate.h>
#include <runtime/local/datastructures/MncSketchBuild.h>

namespace mlir::daphne {
    #include <ir/daphneir/DaphneInferMncSketchIdOpInterface.cpp.inc>
    static thread_local MncSketchRegistry *gMncReg = nullptr;

    void setActiveMncSketchRegistry(MncSketchRegistry *reg) { gMncReg = reg; }
    void clearActiveMncSketchRegistry() { gMncReg = nullptr; }
    MncSketchRegistry *getActiveMncSketchRegistry() { return gMncReg; }
}

#include <vector>
#include <memory>
#include <random>
#include <cstdint>
#include <algorithm>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************
// use this to avoid value mismatches
int64_t UNKNOWN_SKETCH_ID = -1; 

template <typename VT>
static int64_t buildAndStoreSketch(uint64_t matrixAddr) {
    auto *reg = daphne::getActiveMncSketchRegistry();
    if (!reg)
        return {-1};

    const DenseMatrix<VT> *mat = reinterpret_cast<const DenseMatrix<VT> *>(matrixAddr);
    MncSketch h = buildMncFromDenseMatrix(*mat);
    return static_cast<int64_t>(reg->store(std::move(h)));
}


// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

std::vector<int64_t> daphne::RandMatrixOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (!reg)
        return {UNKNOWN_SKETCH_ID};

    auto numRows  = CompilerUtils::isConstant<ssize_t>(getNumRows());
    auto numCols  = CompilerUtils::isConstant<ssize_t>(getNumCols());
    auto sparsity = CompilerUtils::isConstant<double>(getSparsity());
    auto seed     = CompilerUtils::isConstant<int64_t>(getSeed());

    if (!numRows.first || !numCols.first || !sparsity.first || !seed.first)
        return {UNKNOWN_SKETCH_ID};

    MncSketch hC = buildMncFromRand(
        static_cast<std::size_t>(numRows.second),
        static_cast<std::size_t>(numCols.second),
        sparsity.second,
        seed.second
    );

    return {reg->store(std::move(hC))};
}
static bool getConstAsDouble(Value v, double &out) {
    if (auto p = CompilerUtils::isConstant<double>(v); p.first) { out = p.second; return true; }
    if (auto p = CompilerUtils::isConstant<float>(v); p.first)  { out = static_cast<double>(p.second); return true; }
    if (auto p = CompilerUtils::isConstant<int64_t>(v); p.first){ out = static_cast<double>(p.second); return true; }
    if (auto p = CompilerUtils::isConstant<int32_t>(v); p.first){ out = static_cast<double>(p.second); return true; }
    if (auto p = CompilerUtils::isConstant<int8_t>(v); p.first) { out = static_cast<double>(p.second); return true; }
    if (auto p = CompilerUtils::isConstant<uint64_t>(v); p.first){ out = static_cast<double>(p.second); return true; }
    if (auto p = CompilerUtils::isConstant<uint32_t>(v); p.first){ out = static_cast<double>(p.second); return true; }
    if (auto p = CompilerUtils::isConstant<uint8_t>(v); p.first) { out = static_cast<double>(p.second); return true; }
    return false;
}

std::vector<int64_t> daphne::FillOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (reg == nullptr) {
        return {UNKNOWN_SKETCH_ID};
    }
    auto numRows = CompilerUtils::isConstant<ssize_t>(getNumRows());
    auto numCols = CompilerUtils::isConstant<ssize_t>(getNumCols());
    if (!numRows.first || !numCols.first)
        return {UNKNOWN_SKETCH_ID};

    double val;
    if (!getConstAsDouble(getArg(), val))
        return {UNKNOWN_SKETCH_ID};

    MncSketch hC = buildMncFromFill(
        val,
        static_cast<std::size_t>(numRows.second),
        static_cast<std::size_t>(numCols.second)
    );

    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::SeqOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (!reg)
        return {UNKNOWN_SKETCH_ID};

    auto from = CompilerUtils::isConstant<ssize_t>(getFrom());
    auto to   = CompilerUtils::isConstant<ssize_t>(getTo());

    double inc;
    if (!from.first || !to.first || !getConstAsDouble(getInc(), inc))
        return {UNKNOWN_SKETCH_ID};

    MncSketch hC = buildMncFromSeq(
        static_cast<double>(from.second),
        static_cast<double>(to.second),
        static_cast<double>(inc)
    );

    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::MatMulOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (!reg)
        return {UNKNOWN_SKETCH_ID};

    auto lhsTy = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsTy = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());

    auto lhsId = lhsTy.getMncSketchId();
    auto rhsId = rhsTy.getMncSketchId();
    if (lhsId == -1 || rhsId == -1) {
        return {UNKNOWN_SKETCH_ID};
    }
    const MncSketch *hA = reg->get(lhsId);
    const MncSketch *hB = reg->get(rhsId);
    if (hA == nullptr || hB == nullptr) {
        return {UNKNOWN_SKETCH_ID};
    }
    MncSketch hC = propagateMM(*hA, *hB);
    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::TransposeOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (reg == nullptr) {
        return {UNKNOWN_SKETCH_ID};
    }
    auto mt = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    auto id = mt.getMncSketchId();
    if (id != -1) {
        const MncSketch *hA = reg->get(id);
        if (hA == nullptr) {
            return {UNKNOWN_SKETCH_ID};
        }
        MncSketch hC = propagateTranspose(*hA);
        return {reg->store(std::move(hC))};
    }
    return {UNKNOWN_SKETCH_ID};
}

std::vector<int64_t> daphne::RowBindOp::inferMncSketchId(){
    auto *reg = getActiveMncSketchRegistry();
    if (!reg)
        return {UNKNOWN_SKETCH_ID};

    auto lhsTy = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsTy = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());

    auto lhsId = lhsTy.getMncSketchId();
    auto rhsId = rhsTy.getMncSketchId();
    if (lhsId == -1 || rhsId == -1) {
        return {UNKNOWN_SKETCH_ID};
    }
    const MncSketch *hA = reg->get(lhsId);
    const MncSketch *hB = reg->get(rhsId);
    if (hA == nullptr || hB == nullptr) {
        return {UNKNOWN_SKETCH_ID};
    }
    MncSketch hC = propagateRbind(*hA, *hB);
    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::ColBindOp::inferMncSketchId(){
    auto *reg = getActiveMncSketchRegistry();
    if (!reg)
        return {UNKNOWN_SKETCH_ID};

    auto lhsTy = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsTy = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());

    auto lhsId = lhsTy.getMncSketchId();
    auto rhsId = rhsTy.getMncSketchId();
    if (lhsId == -1 || rhsId == -1) {
        return {UNKNOWN_SKETCH_ID};
    }
    const MncSketch *hA = reg->get(lhsId);
    const MncSketch *hB = reg->get(rhsId);
    if (hA == nullptr || hB == nullptr) {
        return {UNKNOWN_SKETCH_ID};
    }
    MncSketch hC = propagateCbind(*hA, *hB);
    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::DiagMatrixOp::inferMncSketchId(){
    auto *reg = getActiveMncSketchRegistry();
    if (!reg)
        return {UNKNOWN_SKETCH_ID};

    auto argTy = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    auto argId = argTy.getMncSketchId();
    if (argId == -1) {
        return {UNKNOWN_SKETCH_ID};
    }
    const MncSketch *hA = reg->get(argId);
    if (hA == nullptr) {
        return {UNKNOWN_SKETCH_ID};
    }
    MncSketch hC = propagateMncFromDiagMatrix(*hA);
    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::ReshapeOp::inferMncSketchId() {
    auto *reg = daphne::getActiveMncSketchRegistry();
    if (!reg)
        return {daphne::UNKNOWN_SKETCH_ID};

    auto mt = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    if (!mt)
        return {daphne::UNKNOWN_SKETCH_ID};

    auto id = mt.getMncSketchId();
    if (id == daphne::UNKNOWN_SKETCH_ID)
        return {daphne::UNKNOWN_SKETCH_ID};

    auto numRows = CompilerUtils::isConstant<ssize_t>(getNumRows());
    auto numCols = CompilerUtils::isConstant<ssize_t>(getNumCols());
    if (!numRows.first || !numCols.first)
        return {daphne::UNKNOWN_SKETCH_ID};

    const MncSketch *hA = reg->get(id);
    if (!hA)
        return {daphne::UNKNOWN_SKETCH_ID};

    MncSketch hC = propagateMncFromReshape(
        *hA,
        static_cast<std::size_t>(numRows.second),
        static_cast<std::size_t>(numCols.second)
    );

    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::MatrixConstantOp::inferMncSketchId() {
    auto p = CompilerUtils::isConstant<uint64_t>(getMatrixAddr());
    if (!p.first)
        return {daphne::UNKNOWN_SKETCH_ID};

    const uint64_t matrixAddr = p.second;
    Type vt = CompilerUtils::getValueType(getResult().getType());

    if (vt.isF64())
        return {buildAndStoreSketch<double>(matrixAddr)};
    if (vt.isF32())
        return {buildAndStoreSketch<float>(matrixAddr)};
    if (vt.isSignedInteger(64))
        return {buildAndStoreSketch<int64_t>(matrixAddr)};
    if (vt.isSignedInteger(32))
        return {buildAndStoreSketch<int32_t>(matrixAddr)};
    if (vt.isSignedInteger(8))
        return {buildAndStoreSketch<int8_t>(matrixAddr)};
    if (vt.isUnsignedInteger(64))
        return {buildAndStoreSketch<uint64_t>(matrixAddr)};
    if (vt.isUnsignedInteger(32))
        return {buildAndStoreSketch<uint32_t>(matrixAddr)};
    if (vt.isUnsignedInteger(8))
        return {buildAndStoreSketch<uint8_t>(matrixAddr)};

    return {daphne::UNKNOWN_SKETCH_ID};
}

// elementwise
std::vector<int64_t> daphne::EwAddOp::inferMncSketchId() {
    auto *reg = daphne::getActiveMncSketchRegistry();
    if (!reg)
        return {daphne::UNKNOWN_SKETCH_ID};

    Value lhs = getLhs();
    Value rhs = getRhs();

    auto lhsMt = llvm::dyn_cast<daphne::MatrixType>(lhs.getType());
    auto rhsMt = llvm::dyn_cast<daphne::MatrixType>(rhs.getType());

    // Case 1: matrix + matrix
    if (lhsMt && rhsMt) {
        auto lhsId = lhsMt.getMncSketchId();
        auto rhsId = rhsMt.getMncSketchId();
        if (lhsId == daphne::UNKNOWN_SKETCH_ID || rhsId == daphne::UNKNOWN_SKETCH_ID)
            return {daphne::UNKNOWN_SKETCH_ID};

        const MncSketch *hA = reg->get(lhsId);
        const MncSketch *hB = reg->get(rhsId);
        if (!hA || !hB)
            return {daphne::UNKNOWN_SKETCH_ID};

        MncSketch hC = propagateAdd(*hA, *hB);
        return {reg->store(std::move(hC))};
    }

    // Case 2: matrix + scalar (either side)
    if (lhsMt || rhsMt) {
        Value matVal = lhsMt ? lhs : rhs;
        Value scaVal = lhsMt ? rhs : lhs;

        auto matTy = llvm::dyn_cast<daphne::MatrixType>(matVal.getType());
        auto matId = matTy.getMncSketchId();
        if (matId == daphne::UNKNOWN_SKETCH_ID)
            return {daphne::UNKNOWN_SKETCH_ID};

        const MncSketch *hA = reg->get(matId);
        if (!hA)
            return {daphne::UNKNOWN_SKETCH_ID};

        double s;
        if (!getConstAsDouble(scaVal, s))
            return {daphne::UNKNOWN_SKETCH_ID};

        if (s == 0.0) {
            return {matId}; // adding 0 doesn't change pattern
        } else {
            // Conservative: densify
            MncSketch hC = buildMncFromRand(hA->m, hA->n, 1.0, 0);
            return {reg->store(std::move(hC))};
        }
    }

    return {daphne::UNKNOWN_SKETCH_ID};
}

std::vector<int64_t> daphne::EwSubOp::inferMncSketchId() {
    auto *reg = daphne::getActiveMncSketchRegistry();
    if (!reg)
        return {-1};

    Value lhs = getLhs();
    Value rhs = getRhs();

    auto lhsMt = llvm::dyn_cast<daphne::MatrixType>(lhs.getType());
    auto rhsMt = llvm::dyn_cast<daphne::MatrixType>(rhs.getType());

    // -------------------------
    // Case 1: matrix - matrix
    // -------------------------
    if (lhsMt && rhsMt) {
        auto lhsId = lhsMt.getMncSketchId();
        auto rhsId = rhsMt.getMncSketchId();
        if (lhsId == -1 || rhsId == -1)
            return {-1};

        const MncSketch *hA = reg->get(lhsId);
        const MncSketch *hB = reg->get(rhsId);
        if (!hA || !hB)
            return {-1};

        MncSketch hC = propagateAdd(*hA, *hB);
        return {reg->store(std::move(hC))};
    }

    // -------------------------
    // Case 2: matrix - scalar OR scalar - matrix
    // -------------------------
    if (lhsMt || rhsMt) {
        bool lhsIsMatrix = static_cast<bool>(lhsMt);
        Value matVal = lhsIsMatrix ? lhs : rhs;
        Value scaVal = lhsIsMatrix ? rhs : lhs;

        auto matTy = llvm::dyn_cast<daphne::MatrixType>(matVal.getType());
        auto matId = matTy.getMncSketchId();
        if (matId == -1)
            return {-1};

        const MncSketch *hA = reg->get(matId);
        if (!hA)
            return {-1};

        // Need constant scalar to infer anything
        double s;
        if (!getConstAsDouble(scaVal, s))
            return {daphne::UNKNOWN_SKETCH_ID};

        if (s == 0.0) {
            // matrix - 0  OR  0 - matrix : nonzero pattern stays the same as matrix (ignoring sign)
            return {matId};
        } else {
            // conservative: adding/subtracting nonzero scalar can turn zeros into nonzeros -> dense
            MncSketch hC = buildMncFromRand(hA->m, hA->n, 1.0, 0);
            return {reg->store(std::move(hC))};
        }
    }

    // -------------------------
    // Case 3: scalar - scalar
    // -------------------------
    return {-1};
}
std::vector<int64_t> daphne::EwMulOp::inferMncSketchId() {
    auto *reg = daphne::getActiveMncSketchRegistry();
    if (!reg)
        return {-1};

    Value lhs = getLhs();
    Value rhs = getRhs();

    auto lhsMt = llvm::dyn_cast<daphne::MatrixType>(lhs.getType());
    auto rhsMt = llvm::dyn_cast<daphne::MatrixType>(rhs.getType());

    // -------------------------
    // Case 1: matrix * matrix
    // -------------------------
    if (lhsMt && rhsMt) {
        auto lhsId = lhsMt.getMncSketchId();
        auto rhsId = rhsMt.getMncSketchId();
        if (lhsId == -1 || rhsId == -1)
            return {-1};

        const MncSketch *hA = reg->get(lhsId);
        const MncSketch *hB = reg->get(rhsId);
        if (!hA || !hB)
            return {-1};

        MncSketch hC = propagateMul(*hA, *hB);
        return {reg->store(std::move(hC))};
    }

    // -------------------------
    // Case 2: matrix * scalar OR scalar * matrix
    // -------------------------
    if (lhsMt || rhsMt) {
        Value matVal = lhsMt ? lhs : rhs;
        Value scaVal = lhsMt ? rhs : lhs;

        auto matTy = llvm::dyn_cast<daphne::MatrixType>(matVal.getType());
        auto matId = matTy.getMncSketchId();
        if (matId == -1)
            return {-1};

        const MncSketch *hA = reg->get(matId);
        if (!hA)
            return {-1};

        // Need constant scalar to refine. If unknown, safest is unknown.
        double c;
        if (!getConstAsDouble(scaVal, c))
            return {daphne::UNKNOWN_SKETCH_ID};

        if (c == 0.0) {
            // Result is all zeros
            MncSketch hZ = buildMncFromFill(0.0, hA->m, hA->n);   // adjust field names if needed
            return {reg->store(std::move(hZ))};
        } else {
            // Multiplying by nonzero keeps the nonzero pattern -> reuse same sketch id
            return {matId};
        }
    }

    // -------------------------
    // Case 3: scalar * scalar
    // -------------------------
    return {-1};
}

std::vector<int64_t> daphne::EwDivOp::inferMncSketchId() {
    auto *reg = daphne::getActiveMncSketchRegistry();
    if (!reg)
        return {-1};

    Value lhs = getLhs();
    Value rhs = getRhs();

    auto lhsMt = llvm::dyn_cast<daphne::MatrixType>(lhs.getType());
    auto rhsMt = llvm::dyn_cast<daphne::MatrixType>(rhs.getType());

    // -------------------------
    // Case 1: matrix * matrix
    // -------------------------
    if (lhsMt && rhsMt) {
        auto lhsId = lhsMt.getMncSketchId();
        auto rhsId = rhsMt.getMncSketchId();
        if (lhsId == -1 || rhsId == -1)
            return {-1};

        const MncSketch *hA = reg->get(lhsId);
        const MncSketch *hB = reg->get(rhsId);
        if (!hA || !hB)
            return {-1};

        MncSketch hC = propagateMul(*hA, *hB);
        return {reg->store(std::move(hC))};
    }

    // -------------------------
    // Case 2: matrix * scalar OR scalar * matrix
    // -------------------------
    if (lhsMt || rhsMt) {
        Value matVal = lhsMt ? lhs : rhs;
        Value scaVal = lhsMt ? rhs : lhs;

        auto matTy = llvm::dyn_cast<daphne::MatrixType>(matVal.getType());
        auto matId = matTy.getMncSketchId();
        if (matId == -1)
            return {-1};

        const MncSketch *hA = reg->get(matId);
        if (!hA)
            return {-1};

        // Need constant scalar to refine. If unknown, safest is unknown.
        double c;
        if (!getConstAsDouble(scaVal, c))
            return {daphne::UNKNOWN_SKETCH_ID};

        if (c == 0.0) {
            // Result is all zeros
            MncSketch hZ = buildMncFromFill(0.0, hA->m, hA->n);   // adjust field names if needed
            return {reg->store(std::move(hZ))};
        } else {
            // Multiplying by nonzero keeps the nonzero pattern -> reuse same sketch id
            return {matId};
        }
    }

    // -------------------------
    // Case 3: scalar * scalar
    // -------------------------
    return {-1};
}
// readop : some property of the input data, mncsketch (the whole) can be stored as metadata files 
// and can be read by the compiler, just for 

// take data from files metadata and infer mnc sketch if available, otherwise return unknown
std::vector<int64_t> daphne::ReadOp::inferMncSketchId() {
    std::pair<bool, std::string> p = CompilerUtils::isConstant<std::string>(getFileName());
    if (p.first) {
        FileMetaData fmd = MetaDataParser::readMetaData(p.second);
        if (fmd.mncSketch.has_value()) {
            auto *reg = getActiveMncSketchRegistry();
            if (reg == nullptr) {
                return {-1};
            }
            return {reg->store(fmd.mncSketch.value())};
        }
    }
    return {-1};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<int64_t> daphne::tryInferMncSketchId(Operation *op) {
    if (auto inferMncSketchIdOp = llvm::dyn_cast<daphne::InferMncSketchId>(op))
        // If the operation implements the inference interface, we apply that.
        return inferMncSketchIdOp.inferMncSketchId();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<int64_t> mncSketchIds;
        for (size_t i = 0; i < op->getNumResults(); i++)
            mncSketchIds.push_back(-1);
        return mncSketchIds;
    }
}