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
#include <runtime/local/datastructures/Structure.h>
#include <util/ErrorHandler.h>

#include <mlir/IR/Value.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferShapeOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************

std::pair<ssize_t, ssize_t> getShape(Value v) {
    Type t = v.getType();
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(t))
        return std::make_pair(mt.getNumRows(), mt.getNumCols());
    if (auto ft = llvm::dyn_cast<daphne::FrameType>(t))
        return std::make_pair(ft.getNumRows(), ft.getNumCols());
    if (auto ct = llvm::dyn_cast<daphne::ColumnType>(t))
        return std::make_pair(ct.getNumRows(), 1);
    if (CompilerUtils::isScaType(t))
        return std::make_pair(1, 1);

    std::stringstream s;
    s << "getShape(): the given value has neither a supported data type nor a supported value type: `" << t << '`';
    throw std::runtime_error(s.str());
}

ssize_t inferNumRowsFromArgs(Operation *op, ValueRange vs) {
    // If there are no arguments, their common number of rows is unknown.
    if (vs.empty())
        return -1;

    // If the #rows of all arguments is known and matches, then this is the
    // inferred #rows. If the known #rows of any two arguments mismatch, an
    // exception is thrown. Otherwise, if the #rows of any argument is unknown,
    // the inferred #rows is unknown.
    ssize_t numRows = getShape(vs[0]).first;
    bool someUnknown = false;
    if (numRows == -1)
        someUnknown = true;
    for (size_t i = 1; i < vs.size(); i++) {
        const ssize_t nextNumRows = getShape(vs[i]).first;
        if (nextNumRows == -1)
            someUnknown = true;
        else if (numRows == -1)
            numRows = nextNumRows;
        else if (nextNumRows != numRows)
            throw ErrorHandler::compilerError(op->getLoc(), "InferShapeOpInterface",
                                              "shape inference: inferNumRowsFromArgs() requires that "
                                              "arguments have the same number of rows, but there is "
                                              "one with " +
                                                  std::to_string(numRows) + " and one with " +
                                                  std::to_string(nextNumRows) + " rows");
    }
    return someUnknown ? -1 : numRows;
}

ssize_t inferNumColsFromArgs(Operation *op, ValueRange vs) {
    // If the #cols of all arguments is known and matches, then this is the
    // inferred #cols. If the known #cols of any two arguments mismatch, an
    // exception is thrown. Otherwise, if the #cols of any argument is unknown,
    // the inferred #cols is unknown.
    ssize_t numCols = getShape(vs[0]).second;
    bool someUnknown = false;
    if (numCols == -1)
        someUnknown = true;
    for (size_t i = 1; i < vs.size(); i++) {
        const ssize_t nextNumCols = getShape(vs[i]).second;
        if (nextNumCols == -1)
            someUnknown = true;
        else if (numCols == -1)
            numCols = nextNumCols;
        else if (nextNumCols != numCols)
            throw ErrorHandler::compilerError(op->getLoc(), "InferShapeOpInterface",
                                              "shape inference: inferNumColsFromArgs() requires that "
                                              "arguments have the same number of columns, but there is "
                                              "one with " +
                                                  std::to_string(numCols) + " and one with " +
                                                  std::to_string(nextNumCols) + " columns");
    }
    return someUnknown ? -1 : numCols;
}

ssize_t inferNumRowsFromSumOfArgs(ValueRange vs) {
    ssize_t sumNumRows = 0;
    for (Value v : vs) {
        const ssize_t numRows = getShape(v).first;
        if (numRows == -1)
            return -1;
        sumNumRows += numRows;
    }
    return sumNumRows;
}

ssize_t inferNumColsFromSumOfArgs(ValueRange vs) {
    ssize_t sumNumCols = 0;
    for (Value v : vs) {
        const ssize_t numCols = getShape(v).second;
        if (numCols == -1)
            return -1;
        sumNumCols += numCols;
    }
    return sumNumCols;
}

// ****************************************************************************
// Shape inference interface implementations
// ****************************************************************************

ssize_t daphne::CartesianOp::inferNumRows() {
    auto ftLhs = llvm::dyn_cast<daphne::FrameType>(getLhs().getType());
    auto ftRhs = llvm::dyn_cast<daphne::FrameType>(getRhs().getType());
    return ftLhs.getNumRows() * ftRhs.getNumRows();
}

ssize_t daphne::SeqOp::inferNumRows() {
    Type fromTy = getFrom().getType();
    if (fromTy.isF64()) {
        try {
            double vFrom = CompilerUtils::constantOrThrow<double>(getFrom());
            double vTo = CompilerUtils::constantOrThrow<double>(getTo());
            double vInc = CompilerUtils::constantOrThrow<double>(getInc());
            return floor(vTo / vInc - vFrom / vInc) + 1;
        } catch (const std::runtime_error &e) {
            return -1;
        }
    }
    if (fromTy.isF32()) {
        try {
            float vFrom = CompilerUtils::constantOrThrow<float>(getFrom());
            float vTo = CompilerUtils::constantOrThrow<float>(getTo());
            float vInc = CompilerUtils::constantOrThrow<float>(getInc());
            return floor(vTo / vInc - vFrom / vInc) + 1;
        } catch (const std::runtime_error &e) {
            return -1;
        }
    } else if (fromTy.isSignedInteger(64)) {
        try {
            int64_t vFrom = CompilerUtils::constantOrThrow<int64_t>(getFrom());
            int64_t vTo = CompilerUtils::constantOrThrow<int64_t>(getTo());
            int64_t vInc = CompilerUtils::constantOrThrow<int64_t>(getInc());
            return abs(vTo - vFrom) / abs(vInc) + 1;
        } catch (const std::runtime_error &e) {
            return -1;
        }
    }
    throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SeqOp::inferNumRows)",
                                      "at the moment, shape inference for SeqOp supports only F64 and "
                                      "SI64 value types");
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::CreateFrameOp::inferShape() {
    return {{inferNumRowsFromArgs(this->getOperation(), getCols()), inferNumColsFromSumOfArgs(getCols())}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::GroupJoinOp::inferShape() {
    // We don't know the exact numbers of rows here, but we know the numbers of
    // columns.
    return {{-1, 2}, {-1, 1}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::GroupOp::inferShape() {
    // We don't know the exact number of groups here.
    const size_t numRows = -1;

    std::vector<std::string> newLabels;

    for (Value t : getKeyCol()) { // Adopting keyCol Labels
        std::string keyLabel = CompilerUtils::constantOrThrow<std::string>(t);
        std::string delimiter = ".";
        const std::string frameName = keyLabel.substr(0, keyLabel.find(delimiter));
        const std::string colLabel = keyLabel.substr(keyLabel.find(delimiter) + delimiter.length(), keyLabel.length());

        if (keyLabel == "*") {
            daphne::FrameType arg = llvm::dyn_cast<daphne::FrameType>(getFrame().getType());
            for (std::string frameLabel : *arg.getLabels()) {
                newLabels.push_back(frameLabel);
            }
        } else if (colLabel.compare("*") == 0) {
            daphne::FrameType arg = llvm::dyn_cast<daphne::FrameType>(getFrame().getType());
            std::vector<std::string> labels = *arg.getLabels();
            for (std::string label : labels) {
                std::string labelFrameName = label.substr(0, label.find(delimiter));
                if (labelFrameName.compare(frameName) == 0) {
                    newLabels.push_back(label);
                }
            }
        } else {
            newLabels.push_back(keyLabel);
        }
    }

    const size_t numCols = newLabels.size() + getAggCol().size();
    return {{numRows, numCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::MatMulOp::inferShape() {
    auto shapeLhs = getShape(getLhs());
    auto shapeRhs = getShape(getRhs());

    ssize_t numRows = -1;
    std::pair<bool, bool> pr = CompilerUtils::isConstant<bool>(getTransa());
    if (pr.first)
        numRows = pr.second ? shapeLhs.second : shapeLhs.first;

    ssize_t numCols = -1;
    std::pair<bool, bool> pc = CompilerUtils::isConstant<bool>(getTransb());
    if (pc.first)
        numCols = pc.second ? shapeRhs.first : shapeRhs.second;

    return {{numRows, numCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::ReadOp::inferShape() {
    auto p = CompilerUtils::isConstant<std::string>(getFileName());
    if (p.first) {
        FileMetaData fmd = CompilerUtils::getFileMetaData(getFileName());
        return {{fmd.numRows, fmd.numCols}};
    } else {
        return {{-1, -1}};
    }
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::OrderOp::inferShape() {
    size_t numRows = -1;
    size_t numCols = -1;

    Type t = getArg().getType();
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(t)) {
        numRows = mt.getNumRows();
        numCols = mt.getNumCols();
    }
    if (auto ft = llvm::dyn_cast<daphne::FrameType>(t)) {
        numRows = ft.getNumRows();
        numCols = ft.getNumCols();
    }
    std::pair<bool, bool> p = CompilerUtils::isConstant<bool>(getReturnIdxs());
    if (p.first) {
        if (p.second)
            numCols = 1;
    } else
        numCols = -1;

    return {{numRows, numCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::CondOp::inferShape() {
    Type condTy = getCond().getType();
    if (llvm::isa<daphne::UnknownType>(condTy))
        // Actually, this should not happen, because if the type of the
        // condition is unknown, the type of the result should be unknown
        // too per type inference, such that shape inference should not
        // even get called. Nevertheless, returning unknown will probably
        // not hurt in case anyone ever calls this from somewhere else.
        return {{-1, -1}};
    if (auto condMatTy = llvm::dyn_cast<daphne::MatrixType>(condTy))
        return {{condMatTy.getNumRows(), condMatTy.getNumCols()}};
    else if (auto condFrmTy = llvm::dyn_cast<daphne::FrameType>(condTy))
        throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::CondOp::inferShape)",
                                          "CondOp does not support frames for the condition yet");
    else if (CompilerUtils::isScaType(condTy)) { // cond is a scalar
        Type thenTy = getThenVal().getType();
        Type elseTy = getElseVal().getType();

        ssize_t thenNumRows = -1;
        ssize_t thenNumCols = -1;
        ssize_t elseNumRows = -1;
        ssize_t elseNumCols = -1;
        auto thenMatTy = llvm::dyn_cast<daphne::MatrixType>(thenTy);
        auto thenFrmTy = llvm::dyn_cast<daphne::FrameType>(thenTy);
        auto elseMatTy = llvm::dyn_cast<daphne::MatrixType>(elseTy);
        auto elseFrmTy = llvm::dyn_cast<daphne::FrameType>(elseTy);
        if (thenMatTy) {
            thenNumRows = thenMatTy.getNumRows();
            thenNumCols = thenMatTy.getNumCols();
        } else if (thenFrmTy) {
            thenNumRows = thenFrmTy.getNumRows();
            thenNumCols = thenFrmTy.getNumCols();
        }
        if (elseMatTy) {
            elseNumRows = elseMatTy.getNumRows();
            elseNumCols = elseMatTy.getNumCols();
        } else if (elseFrmTy) {
            elseNumRows = elseFrmTy.getNumRows();
            elseNumCols = elseFrmTy.getNumCols();
        }

        if ((thenMatTy || thenFrmTy) && (elseMatTy || elseFrmTy))
            return {{(thenNumRows == elseNumRows) ? thenNumRows : -1, (thenNumCols == elseNumCols) ? thenNumCols : -1}};
        else
            // Then-value or else-value is a scalar.
            return {{-1, -1}};
    }

    std::stringstream s;
    s << "CondOp::inferShape(): the condition is neither a supported data type nor a supported value type";
    throw std::runtime_error(s.str());
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::Conv2DForwardOp::inferShape() {
    auto shapeX = getShape(getInput());
    auto shapeW = getShape(getFilter());
    auto Hin = CompilerUtils::constantOrDefault<size_t>(getInputHeight(), 1);
    auto Win = CompilerUtils::constantOrDefault<size_t>(getInputWidth(), 1);
    auto Hf = CompilerUtils::constantOrDefault<size_t>(getFilterHeight(), 1);
    auto Wf = CompilerUtils::constantOrDefault<size_t>(getFilterWidth(), 1);
    auto padh = CompilerUtils::constantOrDefault<size_t>(getPadHeight(), 1);
    auto padw = CompilerUtils::constantOrDefault<size_t>(getPadWidth(), 1);
    auto strideh = CompilerUtils::constantOrDefault<size_t>(getStrideHeight(), 1);
    auto stridew = CompilerUtils::constantOrDefault<size_t>(getStrideWidth(), 1);

    size_t Hout = std::floor((Hin + 2 * padh - Hf) / strideh + 1);
    size_t Wout = std::floor((Win + 2 * padw - Wf) / stridew + 1);
    auto F = shapeW.first;

    ssize_t numRows = shapeX.first;
    ssize_t numCols = F == -1 ? -1 : F * Hout * Wout;

    // op output is [mat, scalar, scalar] for the convolved data and its
    // dimensions
    return {{numRows, numCols}, std::make_pair(1, 1), std::make_pair(1, 1)};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::AvgPoolForwardOp::inferShape() {
    auto Hin = CompilerUtils::constantOrDefault<size_t>(getInputHeight(), 1);
    auto Win = CompilerUtils::constantOrDefault<size_t>(getInputWidth(), 1);
    auto C = CompilerUtils::constantOrDefault<size_t>(getInputNumChannels(), 1);
    auto Hf = CompilerUtils::constantOrDefault<size_t>(getPoolHeight(), 1);
    auto Wf = CompilerUtils::constantOrDefault<size_t>(getPoolWidth(), 1);
    auto padh = CompilerUtils::constantOrDefault<size_t>(getPadHeight(), 1);
    auto padw = CompilerUtils::constantOrDefault<size_t>(getPadWidth(), 1);
    auto strideh = CompilerUtils::constantOrDefault<size_t>(getStrideHeight(), 1);
    auto stridew = CompilerUtils::constantOrDefault<size_t>(getStrideWidth(), 1);
    auto shapeX = getShape(getInput());
    auto numRows = shapeX.first;
    size_t Hout = std::floor((Hin + 2 * padh - Hf) / strideh + 1);
    size_t Wout = std::floor((Win + 2 * padw - Wf) / stridew + 1);
    auto numCols = C * Hout * Wout;

    // op output is [mat, scalar, scalar] for the convolved data and its
    // dimensions
    return {{numRows, numCols}, std::make_pair(1, 1), std::make_pair(1, 1)};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::MaxPoolForwardOp::inferShape() {
    auto Hin = CompilerUtils::constantOrDefault<size_t>(getInputHeight(), 1);
    auto Win = CompilerUtils::constantOrDefault<size_t>(getInputWidth(), 1);
    auto C = CompilerUtils::constantOrDefault<size_t>(getInputNumChannels(), 1);
    auto Hf = CompilerUtils::constantOrDefault<size_t>(getPoolHeight(), 1);
    auto Wf = CompilerUtils::constantOrDefault<size_t>(getPoolWidth(), 1);
    auto padh = CompilerUtils::constantOrDefault<size_t>(getPadHeight(), 1);
    auto padw = CompilerUtils::constantOrDefault<size_t>(getPadWidth(), 1);
    auto strideh = CompilerUtils::constantOrDefault<size_t>(getStrideHeight(), 1);
    auto stridew = CompilerUtils::constantOrDefault<size_t>(getStrideWidth(), 1);
    auto shapeX = getShape(getInput());
    auto numRows = shapeX.first;
    size_t Hout = std::floor((Hin + 2 * padh - Hf) / strideh + 1);
    size_t Wout = std::floor((Win + 2 * padw - Wf) / stridew + 1);
    auto numCols = C * Hout * Wout;

    // op output is [mat, scalar, scalar] for the convolved data and its
    // dimensions
    return {{numRows, numCols}, std::make_pair(1, 1), std::make_pair(1, 1)};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::CTableOp::inferShape() {
    // If the result shape is given as arguments, then we know it.
    // Otherwise, we don't.
    // TODO In case resNumRows/resNumCols are known to be -1 (i.e., if
    // the output shape shall be determined depending on the values in
    // the lhs and rhs input matrices) and the lhs/rhs input matrices
    // are compile-time constants, then we could determine the number
    // of rows/columns here.
    return {{CompilerUtils::constantOrDefault<ssize_t>(getResNumRows(), -1),
             CompilerUtils::constantOrDefault<ssize_t>(getResNumCols(), -1)}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::MatrixConstantOp::inferShape() {
    const Structure *mat =
        reinterpret_cast<const Structure *>(CompilerUtils::constantOrThrow<uint64_t>(getMatrixAddr()));
    return {{mat->getNumRows(), mat->getNumCols()}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::SliceRowOp::inferShape() {
    Type srcTy = getSource().getType();
    ssize_t srcNumRows;
    ssize_t srcNumCols;
    if (llvm::isa<daphne::UnknownType>(srcTy)) {
        srcNumRows = -1;
        srcNumCols = -1;
    } else if (auto srcMatTy = llvm::dyn_cast<daphne::MatrixType>(srcTy)) {
        srcNumRows = srcMatTy.getNumRows();
        srcNumCols = srcMatTy.getNumCols();
    } else if (auto srcFrmTy = llvm::dyn_cast<daphne::FrameType>(srcTy)) {
        srcNumRows = srcFrmTy.getNumRows();
        srcNumCols = srcFrmTy.getNumCols();
    } else
        // If this is the case, shape inference shouldn't have been called.
        throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SliceRowOp::inferShape)",
                                          "SliceRowOp shape inference does only support unknown, matrix, and "
                                          "frame inputs");

    auto loIn = CompilerUtils::isConstant<int64_t>(getLowerIncl());
    auto upEx = CompilerUtils::isConstant<int64_t>(getUpperExcl());

    ssize_t resNumRows = -1;
    if (srcNumRows != -1 && loIn.first && upEx.first) {
        ssize_t loInPos = loIn.second;
        ssize_t upExPos = upEx.second;
        if (loInPos < 0) {
            loInPos += srcNumRows;
            if (upExPos <= 0)
                upExPos += srcNumRows;
        } else if (upExPos < 0)
            upExPos += srcNumRows;

        if (loInPos < 0 || loInPos >= srcNumRows)
            throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SliceRowOp::inferShape)",
                                              "SliceRowOp shape inference: lowerIncl must be in [0, "
                                              "numRows), "
                                              "but is " +
                                                  std::to_string(loIn.second) + " with " + std::to_string(srcNumRows) +
                                                  " rows");
        if (upExPos < 0 || upExPos > srcNumRows)
            throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SliceRowOp::inferShape)",
                                              "SliceRowOp shape inference: upperExcl must be in [0, "
                                              "numRows], "
                                              "but is " +
                                                  std::to_string(upEx.second) + " with " + std::to_string(srcNumRows) +
                                                  " rows");
        if (loInPos > upExPos)
            throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SliceRowOp::inferShape)",
                                              "SliceRowOp shape inference: lowerIncl must not be greater "
                                              "than upperExcl"
                                              " (found " +
                                                  std::to_string(loInPos) + " and " + std::to_string(upExPos) + ")");
        resNumRows = upExPos - loInPos;
    }

    return {{resNumRows, srcNumCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::SliceColOp::inferShape() {
    Type srcTy = getSource().getType();
    ssize_t srcNumRows;
    ssize_t srcNumCols;

    if (llvm::isa<daphne::UnknownType>(srcTy)) {
        srcNumRows = -1;
        srcNumCols = -1;
    } else if (auto srcMatTy = llvm::dyn_cast<daphne::MatrixType>(srcTy)) {
        srcNumRows = srcMatTy.getNumRows();
        srcNumCols = srcMatTy.getNumCols();
    } else if (auto srcFrmTy = llvm::dyn_cast<daphne::FrameType>(srcTy)) {
        srcNumRows = srcFrmTy.getNumRows();
        srcNumCols = srcFrmTy.getNumCols();
    } else
        // If this is the case, shape inference shouldn't have been called.
        throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SliceColOp::inferShape)",
                                          "SliceColOp shape inference does only support unknown, matrix, and "
                                          "frame inputs");

    auto loIn = CompilerUtils::isConstant<int64_t>(getLowerIncl());
    auto upEx = CompilerUtils::isConstant<int64_t>(getUpperExcl());

    ssize_t resNumCols = -1;
    if (srcNumCols != -1 && loIn.first && upEx.first) {
        ssize_t loInPos = loIn.second;
        ssize_t upExPos = upEx.second;
        if (loInPos < 0) {
            loInPos += srcNumCols;
            if (upExPos <= 0)
                upExPos += srcNumCols;
        } else if (upExPos < 0)
            upExPos += srcNumCols;

        if (loInPos < 0 || loInPos >= srcNumCols)
            throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SliceColOp::inferShape)",
                                              "SliceColOp shape inference: lowerIncl must be in [0, "
                                              "numCols), "
                                              "but is " +
                                                  std::to_string(loIn.second) + " with " + std::to_string(srcNumCols) +
                                                  " cols");
        if (upExPos < 0 || upExPos > srcNumCols)
            throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SliceColOp::inferShape)",
                                              "SliceColOp shape inference: upperExcl must be in [0, "
                                              "numCols], "
                                              "but is " +
                                                  std::to_string(upEx.second) + " with " + std::to_string(srcNumCols) +
                                                  " cols");
        if (loInPos > upExPos)
            throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::SliceColOp::inferShape)",
                                              "SliceColOp shape inference: lowerIncl must not be greater "
                                              "than upperExcl"
                                              " (found " +
                                                  std::to_string(loInPos) + " and " + std::to_string(upExPos) + ")");
        resNumCols = upExPos - loInPos;
    }

    return {{srcNumRows, resNumCols}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::ExtractColOp::inferShape() {
    auto ft = llvm::dyn_cast<daphne::FrameType>(getSource().getType());
    auto srcNumRows = getShape(getOperand(0)).first;
    auto st = llvm::dyn_cast<daphne::StringType>(getSelectedCols().getType());

    if (ft && st) {
        std::string label = CompilerUtils::constantOrThrow<std::string>(getSelectedCols());
        std::string delimiter = ".";
        const std::string frameName = label.substr(0, label.find(delimiter));
        const std::string colLabel = label.substr(label.find(delimiter) + delimiter.length(), label.length());
        if (colLabel.compare("*") == 0) {
            std::vector<std::string> labels = *ft.getLabels();
            ssize_t numCols = 0;
            for (size_t i = 0; i < labels.size(); i++) {
                std::string labelFrameName = labels[i].substr(0, labels[i].find(delimiter));
                if (labelFrameName.compare(frameName) == 0) {
                    numCols++;
                }
            }
            return {{srcNumRows, numCols}};
        }
    }
    // Default case except when the selectedCols ends in a wildcard
    return {{srcNumRows, getShape(getOperand(1)).first}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::EigenOp::inferShape() {
    auto shape = getShape(getOperand());
    return {{shape.first, 1}, {shape.first, shape.first}};
}

std::vector<std::pair<ssize_t, ssize_t>> daphne::RecodeOp::inferShape() {
    // Intuition:
    // - The (data) result has the same shape as the argument.
    // - The (dict) result has an unknown number of rows and one column.

    Type argTy = getArg().getType();

    ssize_t resNumRows;
    ssize_t resNumCols;
    if (auto argMatTy = llvm::dyn_cast<daphne::MatrixType>(argTy)) {
        resNumRows = argMatTy.getNumRows();
        resNumCols = argMatTy.getNumCols();
    } else if (auto argFrmTy = llvm::dyn_cast<daphne::FrameType>(argTy)) {
        resNumRows = argFrmTy.getNumRows();
        resNumCols = argFrmTy.getNumCols();
    } else if (llvm::isa<daphne::UnknownType>(argTy)) {
        resNumRows = -1;
        resNumCols = -1;
    } else
        throw ErrorHandler::compilerError(getLoc(), "InferShapeOpInterface (daphne::RecodeOp::inferShape)",
                                          "the argument to recode has an invalid type");

    // TODO We could infer (or estimate) the number of rows of the dictionary
    // result if we knew the number of distinct values in the argument (or could
    // estimate it).
    const ssize_t dictNumRows = -1;
    const ssize_t dictNumCols = 1;

    return {{resNumRows, resNumCols}, {dictNumRows, dictNumCols}};
}

// ****************************************************************************
// Shape inference trait implementations
// ****************************************************************************

template <size_t i> struct tryNumRowsFromIthScalar {
    using T = std::pair<ssize_t, ssize_t>;
    static void apply(T &shape, Operation *op) {
        if (op->hasTrait<NumRowsFromIthScalar<i>::template Impl>()) {
            ssize_t numRows = CompilerUtils::constantOrDefault<int64_t>(op->getOperand(i), -1);
            ssize_t numCols = shape.second;
            shape = {numRows, numCols};
        }
    }
};
template <size_t i> struct tryNumColsFromIthScalar {
    using T = std::pair<ssize_t, ssize_t>;
    static void apply(T &shape, Operation *op) {
        if (op->hasTrait<NumColsFromIthScalar<i>::template Impl>()) {
            ssize_t numRows = shape.first;
            ssize_t numCols = CompilerUtils::constantOrDefault<int64_t>(op->getOperand(i), -1);
            shape = {numRows, numCols};
        }
    }
};

template <size_t i> struct tryNumRowsFromIthArg {
    using T = std::pair<ssize_t, ssize_t>;
    static void apply(T &shape, Operation *op) {
        if (op->hasTrait<NumRowsFromIthArg<i>::template Impl>()) {
            ssize_t numRows = getShape(op->getOperand(i)).first;
            ssize_t numCols = shape.second;
            shape = {numRows, numCols};
        }
    }
};
template <size_t i> struct tryNumColsFromIthArg {
    using T = std::pair<ssize_t, ssize_t>;
    static void apply(T &shape, Operation *op) {
        if (op->hasTrait<NumColsFromIthArg<i>::template Impl>()) {
            ssize_t numRows = shape.first;
            ssize_t numCols = getShape(op->getOperand(i)).second;
            shape = {numRows, numCols};
        }
    }
};

template <size_t i> struct tryNumRowsFromIthArgNumCols {
    using T = std::pair<ssize_t, ssize_t>;
    static void apply(T &shape, Operation *op) {
        if (op->hasTrait<NumRowsFromIthArgNumCols<i>::template Impl>()) {
            ssize_t numRows = getShape(op->getOperand(i)).second;
            ssize_t numCols = shape.second;
            shape = {numRows, numCols};
        }
    }
};
template <size_t i> struct tryNumColsFromIthArgNumRows {
    using T = std::pair<ssize_t, ssize_t>;
    static void apply(T &shape, Operation *op) {
        if (op->hasTrait<NumColsFromIthArgNumRows<i>::template Impl>()) {
            ssize_t numRows = shape.first;
            ssize_t numCols = getShape(op->getOperand(i)).first;
            shape = {numRows, numCols};
        }
    }
};

template <size_t i> struct tryShapeFromIthArg {
    using T = std::pair<ssize_t, ssize_t>;
    static void apply(T &shape, Operation *op) {
        if (op->hasTrait<ShapeFromIthArg<i>::template Impl>())
            shape = getShape(op->getOperand(i));
    }
};

// ****************************************************************************
// Shape inference function
// ****************************************************************************

std::vector<std::pair<ssize_t, ssize_t>> daphne::tryInferShape(Operation *op) {
    if (auto inferShapeOp = llvm::dyn_cast<daphne::InferShape>(op))
        // If the operation implements the shape inference interface,
        // we apply that.
        return inferShapeOp.inferShape();
    else if (op->getNumResults() == 1) {
        // If the operation does not implement the shape inference interface
        // and has exactly one result, we utilize its shape inference traits,
        // or the inference interfaces for the number of rows and columns
        // (separately).

        std::pair<ssize_t, ssize_t> shape = {-1, -1};

        if (op->hasTrait<OneRow>())
            shape = {1, shape.second};
        if (op->hasTrait<OneCol>())
            shape = {shape.first, 1};
        // Our parametric traits addressing a certain argument are supported
        // for up to 10 arguments (this can easily be changed here).
        // There does not seem to be a way in MLIR do it more generically,
        // since the parameters of parametric traits are template parameters.
        const size_t u = 9;
        tryParamTraitUntil<u, tryNumRowsFromIthScalar>::apply(shape, op);
        tryParamTraitUntil<u, tryNumColsFromIthScalar>::apply(shape, op);
        tryParamTraitUntil<u, tryNumRowsFromIthArg>::apply(shape, op);
        tryParamTraitUntil<u, tryNumColsFromIthArg>::apply(shape, op);
        tryParamTraitUntil<u, tryNumRowsFromIthArgNumCols>::apply(shape, op);
        tryParamTraitUntil<u, tryNumColsFromIthArgNumRows>::apply(shape, op);
        if (op->hasTrait<NumRowsFromAllArgs>())
            shape = {inferNumRowsFromArgs(op, op->getOperands()), shape.second};
        if (op->hasTrait<NumColsFromAllArgs>())
            shape = {shape.first, inferNumColsFromArgs(op, op->getOperands())};
        if (op->hasTrait<NumRowsFromSumOfAllArgs>())
            shape = {inferNumRowsFromSumOfArgs(op->getOperands()), shape.second};
        if (op->hasTrait<NumColsFromSumOfAllArgs>())
            shape = {shape.first, inferNumColsFromSumOfArgs(op->getOperands())};
        tryParamTraitUntil<u, tryShapeFromIthArg>::apply(shape, op);
        if (op->hasTrait<ShapeEwBinary>()) {
            // The output has the shape of the left-hand-side operand. This is
            // consistent with the kernel, but in the future, we should extend
            // this to support broadcasting of vectors and scalars from left
            // and right.
            auto shapeLhs = getShape(op->getOperand(0));
            auto shapeRhs = getShape(op->getOperand(1));
            // This first case is just a workaround, we should decide later how
            // to treat incomplete knowledge of the shapes.
            if (shapeLhs.first == -1 && shapeLhs.second == 1 && shapeRhs.first == -1 && shapeRhs.second == 1)
                shape = {-1, 1};
            else if (shapeRhs.first == -1 || shapeRhs.second == -1)
                shape = {-1, -1};
            else
                shape = {shapeLhs.first, shapeLhs.second};
            // TODO Throw if lhs and rhs don't agree.
        }

        if (auto inferNumRowsOp = llvm::dyn_cast<daphne::InferNumRows>(op))
            shape = {inferNumRowsOp.inferNumRows(), shape.second};
        if (auto inferNumColsOp = llvm::dyn_cast<daphne::InferNumCols>(op))
            shape = {shape.first, inferNumColsOp.inferNumCols()};

        // Note that all our shape inference traits assume that the operation
        // has exactly one result (which is the case for most DaphneIR ops).
        return {shape};
    } else {
        // If the operation does not implement the shape inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<std::pair<ssize_t, ssize_t>> shapes;
        for (size_t i = 0; i < op->getNumResults(); i++)
            shapes.push_back({-1, -1});
        return shapes;
    }
}
