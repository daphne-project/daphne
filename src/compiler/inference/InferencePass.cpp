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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>
#include <util/ErrorHandler.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>

#include <runtime/local/datastructures/MncSketch.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace mlir;
daphne::MncSketchRegistry mncRegistry;

daphne::InferenceConfig::InferenceConfig(bool partialInferenceAllowed, bool typeInference, bool shapeInference,
                                         bool frameLabelInference, bool sparsityInference, bool symmetricInference, bool mncSketchIdInference)
    : partialInferenceAllowed(partialInferenceAllowed), typeInference(typeInference), shapeInference(shapeInference),
      frameLabelInference(frameLabelInference), sparsityInference(sparsityInference),
      symmetricInference(symmetricInference), mncSketchIdInference(mncSketchIdInference) {}

namespace {
void castOperandIf(OpBuilder &builder, Operation *op, size_t operandIdx, Type type) {
    Value operand = op->getOperand(operandIdx);
    if (operand.getType() != type) {
        builder.setInsertionPoint(op);
        op->setOperand(operandIdx,
                       // TODO Is this the right loc?
                       builder.create<daphne::CastOp>(op->getLoc(), type, operand));
    }
}

/**
 * @brief Returns a type retaining all common properties of the two
 * given types, and setting all mismatching properties to unknown.
 *
 * If the two given types are of different data types, then `nullptr`
 * is returned.
 */
Type getTypeWithCommonInfo(Type t1, Type t2) {
    MLIRContext *ctx = t1.getContext();
    Type u = daphne::UnknownType::get(ctx);
    auto mat1 = mlir::dyn_cast<daphne::MatrixType>(t1);
    auto mat2 = mlir::dyn_cast<daphne::MatrixType>(t2);
    auto frm1 = mlir::dyn_cast<daphne::FrameType>(t1);
    auto frm2 = mlir::dyn_cast<daphne::FrameType>(t2);

    if (mat1 && mat2) { // both types are matrices
        const Type vt1 = mat1.getElementType();
        const Type vt2 = mat2.getElementType();
        const ssize_t nr1 = mat1.getNumRows();
        const ssize_t nr2 = mat2.getNumRows();
        const ssize_t nc1 = mat1.getNumCols();
        const ssize_t nc2 = mat2.getNumCols();
        const ssize_t sp1 = mat1.getSparsity();
        const ssize_t sp2 = mat2.getSparsity();
        const daphne::MatrixRepresentation repr1 = mat1.getRepresentation();
        const daphne::MatrixRepresentation repr2 = mat2.getRepresentation();
        const BoolOrUnknown sym1 = mat1.getSymmetric();
        const BoolOrUnknown sym2 = mat2.getSymmetric();
        const ssize_t mncSketchId1 = mat1.getMncSketchId();
        const ssize_t mncSketchId2 = mat2.getMncSketchId();
        return daphne::MatrixType::get(ctx, (vt1 == vt2) ? vt1 : u, (nr1 == nr2) ? nr1 : -1, (nc1 == nc2) ? nc1 : -1,
                                       // TODO Maybe do approximate comparison of floating-point values.
                                       (sp1 == sp2) ? sp1 : -1,
                                       (repr1 == repr2) ? repr1 : daphne::MatrixRepresentation::Default,
                                       (sym1 == sym2) ? sym1 : BoolOrUnknown::Unknown,
                                       (mncSketchId1 == mncSketchId2) ? mncSketchId1 : -1);
    } else if (frm1 && frm2) { // both types are frames
        const std::vector<Type> cts1 = frm1.getColumnTypes();
        const std::vector<Type> cts2 = frm2.getColumnTypes();
        std::vector<Type> cts3;
        if (cts1.size() == cts2.size())
            for (size_t i = 0; i < cts1.size(); i++)
                cts3.push_back((cts1[i] == cts2[i]) ? cts1[i] : u);
        else
            // TODO How to represent a frame with unknown column
            // types? See #421.
            cts3.push_back(u);
        const ssize_t nr1 = frm1.getNumRows();
        const ssize_t nr2 = frm2.getNumRows();
        const ssize_t nc1 = frm1.getNumCols();
        const ssize_t nc2 = frm2.getNumCols();
        std::vector<std::string> *lbls1 = frm1.getLabels();
        std::vector<std::string> *lbls2 = frm2.getLabels();
        return daphne::FrameType::get(ctx, cts3, (nr1 == nr2) ? nr1 : -1, (nc1 == nc2) ? nc1 : -1,
                                      // TODO Take #485 into account.
                                      (lbls1 == lbls2) ? lbls1 : nullptr);
    } else if (mat1 || mat2 || frm1 || frm2) // t1 and t2 are of different data
                                             // types (matrix, frame, scalar)
        return nullptr;
    else // both types are unknown or scalars
        return (t1 == t2) ? t1 : u;
}
} // namespace

/**
 * @brief A compiler pass inferring various properties of the data objects.
 *
 * Rooted at a function, the pass walks the operations, and for each operation
 * it encounters, it infers all currently considered properties of the
 * operation's results based on the properties of the operation's arguments.
 * This approach can easily handle dependencies between different properties to
 * be inferred without explicitly modeling them.
 *
 * Note that the actual inference logic is outsourced to MLIR operation
 * interfaces.
 */
class InferencePass : public PassWrapper<InferencePass, OperationPass<func::FuncOp>> {
    daphne::InferenceConfig cfg;

    /**
     * @brief Sets all properties of all results of the given operation to
     * unknown to undo any prior property inference.
     *
     * The data/value types are retained.
     */
    std::function<WalkResult(Operation *)> walkSetUnknown = [&](Operation *op) {
        // For all other operations, we reset the types of all results to
        // unknown.
        for (size_t i = 0; i < op->getNumResults(); i++) {
            Type t = op->getResult(i).getType();
            if (auto mt = mlir::dyn_cast<daphne::MatrixType>(t))
                t = mt.withSameElementType();
            else if (auto ft = mlir::dyn_cast<daphne::FrameType>(t))
                t = ft.withSameColumnTypes();
            else if (auto ct = mlir::dyn_cast<daphne::ColumnType>(t))
                t = ct.withSameValueType();
            op->getResult(i).setType(t);
        }
        return WalkResult::advance();
    };

    /**
     * @brief Triggers the inference of all properties on the given operation.
     */
    std::function<WalkResult(Operation *)> walkOp = [&](Operation *op) {
        const bool isScfOp = op->getDialect() == op->getContext()->getOrLoadDialect<scf::SCFDialect>();

        // ----------------------------------------------------------------
        // Handle all non-control-flow (non-SCF) operations
        // ----------------------------------------------------------------
        if (llvm::isa<arith::SelectOp>(op)) {
            Type typeWithCommonInfo = getTypeWithCommonInfo(op->getOperand(1).getType(), op->getOperand(2).getType());
            if (!typeWithCommonInfo) {
                throw ErrorHandler::compilerError(op, "InferencePass.cpp:" + std::to_string(__LINE__),
                                                  " a variable must not be assigned values of "
                                                  "different data types (matrix, frame, scalar) "
                                                  "in then/else branches (arith.select)");
            }
            OpBuilder builder(op->getContext());
            castOperandIf(builder, op, 1, typeWithCommonInfo);
            castOperandIf(builder, op, 2, typeWithCommonInfo);
            op->getResult(0).setType(typeWithCommonInfo);
        } else if (!isScfOp) {
            if (cfg.typeInference && returnsUnknownType(op)) {
                // Try to infer the types of all results of this operation.
                try {
                    daphne::setInferredTypes(op, cfg.partialInferenceAllowed);
                } catch (std::runtime_error &re) {
                    throw ErrorHandler::rethrowError("InferencePass.cpp:" + std::to_string(__LINE__), re.what());
                }
            }
            if (cfg.shapeInference && returnsUnknownShape(op)) {
                // Try to infer the shapes of all results of this operation.
                std::vector<std::pair<ssize_t, ssize_t>> shapes = daphne::tryInferShape(op);
                const size_t numRes = op->getNumResults();
                if (shapes.size() != numRes) {
                    throw ErrorHandler::compilerError(op, "InferencePass.cpp:" + std::to_string(__LINE__),
                                                      "shape inference for op " + op->getName().getStringRef().str() +
                                                          " returned " + std::to_string(shapes.size()) +
                                                          " shapes, but the op has " + std::to_string(numRes) +
                                                          " results");
                }
                // Set the inferred shapes on all results of this operation.
                for (size_t i = 0; i < numRes; i++) {
                    if (llvm::isa<mlir::daphne::MatrixType>(op->getResultTypes()[i]) ||
                        llvm::isa<mlir::daphne::FrameType>(op->getResultTypes()[i])) {
                        const ssize_t numRows = shapes[i].first;
                        const ssize_t numCols = shapes[i].second;
                        Value rv = op->getResult(i);
                        const Type rt = rv.getType();
                        if (auto mt = mlir::dyn_cast<daphne::MatrixType>(rt))
                            rv.setType(mt.withShape(numRows, numCols));
                        else if (auto ft = mlir::dyn_cast<daphne::FrameType>(rt))
                            rv.setType(ft.withShape(numRows, numCols));
                        else
                            throw ErrorHandler::compilerError(op, "InferencePass.cpp:" + std::to_string(__LINE__),
                                                              "shape inference cannot set the shape of op " +
                                                                  op->getName().getStringRef().str() + " operand " +
                                                                  std::to_string(i) +
                                                                  ", since it "
                                                                  "is neither a matrix nor a frame");
                    }
                }
            }
            if (cfg.sparsityInference && returnsUnknownSparsity(op)) {
                // we infer mncSketchId before sparsity, since the former can be used to infer the latter
                if (cfg.mncSketchIdInference && returnsUnknownMncSketchId(op)) {
                    // Try to infer the mncSketchId of all results of this operation.
                    std::vector<int64_t> mncSketchIds = daphne::tryInferMncSketchId(op);
                    const size_t numRes = op->getNumResults();
                    if (mncSketchIds.size() != numRes)
                        throw ErrorHandler::compilerError(
                            op, "InferencePass",
                            "mncSketchId inference for op " + op->getName().getStringRef().str() + " returned " +
                                std::to_string(mncSketchIds.size()) + " entries, but the op has " + std::to_string(numRes) +
                                " results");
                    // Set the inferred values on all results of this operation.
                    for (size_t i = 0; i < numRes; i++) {
                        const int64_t mncSketchId = mncSketchIds[i];
                        if (llvm::isa<mlir::daphne::MatrixType>(op->getResultTypes()[i])) {
                            Value rv = op->getResult(i);
                            const Type rt = rv.getType();
                            if (auto mt = mlir::dyn_cast<daphne::MatrixType>(rt))
                                rv.setType(mt.withMncSketchId(mncSketchId));
                        }
                    }
                }
                // Try to infer the sparsity of all results of this operation.
                std::vector<double> sparsities = daphne::tryInferSparsity(op);
                const size_t numRes = op->getNumResults();
                if (sparsities.size() != numRes)
                    throw ErrorHandler::compilerError(op, "InferencePass",
                                                      "sparsity inference for op " +
                                                          op->getName().getStringRef().str() + " returned " +
                                                          std::to_string(sparsities.size()) +
                                                          " shapes, but the "
                                                          "op has " +
                                                          std::to_string(numRes) + " results");
                // Set the inferred sparsities on all results of this operation.
                for (size_t i = 0; i < numRes; i++) {
                    const double sparsity = sparsities[i];
                    if (llvm::isa<mlir::daphne::MatrixType>(op->getResultTypes()[i]) ||
                        llvm::isa<mlir::daphne::FrameType>(op->getResultTypes()[i])) {
                        Value rv = op->getResult(i);
                        const Type rt = rv.getType();
                        auto mt = mlir::dyn_cast<daphne::MatrixType>(rt);
                        auto ft = mlir::dyn_cast<daphne::FrameType>(rt);
                        if (mt)
                            rv.setType(mt.withSparsity(sparsity));
                        else if ((ft && sparsity != -1) || !ft)
                            // We do not support sparsity for frames, but if the
                            // sparsity for a frame result is provided as
                            // unknown (-1) that's okay.
                            throw ErrorHandler::compilerError(op, "InferencePass",
                                                              "sparsity inference cannot set the shape of "
                                                              "op " +
                                                                  op->getName().getStringRef().str() + " operand " +
                                                                  std::to_string(i) +
                                                                  ", since it "
                                                                  "is not a matrix");
                    }
                }
            }
            if (cfg.frameLabelInference && returnsFrameWithUnknownLabels(op)) {
                if (auto inferFrameLabelsOp = llvm::dyn_cast<daphne::InferFrameLabels>(op))
                    inferFrameLabelsOp.inferFrameLabels();
                // Else: Not a problem, since currently we use the frame labels
                // only to aid type inference, and for this purpose, we don't
                // need the labels in all cases.
            }
            if (cfg.symmetricInference && returnsUnknownSymmetric(op)) {
                // Try to infer the symmetry of all results of this operation.
                std::vector<BoolOrUnknown> symmetrics = daphne::tryInferSymmetric(op);
                const size_t numRes = op->getNumResults();
                if (symmetrics.size() != numRes)
                    throw ErrorHandler::compilerError(
                        op, "InferencePass",
                        "symmetric inference for op " + op->getName().getStringRef().str() + " returned " +
                            std::to_string(symmetrics.size()) + " entries, but the op has " + std::to_string(numRes) +
                            " results");
                // Set the inferred values on all results of this operation.
                for (size_t i = 0; i < numRes; i++) {
                    const BoolOrUnknown symmetric = symmetrics[i];
                    if (llvm::isa<mlir::daphne::MatrixType>(op->getResultTypes()[i])) {
                        Value rv = op->getResult(i);
                        const Type rt = rv.getType();
                        if (auto mt = mlir::dyn_cast<daphne::MatrixType>(rt))
                            rv.setType(mt.withSymmetric(symmetric));
                    }
                }
            }
        }
        // ----------------------------------------------------------------
        // Special treatment for some control-flow (SCF) operations
        // ----------------------------------------------------------------
        // The following control-flow operations require that certain SSA values
        // have the same MLIR type. For instance, for IfOp, the value yielded in
        // the then-branch and the value yielded in the else-branch must have
        // the same type in MLIR. At the same time, we encode interesting data
        // properties (such as those inferred by this pass) as MLIR type
        // parameters. As a consequence, e.g., a matrix with two rows and a
        // matrix with three rows are technically different MLIR types. Thus,
        // e.g., an IfOp cannot simply yield matrices of different shapes from
        // the then- and else-branches. To solve this general problem, and to
        // allow control-flow operations to change all properties of a data
        // object, we generally set mismatching properties to unknown. The
        // details depend on the specific SCF operation.
        else if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(op)) {
            Block &beforeBlock = whileOp.getBefore().front();
            Block &afterBlock = whileOp.getAfter().front();
            OpBuilder builder(whileOp.getContext());
            // Infer the types/properties inside the loop body. If some property
            // of some argument is changed inside the loop body, this property
            // is set to unknown for both the argument and the yielded value. If
            // that is the case, we need to do the inference anew, with the new
            // set of arguments' properties. This loop searches a fix-point and
            // always terminates, since we only set properties to unknown and in
            // the extreme case, after a finite number of iterations all of the
            // arguments' properties will have become unknown.
            while (true) {
                bool repeat = false;

                // Transfer the WhileOp's operand types to the block arguments
                // of the before-block to fulfill constraints on the WhileOp.
                for (size_t i = 0; i < whileOp.getNumOperands(); i++) {
                    Type t = whileOp->getOperand(i).getType();
                    beforeBlock.getArgument(i).setType(t);
                }

                // Continue the walk in the before-block, to infer the operand
                // types of the ConditionOp.
                beforeBlock.walk<WalkOrder::PreOrder>(walkOp);

                // Get the ConditionOp.
                Operation *condOp = beforeBlock.getTerminator();

                if (!llvm::isa<scf::ConditionOp>(condOp))
                    throw ErrorHandler::compilerError(op, "InferencePass", "WhileOp terminator is not a ConditionOp");

                // Transfer the ConditionOp's operand types to the block
                // arguments of the after-block and the results of the WhileOp
                // to fulfill constraints on the WhileOp. Note that the first
                // operand of the ConditionOp is skipped, since it is the
                // condition value itself.
                for (size_t i = 1; i < condOp->getNumOperands(); i++) {
                    Type t = condOp->getOperand(i).getType();
                    afterBlock.getArgument(i - 1).setType(t);
                    whileOp.getResult(i - 1).setType(t);
                }

                // Continue the walk in the after-block, to infer the operand
                // types of the YieldOp.
                afterBlock.walk<WalkOrder::PreOrder>(walkOp);

                // Get the YieldOp.
                Operation *yieldOp = afterBlock.getTerminator();

                if (whileOp->getNumOperands() != yieldOp->getNumOperands())
                    throw ErrorHandler::compilerError(op, "InferencePass",
                                                      "WhileOp and YieldOp must have the same number of "
                                                      "operands");

                // Check if the inferred MLIR types match the result MLIR types.
                // If any interesting properties were changed inside the loop
                // body, we set them to unknown to make the type comparison
                // pass.
                for (size_t i = 0; i < whileOp.getNumOperands(); i++) {
                    Type yieldedTy = yieldOp->getOperand(i).getType();
                    Type operandTy = op->getOperand(i).getType();
                    if (yieldedTy != operandTy) {
                        // Get a type with the conflicting properties set to
                        // unknown.
                        Type typeWithCommonInfo = getTypeWithCommonInfo(yieldedTy, operandTy);
                        if (!typeWithCommonInfo) {
                            throw ErrorHandler::compilerError(op, "InferencePass",
                                                              "the data type (matrix, frame, scalar) of a "
                                                              "variable "
                                                              "must not be changed within the body of a "
                                                              "while-loop");
                        }
                        // Use casts to remove those properties accordingly.
                        castOperandIf(builder, yieldOp, i, typeWithCommonInfo);
                        castOperandIf(builder, whileOp, i, typeWithCommonInfo);
                        // Since the WhileOp's argument types/properties have
                        // changed, we must repeat the inference for the loop
                        // body.
                        repeat = true;
                    }
                }
                if (repeat) {
                    // Before we can repeat the inference, we reset all
                    // information inferred so far to unknown (in the loop
                    // body).
                    beforeBlock.walk<WalkOrder::PreOrder>(walkSetUnknown);
                    afterBlock.walk<WalkOrder::PreOrder>(walkSetUnknown);
                } else
                    // If all types matched, we are done.
                    break;
            }
            // Tell the walker to skip the descendants of the WhileOp, we
            // have already triggered a walk on them explicitly.
            return WalkResult::skip();
        } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(op)) {
            Block &block = forOp.getRegion().front();
            const size_t numIndVars = forOp.getNumInductionVars();
            OpBuilder builder(forOp.getContext());
            // Infer the types/properties inside the loop body. If some property
            // of some argument is changed inside the loop body, this property
            // is set to unknown for both the argument and the yielded value. If
            // that is the case, we need to do the inference anew, with the new
            // set of arguments' properties. This loop searches a fix-point and
            // always terminates, since we only set properties to unknown and in
            // the extreme case, after a finite number of iterations all of the
            // arguments' properties will have become unknown.
            while (true) {
                bool repeat = false;

                // Transfer the ForOp's operand types to the block arguments
                // and results to fulfill constraints on the ForOp.
                for (size_t i = 0; i < forOp.getInitArgs().size(); i++) {
                    Type t = forOp.getInitArgs()[i].getType();
                    block.getArgument(i + numIndVars).setType(t);
                    forOp.getResult(i).setType(t);
                }

                // Continue the walk on the body block of the ForOp. We trigger
                // this explicitly, since we need to do something afterwards.
                block.walk<WalkOrder::PreOrder>(walkOp);

                // Get the YieldOp.
                Operation *yieldOp = block.getTerminator();

                // Check if the inferred MLIR types match the result MLIR types.
                // If any interesting properties were changed inside the loop
                // body, we set them to unknown to make the type comparison
                // pass.
                for (size_t i = 0; i < forOp.getInitArgs().size(); i++) {
                    Type yieldedTy = yieldOp->getOperand(i).getType();
                    Type resultTy = op->getResult(i).getType();
                    if (yieldedTy != resultTy) {
                        // Get a type with the conflicting properties set to
                        // unknown.
                        Type typeWithCommonInfo = getTypeWithCommonInfo(yieldedTy, resultTy);
                        if (!typeWithCommonInfo)
                            throw ErrorHandler::compilerError(op, "InferencePass.cpp:" + std::to_string(__LINE__),
                                                              "the data type (matrix, frame, scalar) of a "
                                                              "variable "
                                                              "must not be changed within the body of a "
                                                              "for-loop.");
                        // Use casts to remove those properties accordingly.
                        castOperandIf(builder, yieldOp, i, typeWithCommonInfo);
                        castOperandIf(builder, forOp, forOp.getNumControlOperands() + i, typeWithCommonInfo);
                        // Since the WhileOp's argument types/properties have
                        // changed, we must repeat the inference for the loop
                        // body.
                        repeat = true;
                    }
                }
                if (repeat)
                    // Before we can repeat the inference, we reset all
                    // information inferred so far to unknown (in the loop
                    // body).
                    block.walk<WalkOrder::PreOrder>(walkSetUnknown);
                else
                    // If all types matched, we are done.
                    break;
            }
            // Tell the walker to skip the descendants of the ForOp, we
            // have already triggered a walk on them explicitly.
            return WalkResult::skip();
        } else if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
            // Walk the then/else blocks first. We need the inference on
            // them before we can do anything about the IfOp itself.
            ifOp.thenBlock()->walk<WalkOrder::PreOrder>(walkOp);
            if (ifOp.elseBlock()) {
                ifOp.elseBlock()->walk<WalkOrder::PreOrder>(walkOp);

                // For all pairs of corresponding values yielded in the
                // then- and else-branch, determine a common type with
                // mismatching properties set to unknown, to make the
                // IfOp's type comparison pass.
                scf::YieldOp thenYield = ifOp.thenYield();
                scf::YieldOp elseYield = ifOp.elseYield();
                OpBuilder builder(ifOp.getContext());
                for (size_t i = 0; i < ifOp.getNumResults(); i++) {
                    Type typeWithCommonInfo =
                        getTypeWithCommonInfo(thenYield->getOperand(i).getType(), elseYield->getOperand(i).getType());
                    if (!typeWithCommonInfo)
                        throw ErrorHandler::compilerError(op, "InferencePass" + std::to_string(__LINE__),
                                                          "a variable must not be assigned values of "
                                                          "different data types (matrix, frame, scalar) "
                                                          "in then/else branches");
                    castOperandIf(builder, thenYield, i, typeWithCommonInfo);
                    castOperandIf(builder, elseYield, i, typeWithCommonInfo);
                    ifOp.getResult(i).setType(typeWithCommonInfo);
                }
            }
            // Tell the walker to skip the descendants of the IfOp, we
            // have already triggered a walk on them explicitly.
            return WalkResult::skip();
        }

        // Continue the walk normally.
        return WalkResult::advance();
    };

  public:
    InferencePass(daphne::InferenceConfig cfg) : cfg(cfg) {}

    void runOnOperation() override {
        func::FuncOp f = getOperation();
        try {
            daphne::setActiveMncSketchRegistry(&mncRegistry);
            f.walk<WalkOrder::PreOrder>(walkOp);
            // clear the registry to free memory and to avoid accidentally using it in later passes 
            // that might run in the same process without setting it up properly
            daphne::clearActiveMncSketchRegistry();
        } catch (std::runtime_error &re) {
            daphne::clearActiveMncSketchRegistry();
            throw ErrorHandler::rethrowError("InferencePass.cpp:" + std::to_string(__LINE__), re.what());
        }
        // infer function return types
        f.setType(FunctionType::get(&getContext(), f.getFunctionType().getInputs(),
                                    f.getBody().back().getTerminator()->getOperandTypes()));
    }

    static bool returnsUnknownType(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type resType) {
            if (llvm::isa<daphne::UnknownType>(resType))
                return true;
            if (auto mt = mlir::dyn_cast<daphne::MatrixType>(resType))
                return llvm::isa<daphne::UnknownType>(mt.getElementType());
            if (auto ft = mlir::dyn_cast<daphne::FrameType>(resType)) {
                for (Type ct : ft.getColumnTypes())
                    if (llvm::isa<daphne::UnknownType>(ct))
                        return true;
                return false;
            }
            if (auto ct = mlir::dyn_cast<daphne::ColumnType>(resType))
                return llvm::isa<daphne::UnknownType>(ct.getValueType());
            return false;
        });
    }

    static bool returnsFrameWithUnknownLabels(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type resultType) {
            auto ft = mlir::dyn_cast<daphne::FrameType>(resultType);
            return ft && !ft.getLabels();
        });
    }

    static bool returnsUnknownShape(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type rt) {
            if (auto mt = mlir::dyn_cast<daphne::MatrixType>(rt))
                return mt.getNumRows() == -1 || mt.getNumCols() == -1;
            if (auto ft = mlir::dyn_cast<daphne::FrameType>(rt))
                return ft.getNumRows() == -1 || ft.getNumCols() == -1;
            return false;
        });
    }

    static bool returnsUnknownSparsity(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type rt) {
            if (auto mt = mlir::dyn_cast<daphne::MatrixType>(rt))
                return mt.getSparsity() == -1.0;
            return false;
        });
    }

    static bool returnsUnknownSymmetric(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type rt) {
            if (auto mt = mlir::dyn_cast<daphne::MatrixType>(rt))
                return mt.getSymmetric() == BoolOrUnknown::Unknown;
            return false;
        });
    }
    static bool returnsUnknownMncSketchId(Operation *op) {
        return llvm::any_of(op->getResultTypes(), [](Type rt) {
            if (auto mt = mlir::dyn_cast<daphne::MatrixType>(rt))
                return mt.getMncSketchId() == -1;
            return false;
        });
    }

    StringRef getArgument() const final { return "inference"; }
    StringRef getDescription() const final { return "TODO"; }
};

std::unique_ptr<Pass> daphne::createInferencePass(daphne::InferenceConfig cfg) {
    return std::make_unique<InferencePass>(cfg);
}
