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
#include <compiler/utils/LoweringUtils.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>
#include <util/ErrorHandler.h>

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

/**
 * @brief Inserts DaphneIR operations for managing the reference counters of
 * runtime data objects.
 *
 * Thus, it takes care of freeing data objects (e.g., intermediate results) at
 * the right points. The operations employed for reference management are
 * `IncRefOp` and `DecRefOp`.
 *
 * The core ideas are:
 * - We decrease the reference counter of each SSA value (block argument or
 *   op result) to prevent memory leaks.
 * - We do this as soon as possible. That is, after the last use of the value,
 *   or directly after its definition, if it has no uses.
 * - As the only exception: We do not decrease the reference of the arguments
 *   to block terminators.
 * - Whenever a value is duplicated (e.g., by passing it as a block argument),
 *   we increase the reference of the underlying data object. This is to ensure
 *   that decreasing the reference on the new value does not destroy a data
 *   object that is still needed in a surrounding scope, i.e., to prevent
 *   double frees.
 */
struct ManageObjRefsPass : public PassWrapper<ManageObjRefsPass, OperationPass<func::FuncOp>> {
    explicit ManageObjRefsPass() {}
    void runOnOperation() final;

    StringRef getArgument() const final { return "manage-obj-refs"; }
    StringRef getDescription() const final { return "TODO"; }
};

void processMemRefInterop(OpBuilder builder, Value v) {
    Operation *lastUseOp = findLastUseOfSSAValue(v);

    builder.setInsertionPointAfter(lastUseOp);
    builder.create<daphne::DecRefOp>(v.getLoc(), v.getDefiningOp()->getOperand(0));
}

/**
 * @brief Inserts a `DecRefOp` in the right place, to decrease the reference
 * counter of the given value.
 *
 * @param builder
 * @param v
 */
void processValue(OpBuilder builder, Value v) {
    // TODO Address handles from the distributed runtime (but they will be
    // removed soon anyway).
    // We only need to manage the reference counters of DAPHNE data objects
    // like matrices and frames (not of scalars).

    Operation *defOp = v.getDefiningOp();
    if (defOp && llvm::isa<daphne::ConvertDenseMatrixToMemRef>(defOp))
        processMemRefInterop(builder, v);

    // Increase the reference counter of string literals, such that they don't
    // get gargabe collected.
    if (defOp && llvm::isa<daphne::ConstantOp>(defOp) && llvm::isa<daphne::StringType>(v.getType())) {
        // The given value is a string literal. We want to increase its
        // reference counter right after its definition, such that it is never
        // removed. But if the defining op is the block of a FuncOp, make sure
        // not to insert the IncRefOp before the CreateDaphneContextOp,
        // otherwise we will run into problems during/after lowering to kernel
        // calls.
        Block *pb = v.getParentBlock();
        if (auto fo = dyn_cast<func::FuncOp>(pb->getParentOp())) {
            Value dctx = CompilerUtils::getDaphneContext(fo);
            builder.setInsertionPointAfterValue(dctx);
        } else
            builder.setInsertionPointAfter(defOp);
        builder.create<daphne::IncRefOp>(v.getLoc(), v);
    }

    // Increase the reference counter of the result of the arith.select op, if
    // it is a string scalar. This is necessary because for arith.select, we
    // have no clue which of its two arguments (2nd or 3rd one) it will return.
    // Unless we do something about it, the reference counter of the result will
    // be too low by 1. Thus, we increase the result's reference counter here.
    if (defOp && llvm::isa<arith::SelectOp>(defOp) &&
        llvm::isa<daphne::MatrixType, daphne::FrameType, daphne::ListType, daphne::StringType>(v.getType())) {
        builder.setInsertionPointAfter(defOp);
        builder.create<daphne::IncRefOp>(v.getLoc(), v);
    }

    if (!llvm::isa<daphne::MatrixType, daphne::FrameType, daphne::ListType, daphne::StringType>(v.getType()))
        return;

    Operation *decRefAfterOp = nullptr;
    if (v.use_empty()) {
        // If the given SSA value has no uses, we want to decrease its
        // reference counter directly after its definition (nullptr for block
        // args). Note that ideally, there should be no unused SSA values.
        if (defOp)
            decRefAfterOp = defOp;
        // else: decRefAfterOp stays nullptr
    } else {
        // If the given SSA value has uses, we need to find the last of them.
        // Note that the iterator over the uses provided by the value does not
        // seem to follow any useful order, in general, so we need to find out
        // which use is the last one.
        // Furthermore, we want to decrease the reference counter of the SSA
        // value in the block where the value was defined, to simplify things.
        // So if the user of the value is in a descendant block, we need to
        // find its parent op in the block where the given value is defined.
        decRefAfterOp = findLastUseOfSSAValue(v);
    }

    // At this point, decRefAfterOp is nullptr, or the last user of v, or the
    // defining op of v.

    if (decRefAfterOp) {
        // The given value is used and/or an OpResult.

        // Don't insert a DecRefOp if the last user is a terminator.
        if (decRefAfterOp->hasTrait<OpTrait::IsTerminator>())
            // The value is handed out of its block (e.g., return, yield, ...).
            // So a new reference to it is created. Thus, the reference counter
            // must remain unchanged. Moreover, it is impossible to insert any
            // op after the terminator.
            return;
        // TODO Remove this workaround once the refactoring of the distributed
        // runtime is on the main branch.
        // Don't insert a DecRefOp if there is already one. Currently, this can
        // happen only on the distributed worker, since the IR it gets already
        // contains
        if (llvm::isa<daphne::DecRefOp>(decRefAfterOp))
            return;

        builder.setInsertionPointAfter(decRefAfterOp);
    } else {
        // The given value is an unused block arg. Decrease its reference
        // counter at the beginning of the block.
        // But if this is the block of a FuncOp, make sure not to insert the
        // DecRefOp before the CreateDaphneContextOp, otherwise we will run
        // into problems during/after lowering to kernel calls.
        Block *pb = v.getParentBlock();
        if (auto fo = dyn_cast<func::FuncOp>(pb->getParentOp())) {
            Value dctx = CompilerUtils::getDaphneContext(fo);
            builder.setInsertionPointAfterValue(dctx);
        } else
            builder.setInsertionPointToStart(pb);
    }

    // Finally create the DecRefOp.
    builder.create<daphne::DecRefOp>(v.getLoc(), v);
}

/**
 * @brief Inserts an `IncRefOp` for the given value if its type is a DAPHNE
 * data type (matrix, frame, list, string).
 *
 * If the type is unknown, throw an exception.
 *
 * @param v
 * @param b
 */
void incRefIfObj(Value v, OpBuilder &b) {
    Type t = v.getType();
    if (llvm::isa<daphne::MatrixType, daphne::FrameType, daphne::ListType, daphne::StringType>(t))
        b.create<daphne::IncRefOp>(v.getLoc(), v);
    else if (llvm::isa<daphne::UnknownType>(t))
        throw ErrorHandler::compilerError(v.getDefiningOp(), "ManageObjRefsPass",
                                          "ManageObjRefsPass encountered a value of unknown type, so it "
                                          "cannot know if it is a data object.");
}

/**
 * @brief Inserts an `IncRefOp` for each operand of the given operation whose
 * type is a DAPHNE data type (matrix, frame, list, string), right before the
 * operation.
 *
 * @param op
 * @param b
 */
void incRefArgs(Operation &op, OpBuilder &b) {
    b.setInsertionPoint(&op);
    for (Value arg : op.getOperands())
        incRefIfObj(arg, b);
}

/**
 * @brief Manages the reference counters of all values defined in the given
 * block by inserting `IncRefOp` and `DecRefOp` in the right places.
 *
 * @param builder
 * @param b
 */
void processBlock(OpBuilder builder, Block *b) {
    // Make sure that the reference counters of block arguments are decreased.
    for (BlockArgument &arg : b->getArguments())
        processValue(builder, arg);

    // Make sure the reference counters of op results are decreased, and
    // Increase the reference counters of operands where necessary.
    for (Operation &op : b->getOperations()) {
        // 1) Increase the reference counters of operands, if necessary.

        // TODO We could use traits to identify those cases.

        // Casts that will not call a kernel.
        if (auto co = dyn_cast<daphne::CastOp>(op)) {
            if (co.isTrivialCast() || co.isRemovePropertyCast())
                incRefArgs(op, builder);
        }
        // Loops and function calls.
        else if (llvm::isa<scf::WhileOp, scf::ForOp, func::CallOp, daphne::GenericCallOp>(op))
            incRefArgs(op, builder);
        // YieldOp of IfOp.
        else if (llvm::isa<scf::YieldOp>(op) && llvm::isa<scf::IfOp>(op.getParentOp())) {
            // Increase the reference counters of data objects that already
            // existed before the IfOp, because yielding them creates a new
            // SSA value referring to them.
            builder.setInsertionPoint(&op);
            for (Value arg : op.getOperands())
                if (arg.getParentBlock() != op.getBlock())
                    incRefIfObj(arg, builder);
        }
        // Terminators.
        else if (op.hasTrait<OpTrait::IsTerminator>()) {
            // By default, we do not decrease the reference counter of a
            // terminator's argument. If the same value is used multiple times
            // as an argument, we need to increase its reference counter.
            builder.setInsertionPoint(&op);
            for (size_t i = 1; i < op.getNumOperands(); i++) {
                Value arg = op.getOperand(i);
                for (size_t k = 0; k < i; k++)
                    if (arg == op.getOperand(k))
                        incRefIfObj(arg, builder);
            }
        }
        // Vectorized pipelines.
        //   Note: We do not increase the reference counters of the arguments
        //   of vectorized pipelines, because internally, a pipeline processes
        //   views into its inputs. These are individual data objects.

        // 2) Make sure the reference counters of op results are decreased.
        for (Value v : op.getResults())
            processValue(builder, v);

        // 3) Recurse into the op, if it has regions.
        for (Region &r : op.getRegions())
            for (Block &b2 : r.getBlocks())
                processBlock(builder, &b2);
    }
}

void ManageObjRefsPass::runOnOperation() {
    func::FuncOp f = getOperation();
    OpBuilder builder(f.getContext());
    processBlock(builder, &(f.getBody().front()));
}

std::unique_ptr<Pass> daphne::createManageObjRefsPass() { return std::make_unique<ManageObjRefsPass>(); }
