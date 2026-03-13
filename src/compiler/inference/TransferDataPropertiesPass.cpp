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

#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

/**
 * @brief Inserts `TransferPropertiesOp`s into the IR to ensure that compile-time information on the data properties of
 * DAPHNE data objects is transferred to the DAPHNE runtime.
 *
 * For each matrix-typed result of an operation, a `TransferPropertiesOp` is inserted that passes the compile-time
 * information on the matrix's data properties (extracted from the `MatrixType`) to the runtime. By inserting these
 * operations right after the creation of matrix-typed op results, we ensure that the runtime kernels subsequent
 * operations are lowered to have access to the property information. With this information available, the kernels can
 * do various runtime optimizations, e.g., choosing a more efficient algorithm on sorted or symmetric data.
 */
struct TransferDataPropertiesPass : public PassWrapper<TransferDataPropertiesPass, OperationPass<func::FuncOp>> {
    explicit TransferDataPropertiesPass() {}
    void runOnOperation() final;

    StringRef getArgument() const final { return "transfer-data-props"; }
    StringRef getDescription() const final { return "TODO"; }

    void processBlock(OpBuilder builder, Block *b) {
        for (Operation &op : b->getOperations()) {
            builder.setInsertionPointAfter(&op);
            Location loc = op.getLoc();
            for (Value v : op.getResults()) {
                Type t = v.getType();
                if (auto mt = mlir::dyn_cast<daphne::MatrixType>(t)) {
                    auto coSparsity = builder.create<daphne::ConstantOp>(loc, mt.getSparsity());
                    auto coSymmetric = builder.create<daphne::ConstantOp>(loc, static_cast<int64_t>(mt.getSymmetric()));
                    auto coMncSketchId = builder.create<daphne::ConstantOp>(loc, mt.getMncSketchId());
                    builder.create<daphne::TransferPropertiesOp>(loc, v, coSparsity, coSymmetric, coMncSketchId);
                }
            }

            // Recurse into the op, if it has regions.
            for (Region &r : op.getRegions())
                for (Block &b2 : r.getBlocks())
                    processBlock(builder, &b2);
        }
    }
};

void TransferDataPropertiesPass::runOnOperation() {
    func::FuncOp f = getOperation();
    OpBuilder builder(f.getContext());
    processBlock(builder, &(f.getBody().front()));
}

std::unique_ptr<Pass> daphne::createTransferDataPropertiesPass() {
    return std::make_unique<TransferDataPropertiesPass>();
}
