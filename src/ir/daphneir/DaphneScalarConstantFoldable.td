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

#ifndef SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_TD
#define SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_TD

include "mlir/IR/OpBase.td"

def ScalarConstantFoldable : OpInterface<"ScalarConstantFoldable"> {
    let description = [{
        Marker interface for scalar ops whose constant-folding semantics can be
        expressed as pure per-kind lambdas over `APInt` / `APFloat` / `bool`.

        Rather than every op reimplementing operand-unwrap, type promotion and
        result-attribute construction, an op provides only the arithmetic per
        numeric kind (via the methods below) and the shared driver
        `foldScalarOp()` handles the mechanics. Unimplemented kinds return
        `std::nullopt`; the driver tries kinds in the same precedence order the
        legacy `Fold.cpp` templates used (float, then signed int, unsigned int,
        bool), first non-empty result wins.

        Comparison ops (numeric input, bool output) implement the `foldScalarCmp*`
        variants: same shape, different return type.

        Not on the interface (kept bespoke in `Fold.cpp`):
          - String concatenation / string comparison: 2 out of ~20 folders, so
            interface pollution outweighs the saved boilerplate.
          - Division / modulo by zero: raised through the per-op lambda by
            calling `ErrorHandler::compilerError`, not centralised in the driver.
    }];

    let methods = [
        // ------------------------------------------------------------------
        // Binary arithmetic
        // ------------------------------------------------------------------
        InterfaceMethod<
            "Fold two float operands. Return nullopt if this op has no float semantics.",
            "std::optional<llvm::APFloat>", "foldScalarFloat",
            (ins "const llvm::APFloat &":$a, "const llvm::APFloat &":$b),
            /*methodBody=*/[{}],
            /*defaultImpl=*/[{ return std::nullopt; }]
        >,
        InterfaceMethod<
            "Fold two signed-integer operands. Return nullopt if unsupported.",
            "std::optional<llvm::APInt>", "foldScalarSInt",
            (ins "const llvm::APInt &":$a, "const llvm::APInt &":$b),
            [{}], [{ return std::nullopt; }]
        >,
        InterfaceMethod<
            "Fold two unsigned-integer operands. Return nullopt if unsupported.",
            "std::optional<llvm::APInt>", "foldScalarUInt",
            (ins "const llvm::APInt &":$a, "const llvm::APInt &":$b),
            [{}], [{ return std::nullopt; }]
        >,
        InterfaceMethod<
            "Fold two bool operands. Return nullopt if unsupported.",
            "std::optional<bool>", "foldScalarBool",
            (ins "bool":$a, "bool":$b),
            [{}], [{ return std::nullopt; }]
        >,

        // ------------------------------------------------------------------
        // Unary arithmetic
        // ------------------------------------------------------------------
        InterfaceMethod<
            "Fold one float operand. Return nullopt if unsupported.",
            "std::optional<llvm::APFloat>", "foldScalarUnaryFloat",
            (ins "const llvm::APFloat &":$a),
            [{}], [{ return std::nullopt; }]
        >,
        InterfaceMethod<
            "Fold one integer operand (works for both signednesses). Return nullopt if unsupported.",
            "std::optional<llvm::APInt>", "foldScalarUnaryInt",
            (ins "const llvm::APInt &":$a),
            [{}], [{ return std::nullopt; }]
        >,

        // ------------------------------------------------------------------
        // Binary comparison (numeric in, bool out)
        // ------------------------------------------------------------------
        InterfaceMethod<
            "Compare two floats. Return nullopt if unsupported.",
            "std::optional<bool>", "foldScalarCmpFloat",
            (ins "const llvm::APFloat &":$a, "const llvm::APFloat &":$b),
            [{}], [{ return std::nullopt; }]
        >,
        InterfaceMethod<
            "Compare two signed ints. Return nullopt if unsupported.",
            "std::optional<bool>", "foldScalarCmpSInt",
            (ins "const llvm::APInt &":$a, "const llvm::APInt &":$b),
            [{}], [{ return std::nullopt; }]
        >,
        InterfaceMethod<
            "Compare two unsigned ints. Return nullopt if unsupported.",
            "std::optional<bool>", "foldScalarCmpUInt",
            (ins "const llvm::APInt &":$a, "const llvm::APInt &":$b),
            [{}], [{ return std::nullopt; }]
        >,
        InterfaceMethod<
            "Compare two bools. Return nullopt if unsupported.",
            "std::optional<bool>", "foldScalarCmpBool",
            (ins "bool":$a, "bool":$b),
            [{}], [{ return std::nullopt; }]
        >,
    ];
}

#endif // SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_TD
