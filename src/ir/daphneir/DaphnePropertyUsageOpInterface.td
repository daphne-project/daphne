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

#ifndef DAPHNE_PROPERTY_USAGE_INTERFACE_TD
#define DAPHNE_PROPERTY_USAGE_INTERFACE_TD
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def DataPropertyNonzeroCount : I32EnumAttrCase<"NonzeroCount", 0>;
def DataPropertyValueBounds : I32EnumAttrCase<"ValueBounds", 1>;
def DataPropertyDistinctCount : I32EnumAttrCase<"DistinctCount", 2>;
def DataPropertyNumericalSymmetry : I32EnumAttrCase<"NumericalSymmetry", 3>;

def DataPropertyKind : I32EnumAttr<"DataPropertyKind", "kind of data property", [
    DataPropertyNonzeroCount, DataPropertyValueBounds,
    DataPropertyDistinctCount, DataPropertyNumericalSymmetry
]> {
    let cppNamespace = "::mlir::daphne";
    let genSpecializedAttr = 0;
}

def PropertyUsageOpInterface : OpInterface<"PropertyUsage"> {
    let description = [{
        Declares operand properties that may be used for optimization.
    }];
    let methods = [InterfaceMethod<
        "Append property kinds for the specified operand index.",
        "void", "getUsedProperties",
        (ins "unsigned":$operandIndex,
             "::llvm::SmallVectorImpl<::mlir::daphne::DataPropertyKind> &":$properties)
    >];
}
#endif
