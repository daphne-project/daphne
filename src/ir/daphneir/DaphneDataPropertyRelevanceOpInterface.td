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

#ifndef DAPHNE_DATA_PROPERTY_RELEVANCE_INTERFACE_TD
#define DAPHNE_DATA_PROPERTY_RELEVANCE_INTERFACE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def DataPropertySparsity : I32EnumAttrCase<"Sparsity", 0>;
def DataPropertyBounds : I32EnumAttrCase<"Bounds", 1>;
def DataPropertyNumDistinct : I32EnumAttrCase<"NumDistinct", 2>;
def DataPropertySymmetric : I32EnumAttrCase<"Symmetric", 3>;

def DataPropertyNumRows : I32EnumAttrCase<"NumRows", 4>;
def DataPropertyNumCols : I32EnumAttrCase<"NumCols", 5>;

def DataPropertyKind : I32EnumAttr<"DataPropertyKind", "kind of data property", [
    DataPropertySparsity, DataPropertyBounds,
    DataPropertyNumDistinct, DataPropertySymmetric,
    DataPropertyNumRows, DataPropertyNumCols
]> {
    let cppNamespace = "::mlir::daphne";
    let genSpecializedAttr = 0;
}

def DataPropertyRelevanceOpInterface : OpInterface<"DataPropertyRelevance"> {
    let description = [{
        Declares input data properties that runtime implementations of an operation
        may use to choose algorithms or specialized execution paths. The query
        identifies relevant properties for an operand.
    }];
    let methods = [InterfaceMethod<
        [{Append relevant data-property kinds for the specified operand index.
          The index must identify an existing operand. An empty list means that
          no relevance is declared.}],
        "void", "getRelevantProperties",
        (ins "unsigned":$operandIndex,
             "::llvm::SmallVectorImpl<::mlir::daphne::DataPropertyKind> &":$properties)
    >];
}
#endif
