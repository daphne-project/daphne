# Copyright 2021 The DAPHNE Consortium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_mlir_dialect(DaphneOps daphne) # creates the target MLIRDaphneOpsIncGen
add_mlir_doc(DaphneOps -gen-dialect-doc DaphneDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS DaphneOps.td)
mlir_tablegen(DaphneOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(DaphneOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRDaphneOpsEnumsIncGen)
add_dependencies(mlir-headers MLIRDaphneOpsEnumsIncGen)

add_mlir_interface(DaphneDistributableOpInterface)
add_mlir_interface(DaphneInferFrameLabelsOpInterface)
add_mlir_interface(DaphneInferSymmetricOpInterface)
add_mlir_interface(DaphneInferShapeOpInterface)
add_mlir_interface(DaphneInferSparsityOpInterface)
add_mlir_interface(DaphneInferTypesOpInterface)
add_mlir_interface(DaphneVectorizableOpInterface)
add_mlir_interface(DaphneScalarConstantFoldable)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Daphne)
add_public_tablegen_target(MLIRDaphneTransformsIncGen)

set(LLVM_TARGET_DEFINITIONS DaphneCanonicalizePatterns.td)
mlir_tablegen(DaphneCanonicalizePatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRDaphneCanonicalizePatternsIncGen)

add_mlir_doc(Passes -gen-pass-doc DaphnePasses Dialects/)

add_mlir_dialect_library(MLIRDaphne
    DaphneDialect.cpp
    Fold.cpp
    Canonicalize.cpp
    DataPropertyAccessors.cpp
    DaphneAlgebraicTraitPatterns.cpp
    DaphneCanonicalizePatterns.cpp
    DaphneDistributableOpInterface.cpp
    DaphneInferFrameLabelsOpInterface.cpp
    DaphneInferSymmetricOpInterface.cpp
    DaphneInferShapeOpInterface.cpp
    DaphneInferSparsityOpInterface.cpp
    DaphneInferTypesOpInterface.cpp
    DaphneVectorizableOpInterface.cpp
    DaphneScalarConstantFoldable.cpp

    DEPENDS
    MLIRDaphneOpsIncGen
    MLIRDaphneTransformsIncGen
    MLIRDaphneCanonicalizePatternsIncGen
    MLIRDaphneDistributableOpInterfaceIncGen
    MLIRDaphneInferFrameLabelsOpInterfaceIncGen
    MLIRDaphneInferSymmetricOpInterfaceIncGen
    MLIRDaphneInferShapeOpInterfaceIncGen
    MLIRDaphneInferSparsityOpInterfaceIncGen
    MLIRDaphneInferTypesOpInterfaceIncGen
    MLIRDaphneVectorizableOpInterfaceIncGen
    MLIRDaphneScalarConstantFoldableIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    CompilerUtils
    DataStructures
    Util
)
