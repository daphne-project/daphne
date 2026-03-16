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

add_mlir_dialect_library(MLIRDaphneTransforms
    RewriteSqlOpPass.cpp
    DistributeComputationsPass.cpp
    DistributePipelinesPass.cpp
    MarkCUDAOpsPass.cpp
    MarkFPGAOPENCLOpsPass.cpp
    InsertDaphneContextPass.cpp
    ProfilingPass.cpp
    ManageObjRefsPass.cpp
    LowerToLLVMPass.cpp
    PhyOperatorSelectionPass.cpp
    RewriteToCallKernelOpPass.cpp
    RewriteToColumnarOpsPass.cpp
    SpecializeGenericFunctionsPass.cpp
    VectorizeComputationsPass.cpp
    HorizontalFusionPass.cpp
    DaphneOptPass.cpp

    DEPENDS
    MLIRDaphneOpsIncGen
    MLIRDaphneTransformsIncGen

    LINK_COMPONENTS
    Core
)
find_library(HWLOC_LIB NAMES hwloc HINTS ${PROJECT_BINARY_DIR}/installed/lib REQUIRED)

target_link_libraries(MLIRDaphneTransforms PUBLIC
    CompilerUtils
    Util
    MLIRSCFToControlFlow
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRAffineToStandard
    MLIRLinalgToStandard
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRFuncTransforms
    SQLParser
    ${HWLOC_LIB}
    fmt::fmt

)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()
