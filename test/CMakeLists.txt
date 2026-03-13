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

include_directories(${PROJECT_SOURCE_DIR}/test)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/catch2) # for "catch.hpp"

set(TEST_SOURCES
        run_tests.h
        run_tests.cpp

        api/cli/algorithms/AlgorithmsTest.cpp
        api/cli/algorithms/DecisionTreeRandomForestTest.cpp
        api/cli/config/ConfigTest.cpp
        api/cli/controlflow/ControlFlowTest.cpp
        api/cli/distributed/DistributedTest.cpp
        api/cli/expressions/CastTest.cpp
        api/cli/expressions/CondTest.cpp
        api/cli/expressions/FrameLiteralTest.cpp
        api/cli/expressions/MatrixLiteralTest.cpp
        api/cli/extensibility/ExtensionTest.cpp
        api/cli/extensibility/HintTest.cpp
        api/cli/functions/FunctionsTest.cpp
        api/cli/functions/RecursiveFunctionsTest.cpp
        api/cli/import/ImportTest.cpp
        api/cli/indexing/IndexingTest.cpp
        api/cli/inference/InferenceTest.cpp
        api/cli/io/ReadWriteTest.cpp
        api/cli/lists/ListsTest.cpp
        api/cli/literals/LiteralsTest.cpp
        api/cli/operations/ConstantFoldingTest.cpp
        api/cli/operations/OperationsTest.cpp
        api/cli/operations/TypeOfTest.cpp
        api/cli/parser/ParserTest.cpp
        api/cli/parser/MetaDataParserTest.cpp
        api/cli/scoping/ScopingTest.cpp
        api/cli/scriptargs/ScriptArgsTest.cpp
        api/cli/secondorder/SecondOrderTest.cpp
        api/cli/sql/ColumnarTest.cpp
        api/cli/sql/SQLTest.cpp
        api/cli/sql/SQLResultTest.cpp
        api/cli/syntax/SyntaxTest.cpp
        api/cli/vectorized/MultiThreadedOpsTest.cpp
        api/cli/vectorized/VectorizedPipelineTest.cpp
        api/cli/utils/UtilsTest.cpp
        api/cli/Utils.cpp

        api/python/DaphneLibTest.cpp

        codegen/CodegenTest.cpp
        api/cli/codegen/AggAllTest.cpp
        # api/cli/codegen/AggDimTest.cpp
        # api/cli/codegen/EwBinaryTest.cpp
        # api/cli/codegen/EwUnaryTest.cpp
        # api/cli/codegen/EwOpLoopFusionTest.cpp
        # api/cli/codegen/MapOpTest.cpp
        # api/cli/codegen/SparsityExploitTest.cpp
        # api/cli/codegen/TransposeTest.cpp

        ir/daphneir/InferTypesTest.cpp
        api/cli/operations/CanonicalizationConstantFoldingOpTest.cpp

        parser/config/ConfigParserTest.cpp

        runtime/distributed/worker/WorkerTest.cpp

        runtime/local/datastructures/CSRMatrixTest.cpp
        runtime/local/datastructures/DenseMatrixTest.cpp
        runtime/local/datastructures/MncSketchTest.cpp
        runtime/local/datastructures/FrameTest.cpp
        runtime/local/datastructures/MatrixTest.cpp
        runtime/local/datastructures/TaskQueueTest.cpp
        runtime/local/datastructures/TensorTest.cpp

        runtime/local/io/ReadCsvTest.cpp
        runtime/local/io/ReadParquetTest.cpp
        runtime/local/io/ReadMMTest.cpp
        runtime/local/io/WriteDaphneTest.cpp
        runtime/local/io/ReadDaphneTest.cpp
        runtime/local/io/DaphneSerializerTest.cpp

        runtime/local/kernels/AggAllTest.cpp
        runtime/local/kernels/AggColTest.cpp
        runtime/local/kernels/AggCumTest.cpp
        runtime/local/kernels/AggRowTest.cpp
        runtime/local/kernels/BinTest.cpp
        runtime/local/kernels/CartesianTest.cpp
        runtime/local/kernels/CastObjTest.cpp
        runtime/local/kernels/CastObjScaTest.cpp
        runtime/local/kernels/CastScaTest.cpp
        runtime/local/kernels/CastScaObjTest.cpp
        runtime/local/kernels/CheckEqTest.cpp
        runtime/local/kernels/CheckEqApproxTest.cpp
        runtime/local/kernels/ColAggAllTest.cpp
        runtime/local/kernels/ColAggGrpTest.cpp
        runtime/local/kernels/ColBindTest.cpp
        runtime/local/kernels/ColCalcBinaryTest.cpp
        runtime/local/kernels/ColGroupFirstTest.cpp
        runtime/local/kernels/ColGroupNextTest.cpp
        runtime/local/kernels/ColIntersectTest.cpp
        runtime/local/kernels/ColJoinTest.cpp
        runtime/local/kernels/ColMergeTest.cpp
        runtime/local/kernels/ColProjectTest.cpp
        runtime/local/kernels/ColSelectCmpTest.cpp
        runtime/local/kernels/ColSemiJoinTest.cpp
        runtime/local/kernels/CondMatMatMatTest.cpp
        runtime/local/kernels/CondMatMatScaTest.cpp
        runtime/local/kernels/CondMatScaMatTest.cpp
        runtime/local/kernels/CondMatScaScaTest.cpp
        runtime/local/kernels/ConvertBitmapToPosListTest.cpp
        runtime/local/kernels/ConvertPosListToBitmapTest.cpp
        runtime/local/kernels/CreateFrameTest.cpp
        runtime/local/kernels/CTableTest.cpp
        runtime/local/kernels/DiagMatrixTest.cpp
        runtime/local/kernels/DiagVectorTest.cpp
        runtime/local/kernels/DNNPoolingTestForward.cpp
        runtime/local/kernels/DNNConv2DBackwardTest.cpp
        runtime/local/kernels/DNNConv2DForwardTest.cpp
        runtime/local/kernels/DNNBatchNorm2DForwardTest.cpp
        runtime/local/kernels/DNNBatchNorm2DBackwardTest.cpp
        runtime/local/kernels/DNNBiasAddForwardTest.cpp
        runtime/local/kernels/DNNPoolingBackwardTest.cpp
        runtime/local/kernels/EigenCalTest.cpp
        runtime/local/kernels/EwBinaryMatTest.cpp
        runtime/local/kernels/EwBinaryObjScaTest.cpp
        runtime/local/kernels/EwBinaryScaTest.cpp
        runtime/local/kernels/EwUnaryMatTest.cpp
        runtime/local/kernels/EwUnaryScaTest.cpp
        runtime/local/kernels/ExtractColTest.cpp
        runtime/local/kernels/ExtractRowTest.cpp
        runtime/local/kernels/FillTest.cpp
        runtime/local/kernels/FilterColTest.cpp
        runtime/local/kernels/FilterRowTest.cpp
        runtime/local/kernels/GroupJoinTest.cpp
        runtime/local/kernels/GroupTest.cpp
        runtime/local/kernels/HasSpecialValueTest.cpp
        runtime/local/kernels/InnerJoinTest.cpp
        runtime/local/kernels/InsertColTest.cpp
        runtime/local/kernels/InsertRowTest.cpp
        runtime/local/kernels/IsSymmetricTest.cpp
        runtime/local/kernels/NumDistinctApproxTest.cpp
        runtime/local/kernels/MapTest.cpp
        runtime/local/kernels/MatMulTest.cpp
        runtime/local/kernels/OneHotTest.cpp
        runtime/local/kernels/OrderTest.cpp
        runtime/local/kernels/OuterBinaryTest.cpp
        runtime/local/kernels/QuantizeTest.cpp
        runtime/local/kernels/RandMatrixTest.cpp
        runtime/local/kernels/ReadTest.cpp
        runtime/local/kernels/RecodeTest.cpp
        runtime/local/kernels/ReplaceTest.cpp
        runtime/local/kernels/ReshapeTest.cpp
        runtime/local/kernels/ReverseTest.cpp
        runtime/local/kernels/RowBindTest.cpp
        runtime/local/kernels/SampleTest.cpp
        runtime/local/kernels/SemiJoinTest.cpp
        runtime/local/kernels/SeqTest.cpp
        runtime/local/kernels/SetColLabelsTest.cpp
        runtime/local/kernels/SetColLabelsPrefixTest.cpp
        runtime/local/kernels/SliceColTest.cpp
        runtime/local/kernels/SliceRowTest.cpp
        runtime/local/kernels/SoftmaxTest.cpp
        runtime/local/kernels/SolveTest.cpp
        runtime/local/kernels/StopTest.cpp
        runtime/local/kernels/SyrkTest.cpp
        runtime/local/kernels/ThetaJoinTest.cpp
        runtime/local/kernels/TransposeTest.cpp
        runtime/local/kernels/TriTest.cpp

        runtime/local/vectorized/MultiThreadedKernelTest.cpp

        util/RecordPropertiesTest.cpp
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    list(APPEND TEST_SOURCES
            runtime/local/kernels/CUDA/BiasAddTest.cpp
            runtime/local/kernels/CUDA/FillTest.cpp
            runtime/local/kernels/CUDA/MatMulTest.cpp
            runtime/local/kernels/CUDA_ContextTest.cpp
            runtime/local/kernels/CUDA/DNNActivationTest.cpp
            runtime/local/kernels/CUDA/DNNAffineTest.cpp
            runtime/local/kernels/CUDA/DNNBatchNormTest.cpp
            runtime/local/kernels/CUDA/DNNConvolutionTest.cpp
            runtime/local/kernels/CUDA/DNNSoftmaxTest.cpp)
endif()

add_executable(run_tests ${TEST_SOURCES})
set_target_properties(run_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_dependencies(run_tests daphne daphnelib DistributedWorker daphne-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBS AllKernels ${dialect_libs} DataStructures DaphneDSLParser MLIRDaphne WorkerImpl Proto DaphneConfigParser
        DaphneMetaDataParser Util)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_include_directories(run_tests PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    # search "custom" cudnn lib in CUDA SDK dir
    set(lib_name cudnn)
    find_library(CUDA_${lib_name}_LIBRARY NAMES ${lib_name} HINTS ${CUDAToolkit_LIBRARY_DIR} ENV CUDA_PATH
            PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

    list(APPEND LIBS CUDAKernels CUDA::cudart CUDA::cublasLt CUDA::cublas CUDA::cusparse ${CUDA_cudnn_LIBRARY})
endif()

target_link_libraries(run_tests PRIVATE ${LIBS})
target_link_directories(run_tests PRIVATE ${PROJECT_BINARY_DIR}/lib)
add_executable(theta_join_test run_tests.cpp run_tests.h runtime/local/kernels/ThetaJoinTest.cpp)
set_target_properties(theta_join_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_dependencies(theta_join_test daphne DistributedWorker)
target_link_libraries(theta_join_test PRIVATE ${LIBS})
target_compile_options(theta_join_test PUBLIC -g -O0)
