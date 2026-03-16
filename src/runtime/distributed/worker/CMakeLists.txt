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


# *****************************************************************************
# WorkerImpl library
# *****************************************************************************

# FIXME: use DaphneIrExecutor target instead
set(SOURCES 
        WorkerImpl.cpp 
        WorkerImplGRPCAsync.cpp
        WorkerImplGRPCSync.cpp
        ../../../compiler/execution/DaphneIrExecutor.cpp)

#source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

if(USE_HDFS)
        find_library(LIBHDFS3 NAMES hdfs3 HINTS ${PROJECT_BINARY_DIR}/installed/lib REQUIRED)
else()
        set(LIBHDFS3 "")
endif()

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRDaphne
        MLIRDaphneTransforms
        MLIRExecutionEngine
        MLIRReconcileUnrealizedCasts
        #DaphneIrExecutor
        DataStructures
        IO        
        CallData
        Proto
        DaphneCatalogParser
        ${LIBHDFS3}
        DaphneMetaDataParser
        Arrow::arrow_shared
        Parquet::parquet_shared
        DaphneConfigParser
        Util)

add_library(WorkerImpl ${SOURCES})
#llvm_update_compile_flags(WorkerImpl)
target_link_libraries(WorkerImpl PRIVATE ${LIBS})
#mlir_check_all_link_libraries(WorkerImpl)

# *****************************************************************************
# DistributedWorker executable
# *****************************************************************************

add_executable(DistributedWorker main.cpp)
set_target_properties(DistributedWorker PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
target_link_libraries(DistributedWorker PRIVATE Proto WorkerImpl MLIRDaphne)