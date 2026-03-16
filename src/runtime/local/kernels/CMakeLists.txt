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

# Specifies how to generate the file "kernels.cpp" (which resides in the build
# directory) as the basis for the pre-compiled kernels library.

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# Apple Clang needs Homebrew's libomp for OpenMP support
if(APPLE)
    execute_process(COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
        set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
    endif()
endif()
find_package(OpenMP REQUIRED)

# The library of pre-compiled CUDA kernels
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    execute_process(
      COMMAND
        ${Python3_EXECUTABLE} genKernelInst.py kernels.json
        ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/CUDAkernels
        ${PROJECT_SOURCE_DIR}/lib/CUDAcatalog.json CUDA
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/)

  file(GLOB CUDA_CODEGEN_CPP_FILES CONFIGURE_DEPENDS
      "${PROJECT_BINARY_DIR}/src/runtime/local/kernels/CUDA*.cpp")

    set(PREFIX ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/CUDA)
    set(CUDAKernels_SRC
            ${PREFIX}/../../instrumentation/KernelInstrumentation.cpp
            ${PREFIX}/../../context/CUDAContext.cpp
            ${PREFIX}/CreateCUDAContext.cpp
            ${PREFIX}/AggAll.cu
            ${PREFIX}/AggCol.cu
            ${PREFIX}/AggRow.cu
            ${PREFIX}/Activation.cpp
            ${PREFIX}/Affine.cpp
            ${PREFIX}/BatchNorm.cpp
            ${PREFIX}/BiasAdd.cpp
            ${PREFIX}/Convolution.cpp
            ${PREFIX}/DeviceUtils.cuh
            ${PREFIX}/Fill.cu
            ${PREFIX}/Pooling.cpp
            ${PREFIX}/Softmax.cpp
            ${PREFIX}/ColBind.cu
            ${PREFIX}/EwBinaryMat.cu
            ${PREFIX}/EwBinaryObjSca.cu
            ${PREFIX}/ExtractCol.cu
            ${PREFIX}/Gemv.cpp
            ${PREFIX}/MatMul.cpp
            ${PREFIX}/Solve.cpp
            ${PREFIX}/Syrk.cu
            ${PREFIX}/Transpose.cpp
            ${CUDA_CODEGEN_CPP_FILES}
            ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/TasksCUDA.cpp
            ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/VectorizedPipeline.h
            ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/WorkerGPU.h
    )

    add_library(CUDAKernels SHARED ${CUDAKernels_SRC})

    # search "custom" cudnn lib in CUDA SDK dir
    set(lib_name cudnn)
    find_library(CUDA_${lib_name}_LIBRARY NAMES ${lib_name} HINTS ${CUDAToolkit_LIBRARY_DIR} ENV CUDA_PATH
            PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)

    target_link_libraries(CUDAKernels PUBLIC DataStructures LLVMSupport MLIRDaphne MLIRDaphneTransforms CUDA::cudart CUDA::cublasLt CUDA::cublas
            CUDA::cusparse ${CUDA_cudnn_LIBRARY} CUDA::cusolver Util MLIRDaphneInference fmt::fmt)
    set_target_properties(CUDAKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
    set_property(TARGET CUDAKernels PROPERTY CUDA_ARCHITECTURES all)
endif()

execute_process(
  COMMAND
    ${Python3_EXECUTABLE} genKernelInst.py kernels.json
    ${PROJECT_BINARY_DIR}/src/runtime/local/kernels/kernels
    ${PROJECT_SOURCE_DIR}/lib/catalog.json CPP
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/)

file(GLOB CODEGEN_CPP_FILES CONFIGURE_DEPENDS
     "${PROJECT_BINARY_DIR}/src/runtime/local/kernels/kernels_*.cpp")
# message("CODEGEN_CPP_FILES: ${CODEGEN_CPP_FILES}")

list(APPEND LIBS DataStructures IO BLAS::BLAS MLIRParser OpenMP::OpenMP_CXX)

set(PREFIX ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/)
set(HEADERS_cpp_kernels
    ${PREFIX}/MatMul.h
)

set(SOURCES_cpp_kernels
        ${PREFIX}/MatMul.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/instrumentation/KernelInstrumentation.cpp
        ${CODEGEN_CPP_FILES}
        ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/CreateDaphneContext.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/Pooling.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/kernels/VectorizedPipeline.h
        ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/MTWrapper_dense.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/MTWrapper_sparse.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/Tasks.cpp
        ${PROJECT_SOURCE_DIR}/src/runtime/local/vectorized/WorkerCPU.h
        )
# The library of pre-compiled kernels. Will be linked into the JIT-compiled user program.
add_library(KernelObjLib OBJECT ${SOURCES_cpp_kernels} ${HEADERS_cpp_kernels})
set_target_properties(KernelObjLib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(AllKernels SHARED $<TARGET_OBJECTS:KernelObjLib>)
set_target_properties(AllKernels PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_include_directories(AllKernels PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    list(APPEND LIBS CUDAKernels)
else()
    # This can only appear once, so it's either included in CUDAKernels or appended here to AllKernels directly
    list(APPEND LIBS LLVMSupport Util MLIRDaphneInference)
endif()


list(APPEND LIBS DaphneMetaDataParser MLIRDaphne MLIRDaphneTransforms)
list(APPEND LIBS Eigen3::Eigen Arrow::arrow_shared Parquet::parquet_shared)

if(USE_PAPI)
    find_library(PAPI_LIB NAMES papi HINTS ${PROJECT_BINARY_DIR}/installed/lib REQUIRED)
endif()

find_library(HWLOC_LIB NAMES hwloc HINTS ${PROJECT_BINARY_DIR}/installed/lib REQUIRED)

if(USE_HDFS)
        target_include_directories(AllKernels PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty/installed/include/hdfs)
        find_library(LIBHDFS3 NAMES hdfs3 HINTS ${PROJECT_BINARY_DIR}/installed/lib REQUIRED)
endif()

target_link_libraries(KernelObjLib PUBLIC ${LIBS} ${MPI_LIBRARIES} ${PAPI_LIB} ${HWLOC_LIB} ${LIBHDFS3})
target_link_libraries(AllKernels PUBLIC ${LIBS} ${MPI_LIBRARIES} ${PAPI_LIB} ${HWLOC_LIB} ${LIBHDFS3})
