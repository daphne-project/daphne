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

#pragma once

// Only the forward declarations live here. The definition of distributedPipeline
// is in DistributedPipeline.cpp so that translation units using this kernel don't
// have to pull in all of DistributedWrapper.h (gRPC, protobuf, MPI, MLIR init-all-
// dialects). That brings the compile-time peak RSS of the generated kernels_*.cpp
// that reference this kernel down from ~3.6 GiB to ~1 GiB.

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cstddef>
#include <cstdint>

class Structure;

// One output. The template is instantiated only for DTRes = DenseMatrix<double>
// via kernels.json; that instantiation is provided out-of-line in
// DistributedPipeline.cpp.
template <class DTRes>
void distributedPipeline(DTRes **outputs, size_t numOutputs, const Structure **inputs, size_t numInputs,
                         int64_t *outRows, int64_t *outCols, int64_t *splits, int64_t *combines, const char *irCode,
                         DCTX(ctx));

// The explicit instantiations provided in DistributedPipeline.cpp. The extern
// template keeps each including TU from instantiating the template again, so it
// links against the one in the .cpp. checkExplicitInstantiations.py also checks
// this list against kernels.json at configure time.
extern template void distributedPipeline<DenseMatrix<double>>(DenseMatrix<double> **outputs, size_t numOutputs,
                                                              const Structure **inputs, size_t numInputs,
                                                              int64_t *outRows, int64_t *outCols, int64_t *splits,
                                                              int64_t *combines, const char *irCode,
                                                              DaphneContext *ctx);
