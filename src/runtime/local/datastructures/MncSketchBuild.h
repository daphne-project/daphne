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
#include <runtime/local/datastructures/MncSketch.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

/**
 * A function to build MNC sketch from a CSRMatrix
 * @param A The input CSRMatrix
 * @return The MNC sketch of A
*/
template<typename VT>
MncSketch buildMncFromCsrMatrix(const CSRMatrix<VT> &A);

/**
 * A function to build MNC sketch from a DenseMatrix
 * @param A The input DenseMatrix
 * @return The MNC sketch of A
*/
template<typename VT>
MncSketch buildMncFromDenseMatrix(const DenseMatrix<VT> &A);

// ----------------------------------------------------------------------------
// Source Operations (Rand, Seq, Fill)
// ----------------------------------------------------------------------------
MncSketch buildMncFromRand(std::size_t m, std::size_t n, double sparsity, std::int64_t seed);

MncSketch buildMncFromFill(double val, size_t rows, size_t cols);

MncSketch buildMncFromSeq(double start, double end, double step);