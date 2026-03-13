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
/**
 * Estimate the sparsity of the product of two matrices given their MNC sketches. 
 * Based on Algorithm 1 from the MNC paper.
 * @param hA MNC sketch of matrix A
 * @param hB MNC sketch of matrix B
 * @return Estimated sparsity of the product A * B
 * 
 */
double estimateSparsity_fromSketch(const MncSketch &mncSketch);
double estimateSparsity_product(const MncSketch &hA, const MncSketch &hB);

double estimateSparsity_ElementWiseAddition(const MncSketch &hA, const MncSketch &hB);

double estimateSparsity_ElementWiseMultiplication(const MncSketch &hA, const MncSketch &hB);