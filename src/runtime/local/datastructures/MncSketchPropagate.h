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

/* Single-step propagation (A * B)
Preforms a single step propogation for 2 the matrices A and B i.e(AB)
*It first checks if the simple diaganol condition is met if not, then we use
*non trvial method where the MNC Sparisity estimator is used to find the target
*NNZ and then calcualte the scaling factor
*Input : Sketches of hA and hB
*Output : Sketch of hC with the predicted sparsity of the result
*/

/*
Preforms a matrix transpose of a Matrix
*Achieves O(1) Time complexity as we do a simple pointer swap
*Input : Matrix A
*Output : Matrix C (Transposed version of Matrix A)
*/
MncSketch propagateTranspose(const MncSketch &hA);

MncSketch propagateRbind(const MncSketch &A, const MncSketch &B);
MncSketch propagateCbind(const MncSketch &A, const MncSketch &B);
MncSketch propagateMncFromDiagMatrix(const MncSketch &arg);

MncSketch propagateMncFromReshape(const MncSketch &A, std::size_t outM, std::size_t outN);
/*
*Handles the base cases where one of the matices is a simple square diagnal matrix i.e Indetity matrix
*For this case we can avoid the scaling and just copy the input sketch of the other matrix i.e non-identiy matrix
*Input : SKetches hA and hB
*Output : Sketch hC as the copy of the matrix B if diagnol condition is met
*Return : True if we have the exact case, otherwise False if we require scaling 
*/

// ----------------------------------------------------------------------------
//    MNC Sparsity Propagation
//    This module implements the propagation logic for the MNC (Matrix non zero count) sketches
//    With this module we are able to predict the intermediate sparsity and structure of a chain
//    of multiplication of matrices (ABC) without having to execute the whole operation, this
//    enables us to save both time and memory. The Mnc Sparsity Propagation uses the MNC Sparsity estimation
//    to get an inital density, it then scales the row and column histograms accordingly and then applies
//    probablistic rounding(to avoid always rouding down to zero)
// ----------------------------------------------------------------------------

/**
 * Helper function that propagates the histogram values (rows or columns) with rounding.
 * The scaled counts are derived from the ratio of output to input NNZ.
 * Probabilistic rounding is applied to prevent always rounding down to zero and
 * to maintain statistical accuracy.*
 * @param input The source Histogram from the input sketch.
 * @param output The target vector to be filled in the new sketch.
 * @param outNNZ Number of non-zero entries in the resulting vector.
 * 
*/
 void propagateVector(
    const std::vector<std::uint32_t>& input,
    std::vector<std::uint32_t>& output,
    double scale,
    std::uint32_t& outNNZ,    
    std::uint32_t& outMaxVal,  
    std::uint32_t& outEq1,     
    std::uint32_t& outGtHalf,  
    std::size_t halfThreshold, 
    std::mt19937& gen
);
// ----------------------------------------------------------------------------
// Exact propagation for Trivial cases (Diagonal matrices)
// ----------------------------------------------------------------------------
bool propagateExact(const MncSketch &hA, const MncSketch &hB, MncSketch &hC);
/*
*This estimates the sparsiy of a chain of matrix multiplications, it works recursively,
*by taking the result sketch of the first last multiplicatio and then using that as an
input for the next matrix in the chain, for example (AB*C)
*Input : A vector of MncSketches of the matrix chain
*Output : The final sketch for the multiplication chain
 */

/* Single-step propagation (A * B)
Preforms a single step propogation for 2 the matrices A and B i.e(AB)
*It first checks if the simple diaganol condition is met if not, then we use
*non trvial method where the MNC Sparisity estimator is used to find the target
*NNZ and then calcualte the scaling factor
*Input : Sketches of hA and hB
*Output : Sketch of hC with the predicted sparsity of the result
*/
MncSketch propagateMM(const MncSketch &hA, const MncSketch &hB);

// ----------------------------------------------------------------------------
// Chain propagation (multiple sketches)
// ----------------------------------------------------------------------------
MncSketch propagateChain(const std::vector<MncSketch> &chain);

MncSketch propagateAdd(const MncSketch &A, const MncSketch &B);

MncSketch propagateMul(const MncSketch &A, const MncSketch &B);