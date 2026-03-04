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

#include <runtime/local/datastructures/MncSketchEstimate.h>

/** This implementation of estimate Sparsity follows the pseudocode from the paper 
"MNC: Structure-Exploiting Sparsity Estimation for
Matrix Expressions" section 3.2 
*/
// Edm returns an estimated DENSITY in [0, 1] over the remaining 'p' cells.
// Caller can do: nnz += Edm(...) * p;
double EdmDensity(const std::vector<std::uint32_t>& hcA_res,
                  const std::vector<std::uint32_t>& hrB_res,
                  std::uint32_t p)
{
    if (p == 0) return 0.0;
    // std::cout << "hcA_res size: " << hcA_res.size() << ", hrB_res size: " << hrB_res.size() << ", p: " << p << std::endl;
    if (hcA_res.size() != hrB_res.size())
        throw std::invalid_argument("EdmDensity: vector sizes must match (same inner dimension).");

    // t = total number of candidate contributions ("hits")
    long double t = 0.0L;
    for (std::size_t k = 0; k < hcA_res.size(); ++k) {
        t += static_cast<long double>(hcA_res[k]) * static_cast<long double>(hrB_res[k]);
    }

    if (t <= 0.0L) return 0.0;

    // s = 1 - (1 - 1/p)^t  (use log/exp for numerical stability)
    const long double invP = 1.0L / static_cast<long double>(p);
    const long double logBase = std::log1p(-invP);           // log(1 - 1/p), negative
    const long double emptyProb = std::exp(t * logBase);    // (1 - 1/p)^t
    long double s = 1.0L - emptyProb;

    // Clamp due to floating point rounding
    if (s < 0.0L) s = 0.0L;
    if (s > 1.0L) s = 1.0L;
    return static_cast<double>(s);
}

double estimateSparsity_fromSketch(const MncSketch &s) {
    //"sparsity" == density == fraction of non-zeros in [0,1].
    if (s.m == 0 || s.n == 0)
        return 0.0;

    const bool haveHr = (s.hr && s.hr->size() == s.m);
    const bool haveHc = (s.hc && s.hc->size() == s.n);

    if (!haveHr && !haveHc)
        return -1.0; // no info, return "unknown"

    std::uint64_t sumHr = 0;
    std::uint64_t sumHc = 0;

    if (haveHr) {
        for (std::uint32_t v : *s.hr)
            sumHr += static_cast<std::uint64_t>(v);
    }
    if (haveHc) {
        for (std::uint32_t v : *s.hc)
            sumHc += static_cast<std::uint64_t>(v);
    }

    std::uint64_t nnz = 0;
    if (haveHr && haveHc) {
        // In a real matrix sketch these should be equal.
        // If not (e.g. synthetic generator), take a stable compromise.
        nnz = (sumHr == sumHc) ? sumHr : (sumHr + sumHc) / 2;
    }
    else {
        nnz = haveHr ? sumHr : sumHc;
    }

    const long double denom = static_cast<long double>(s.m) * static_cast<long double>(s.n);
    const long double density = (denom > 0.0L) ? (static_cast<long double>(nnz) / denom) : 0.0L;

    return static_cast<double>(std::clamp(density, 0.0L, 1.0L));
}

double estimateSparsity_product(const MncSketch &hA, const MncSketch &hB) {
    const std::size_t m = hA.m;
    const std::size_t l = hB.n;

    //Fix use size_t instead of double for avoiding counting errors
    std::size_t exact_nnz = 0;
    double prob_nnz = 0.0;

    // Case 1: Exact count
    if(hA.maxHr <= 1 || hB.maxHc <= 1) {
        for(std::size_t j = 0; j < hA.n; ++j)
            exact_nnz += static_cast<std::size_t>((*hA.hc)[j]) * static_cast<std::size_t>((*hB.hr)[j]);
    }

    // Case 2: Extended count
    else if(hA.her && hB.her) { 
        
        // Fused (Exact Part 1 + Exact Part 2)
        for(std::size_t k = 0; k < hA.n; ++k) {
            // Term 1: hA^ec * hB^r
            exact_nnz += static_cast<std::size_t>((*hA.hec)[k]) * static_cast<std::size_t>((*hB.hr)[k]);

            // Term 2: hB^er * (hA^c - hA^ec)
            // Check to ensure positive result before subtraction
            if ((*hA.hc)[k] > (*hA.hec)[k]) {
                exact_nnz += static_cast<std::size_t>((*hB.her)[k]) * (static_cast<std::size_t>((*hA.hc)[k]) - static_cast<std::size_t>((*hA.hec)[k]));
            }
        }

        // Remaining uncertain cells (Probabilistic part)
        std::size_t p = (hA.nnzRows - hA.rowsEq1) * (hB.nnzCols - hB.colsEq1);

        if(p > 0) {
            std::vector<uint32_t> hcA_res;
            std::vector<uint32_t> hrB_res;
            // Pre-allocate memory 
            hcA_res.reserve(hA.n);
            hrB_res.reserve(hB.m);

            for(std::size_t j = 0; j < hA.n; ++j) {
                std::uint32_t hcA_val = static_cast<std::uint32_t>((*hA.hc)[j]);
                // Safety check to ensure we don't underflow if something is wrong
                std::uint32_t sub = static_cast<std::uint32_t>((*hA.hec)[j]);
                hcA_res.push_back((hcA_val > sub) ? (hcA_val - sub) : 0);
            }
            for(std::size_t i = 0; i < hB.m; ++i) {
                std::uint32_t hrB_val = static_cast<std::uint32_t>((*hB.hr)[i]);
                std::uint32_t sub = static_cast<std::uint32_t>((*hB.her)[i]);
                hrB_res.push_back((hrB_val > sub) ? (hrB_val - sub) : 0);
            }
            double dens = EdmDensity(hcA_res, hrB_res, p);
            prob_nnz += dens * static_cast<double>(p);
        }
    }

    // Case 3: Generic fallback
    else {
        std::size_t p = hA.nnzRows * hB.nnzCols;
        if(p > 0) {
            double dens = EdmDensity(*hA.hc, *hB.hr, p);
            prob_nnz = dens * static_cast<double>(p);
        }
    }

   // Combine exact and probabilistic counts
   double totalNNZ = static_cast<double>(exact_nnz) + prob_nnz;

   // Lower bound
   std::size_t lower = hA.rowsGtHalf * hB.colsGtHalf;

   if(totalNNZ < static_cast<double>(lower))
       totalNNZ = static_cast<double>(lower); 

   return totalNNZ / static_cast<double>(m * l);
}

/** This implementation of estimate Sparsity for element-wise
addition and multiplication follows the paper 
"MNC: Structure-Exploiting Sparsity Estimation for
Matrix Expressions" section 4.1 
*/

//nnzOverlap returns Lambda = (sum_j hcA[j] * hcB[j])/ (nnz(A) * nnz(B))
double nnzOverlap(const MncSketch &hA, const MncSketch &hB) { 
    // Compute nnz(A) and nnz(B) 
    uint64_t nnzA = 0; 
    uint64_t nnzB = 0; 
    
    for(uint32_t v : *hA.hc) nnzA += v;
    for(uint32_t v : *hB.hc) nnzB += v;

    if(nnzA == 0 || nnzB == 0) {
        return 0.0;
    }
    // numerator = sum_j hcA[j] * hcB[j] 
    long double numerator = 0.0L; 

    for(size_t j = 0; j < hA.n; ++j) { 
        numerator += static_cast<long double>((*hA.hc)[j]) * 
            static_cast<long double>((*hB.hc)[j]); 
        } 

    // denominator = nnz(A) * nnz(B)
    long double denom = 
        static_cast<long double>(nnzA) * 
        static_cast<long double>(nnzB); 

    return static_cast<double>(numerator / denom); 
}

/*     
* Estimate sparsity of C = A + B (element-wise addition)
*/
double estimateSparsity_ElementWiseAddition(const MncSketch &hA, const MncSketch &hB) { 
    const std::size_t m = hA.m;
    const std::size_t n = hA.n;

    const long double lambda = nnzOverlap(hA, hB); 

    long double total = 0.0L; 

    for (std::size_t i = 0; i < m; ++i) { 
        long double a = (*hA.hr)[i];
        long double b = (*hB.hr)[i];

        total += a + b - (a * b * lambda);
    } 
    
    long double s = total / static_cast<long double>(m * n); 
        
    return static_cast<double>(s); 
}

/*     
* Estimate sparsity of C = A ⊙ B (element-wise multiplication)
*/
double estimateSparsity_ElementWiseMultiplication(const MncSketch &hA, const MncSketch &hB) {
    const std::size_t m = hA.m;
    const std::size_t n = hA.n;

    const long double lambda = nnzOverlap(hA, hB);

    long double total = 0.0L;

    for (std::size_t i = 0; i < m; ++i) {
        long double a = (*hA.hr)[i];
        long double b = (*hB.hr)[i];

        total += a * b * lambda;  
    }

    long double s = total / static_cast<long double>(m * n);

    return static_cast<double>(s);
}