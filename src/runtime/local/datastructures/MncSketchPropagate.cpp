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
#include <runtime/local/datastructures/MncSketchPropagate.h>
#include <runtime/local/datastructures/MncSketchEstimate.h>

MncSketch propagateTranspose(const MncSketch &hA){
    MncSketch hC;
    // Swap Dimensions
    hC.m = hA.n;
    hC.n = hA.m;

    //Rows become collumns, collums become rows
    hC.hr = hA.hc;
    hC.hc = hA.hr;

    //Extended Vectors swap
    hC.her = hA.hec;
    hC.hec = hA.her;

    //Swap characteristics
    hC.maxHr = hA.maxHc;
    hC.maxHc = hA.maxHr;
    hC.nnzRows = hA.nnzCols;
    hC.nnzCols = hA.nnzRows;
    hC.rowsEq1 = hA.colsEq1;
    hC.colsEq1 = hA.rowsEq1;
    hC.rowsGtHalf = hA.colsGtHalf;
    hC.colsGtHalf = hA.rowsGtHalf;

    // Keeping the diaganal property
    hC.isDiagonal = hA.isDiagonal;
    return hC;
}
static std::uint64_t sumVec(const std::shared_ptr<std::vector<std::uint32_t>> &v) {
    if(!v) return 0;
    return std::accumulate(v->begin(), v->end(), std::uint64_t{0});
}

static std::uint64_t totalNnz(const MncSketch &A) {
    // Prefer hr if valid, else hc, else 0
    if (A.hr && A.hr->size() == A.m) return sumVec(A.hr);
    if (A.hc && A.hc->size() == A.n) return sumVec(A.hc);
    return 0;
}
static bool hasHr(const MncSketch &S) { return S.hr && S.hr->size() == S.m; }
static bool hasHc(const MncSketch &S) { return S.hc && S.hc->size() == S.n; }
static bool hasHer(const MncSketch &S) { return S.her && S.her->size() == S.m; }
static bool hasHec(const MncSketch &S) { return S.hec && S.hec->size() == S.n; }

static std::uint32_t sat_u32(std::uint64_t v) {
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(v, std::numeric_limits<std::uint32_t>::max())
    );
}

// If hec isn't materialized and maxHr<=1, then all nnz lie in singleton rows => hec == hc.
static std::uint32_t getHecVal(const MncSketch &S, std::size_t j) {
    if (hasHec(S)) return (*S.hec)[j];
    if (S.maxHr <= 1 && hasHc(S)) return (*S.hc)[j];
    return 0;
}

// If her isn't materialized and maxHc<=1, then all nnz lie in singleton cols => her == hr.
static std::uint32_t getHerVal(const MncSketch &S, std::size_t i) {
    if (hasHer(S)) return (*S.her)[i];
    if (S.maxHc <= 1 && hasHr(S)) return (*S.hr)[i];
    return 0;
}

static void computeSummary(MncSketch &S) {
    S.maxHr = 0; S.maxHc = 0;
    S.nnzRows = 0; S.nnzCols = 0;
    S.rowsEq1 = 0; S.colsEq1 = 0;
    S.rowsGtHalf = 0; S.colsGtHalf = 0;

    if (hasHr(S)) {
        for (std::size_t i = 0; i < S.m; i++) {
            auto v = (*S.hr)[i];
            S.maxHr = std::max(S.maxHr, v);
            if (v > 0 && S.nnzRows < std::numeric_limits<std::uint32_t>::max()) S.nnzRows++;
            if (v == 1 && S.rowsEq1 < std::numeric_limits<std::uint32_t>::max()) S.rowsEq1++;
            if (v > S.n / 2 && S.rowsGtHalf < std::numeric_limits<std::uint32_t>::max()) S.rowsGtHalf++;
        }
    }
    if (hasHc(S)) {
        for (std::size_t j = 0; j < S.n; j++) {
            auto v = (*S.hc)[j];
            S.maxHc = std::max(S.maxHc, v);
            if (v > 0 && S.nnzCols < std::numeric_limits<std::uint32_t>::max()) S.nnzCols++;
            if (v == 1 && S.colsEq1 < std::numeric_limits<std::uint32_t>::max()) S.colsEq1++;
            if (v > S.m / 2 && S.colsGtHalf < std::numeric_limits<std::uint32_t>::max()) S.colsGtHalf++;
        }
    }
}

/**
 * rbind(lhs, rhs): vertical concat, requires same number of columns
 * Paper: hr = concat(hrA, hrB), hc = hcA + hcB, her = 0, hec = hecA + hecB
 */
MncSketch propagateRbind(const MncSketch &A, const MncSketch &B) {
    if (A.n != B.n)
        throw std::runtime_error("rbind: lhs and rhs must have the same number of columns");

    MncSketch C;
    C.m = A.m + B.m;
    C.n = A.n;
    C.isDiagonal = false;

    C.hr = std::make_shared<std::vector<std::uint32_t>>(C.m, 0);
    C.hc = std::make_shared<std::vector<std::uint32_t>>(C.n, 0);

    // hr^C = rbind(hr^A, hr^B)
    if (hasHr(A)) {
        for (std::size_t i = 0; i < A.m; i++) (*C.hr)[i] = (*A.hr)[i];
    }
    if (hasHr(B)) {
        for (std::size_t i = 0; i < B.m; i++) (*C.hr)[A.m + i] = (*B.hr)[i];
    }

    // hc^C = hc^A + hc^B
    for (std::size_t j = 0; j < C.n; j++) {
        std::uint64_t a = hasHc(A) ? (*A.hc)[j] : 0;
        std::uint64_t b = hasHc(B) ? (*B.hc)[j] : 0;
        (*C.hc)[j] = sat_u32(a + b);
    }

    computeSummary(C);

    // extended vectors only if you follow your convention
    if (C.maxHr > 1 || C.maxHc > 1) {
        C.her = std::make_shared<std::vector<std::uint32_t>>(C.m, 0); // h_er^C = 0
        C.hec = std::make_shared<std::vector<std::uint32_t>>(C.n, 0);

        // h_ec^C = h_ec^A + h_ec^B (with implicit hec==hc when maxHr<=1)
        for (std::size_t j = 0; j < C.n; j++) {
            std::uint64_t a = getHecVal(A, j);
            std::uint64_t b = getHecVal(B, j);
            (*C.hec)[j] = sat_u32(a + b);
        }
    }

    return C;
}

/**
 * cbind(lhs, rhs): horizontal concat, requires same number of rows
 * Paper (symmetric): hr = hrA + hrB, hc = concat(hcA, hcB), her = herA + herB, hec = 0
 */
MncSketch propagateCbind(const MncSketch &A, const MncSketch &B) {
    if (A.m != B.m)
        throw std::runtime_error("cbind: lhs and rhs must have the same number of rows");

    MncSketch C;
    C.m = A.m;
    C.n = A.n + B.n;
    C.isDiagonal = false;

    C.hr = std::make_shared<std::vector<std::uint32_t>>(C.m, 0);
    C.hc = std::make_shared<std::vector<std::uint32_t>>(C.n, 0);

    // hr^C = hr^A + hr^B
    for (std::size_t i = 0; i < C.m; i++) {
        std::uint64_t a = hasHr(A) ? (*A.hr)[i] : 0;
        std::uint64_t b = hasHr(B) ? (*B.hr)[i] : 0;
        (*C.hr)[i] = sat_u32(a + b);
    }

    // hc^C = cbind(hc^A, hc^B)
    if (hasHc(A)) {
        for (std::size_t j = 0; j < A.n; j++) (*C.hc)[j] = (*A.hc)[j];
    }
    if (hasHc(B)) {
        for (std::size_t j = 0; j < B.n; j++) (*C.hc)[A.n + j] = (*B.hc)[j];
    }

    computeSummary(C);

    if (C.maxHr > 1 || C.maxHc > 1) {
        C.her = std::make_shared<std::vector<std::uint32_t>>(C.m, 0);
        C.hec = std::make_shared<std::vector<std::uint32_t>>(C.n, 0); // h_ec^C = 0

        // h_er^C = h_er^A + h_er^B (with implicit her==hr when maxHc<=1)
        for (std::size_t i = 0; i < C.m; i++) {
            std::uint64_t a = getHerVal(A, i);
            std::uint64_t b = getHerVal(B, i);
            (*C.her)[i] = sat_u32(a + b);
        }
    }

    return C;
}

MncSketch propagateMncFromDiagMatrix(const MncSketch &arg) {
    // arg must be (n x 1)
    if (arg.n != 1)
        throw std::runtime_error("diagMatrix: arg must be a (n x 1) column-matrix");

    const std::size_t n = arg.m;

    MncSketch out;
    out.m = n;
    out.n = n;

    out.isDiagonal = true;

    out.hr = std::make_shared<std::vector<std::uint32_t>>(n, 0);
    out.hc = std::make_shared<std::vector<std::uint32_t>>(n, 0);

    for (std::size_t i = 0; i < n; i++) {
        const std::uint32_t nz = ((*arg.hr)[i] > 0) ? 1u : 0u;
        (*out.hr)[i] = nz;
        (*out.hc)[i] = nz;
    }

    computeSummary(out);
    return out;
}
MncSketch propagateMncFromReshape(const MncSketch &A, std::size_t outM, std::size_t outN) {
    // Cell count must be preserved.
    const __int128 inCells  = static_cast<__int128>(A.m) * static_cast<__int128>(A.n);
    const __int128 outCells = static_cast<__int128>(outM) * static_cast<__int128>(outN);
    if (inCells != outCells) {
        throw std::runtime_error("reshape: number of cells must be retained");
    }

    MncSketch C;
    C.m = outM;
    C.n = outN;
    C.isDiagonal = (outM == A.m && outN == A.n) ? A.isDiagonal : false;

    // Always allocate core vectors.
    C.hr = std::make_shared<std::vector<std::uint32_t>>(outM, 0);
    C.hc = std::make_shared<std::vector<std::uint32_t>>(outN, 0);

    // Reshape does not preserve extended vectors in general -> do not propagate them.
    C.her.reset();
    C.hec.reset();

    // Handle degenerate shapes
    if (outM == 0 || outN == 0) {
        computeSummary(C);
        return C;
    }

    // Need input vectors for the exact (paper) case.
    const bool haveHrA = A.hr && A.hr->size() == A.m;
    const bool haveHcA = A.hc && A.hc->size() == A.n;

    const std::uint64_t nnz = totalNnz(A);

    // --- Exact row-wise concatenation case from the MNC paper ---
    // They focus on: reshape m×n -> k×l with m mod k = 0 (k = outM),
    // so each output row is a concatenation of g = m/k input rows. :contentReference[oaicite:2]{index=2}
    if (haveHrA && haveHcA && outM > 0 && (A.m % outM == 0)) {
        const std::size_t g = A.m / outM; // input rows per output row
        // For row-wise reshape, l = n * g must hold.
        if (outN == A.n * g) {
            // hr_C: sum every g input rows (exact in this case)
            for (std::size_t r = 0; r < outM; r++) {
                std::uint64_t acc = 0;
                for (std::size_t t = 0; t < g; t++) {
                    acc += (*A.hr)[r * g + t];
                }
                (*C.hr)[r] = static_cast<std::uint32_t>(acc);
            }

            // hc_C: scale+replicate column counts (paper says rep(round(hcA/g), g)).
            // We do an equivalent deterministic split that preserves totals exactly.
            // For each original column j, distribute hcA[j] across g replicated columns.
            for (std::size_t j = 0; j < A.n; j++) {
                const std::uint32_t x = (*A.hc)[j];
                const std::uint32_t base = (g == 0) ? 0 : (x / static_cast<std::uint32_t>(g));
                const std::uint32_t rem  = (g == 0) ? 0 : (x % static_cast<std::uint32_t>(g));
                for (std::size_t t = 0; t < g; t++) {
                    (*C.hc)[t * A.n + j] = base + (t < rem ? 1u : 0u);
                }
            }

            computeSummary(C);
            return C;
        }
    }

    // --- Fallback (best-effort): preserve nnz and spread uniformly ---
    // (This is needed when reshape splits rows, or dimensions don't align nicely.)
    {
        const std::uint64_t baseR = nnz / outM;
        const std::uint64_t remR  = nnz % outM;
        for (std::size_t i = 0; i < outM; i++) {
            (*C.hr)[i] = static_cast<std::uint32_t>(baseR + (i < remR ? 1 : 0));
        }

        const std::uint64_t baseC = nnz / outN;
        const std::uint64_t remC  = nnz % outN;
        for (std::size_t j = 0; j < outN; j++) {
            (*C.hc)[j] = static_cast<std::uint32_t>(baseC + (j < remC ? 1 : 0));
        }

        computeSummary(C);
        return C;
    }
}

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
) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (std::size_t i = 0; i < input.size(); ++i) {
        double val = input[i] * scale;
        std::uint32_t count = static_cast<std::uint32_t>(val);
        // Probabilistic rounding
        if ((val - count) > dis(gen)) {
            count++;
        }
        output[i] = count;
        // Update statistics
        if (count > 0) {
            outNNZ++;
            if (count > outMaxVal) 
                outMaxVal = count;
            if (count == 1) 
                outEq1++;              
            if (count > halfThreshold) 
                outGtHalf++; 
        }
    }
}

bool propagateExact(const MncSketch &hA, const MncSketch &hB, MncSketch &hC) {
    // Case 1: A is diagonal square
    if (hA.isDiagonal && hA.m == hA.n && hA.nnzRows == hA.m) {
        hC = hB;
        return true;
    }
    // Case 2: B is diagonal square
    if (hB.isDiagonal && hB.m == hB.n && hB.nnzCols == hB.n) {
        hC = hA;
        return true;
    }
    return false;
}

MncSketch propagateMM(const MncSketch &hA, const MncSketch &hB) {
    // 1. Try exact propagation first
    MncSketch hC;
    if (propagateExact(hA, hB, hC)) {
        return hC;
    }

    // 2. Prepare approximate propagation
    hC.m = hA.m;
    hC.n = hB.n;
    //Changes (Abdullah)
    hC.hr = std::make_shared<std::vector<std::uint32_t>>(hC.m, 0);
    hC.hc = std::make_shared<std::vector<std::uint32_t>>(hC.n, 0);

    // FIX: Removed thread_local to simplify
    static std::random_device rd;
    static std::mt19937 gen(rd());

    double sparsity = estimateSparsity_product(hA, hB);
    double targetTotalNNZ = sparsity * hC.m * hC.n;

    //Changes (Abdullah)
    std::uint32_t totalRows = std::accumulate(hA.hr->begin(), hA.hr->end(), 0U);
    std::uint32_t totalCols = std::accumulate(hB.hc->begin(), hB.hc->end(), 0U);

    double rowScale = (totalRows > 0) ? (targetTotalNNZ / totalRows) : 0.0;
    double colScale = (totalCols > 0) ? (targetTotalNNZ / totalCols) : 0.0;

    // 3. Propagate Rows
    propagateVector(
        *hA.hr,          // Change (Abdullah)
        *hC.hr,          // Change (Abdullah)
        rowScale,        
        hC.nnzRows,     
        hC.maxHr,        
        hC.rowsEq1,      
        hC.rowsGtHalf,   
        hC.n / 2,        
        gen
    );

    // 4. Propagate Columns
    propagateVector(
        *hB.hc,          // Change (Abdullah)
        *hC.hc,          // Change (Abdullah)
        colScale,        
        hC.nnzCols,      
        hC.maxHc,        
        hC.colsEq1,      
        hC.colsGtHalf,   
        hC.m / 2,        
        gen
    );

    return hC;
}

MncSketch propagateChain(const std::vector<MncSketch> &chain) {
    if (chain.empty()) return MncSketch();
    
    MncSketch currentResult = chain[0];
    for (size_t i = 1; i < chain.size(); ++i) {
        currentResult = propagateMM(currentResult, chain[i]);
    }
    return currentResult;
}

/*Element-wise Addition propagation (see section 4.2 from "MNC: Structure-Exploiting Sparsity Estimation for
Matrix Expressions" paper)*/
MncSketch propagateAdd(const MncSketch &A, const MncSketch &B) {
    MncSketch C;
    C.m = A.m;
    C.n = A.n;
    C.hr = std::make_shared<std::vector<uint32_t>>(C.m, 0);
    C.hc = std::make_shared<std::vector<uint32_t>>(C.n, 0);

    double sparsity = estimateSparsity_ElementWiseAddition(A, B);
    double targetNNZ = sparsity * C.m * C.n;

    double totalRowA = std::accumulate(A.hr->begin(), A.hr->end(), 0U);
    double totalRowB = std::accumulate(B.hr->begin(), B.hr->end(), 0U);
    double totalColA = std::accumulate(A.hc->begin(), A.hc->end(), 0U);
    double totalColB = std::accumulate(B.hc->begin(), B.hc->end(), 0U);

    double lambda_r = (totalRowA * totalRowB > 0) ? targetNNZ / (totalRowA * totalRowB) : 0.0;
    double lambda_c = (totalColA * totalColB > 0) ? targetNNZ / (totalColA * totalColB) : 0.0;

    static std::random_device rd;
    static std::mt19937 gen(rd());

    propagateVector(*A.hr, *C.hr, 0.0, C.nnzRows, C.maxHr, C.rowsEq1, C.rowsGtHalf, C.n / 2, gen);
    for (size_t i = 0; i < C.m; ++i) {
        double val = (*A.hr)[i] + (*B.hr)[i] - (*A.hr)[i] * (*B.hr)[i] * lambda_c;
        (*C.hr)[i] = std::floor(val) + ((val - std::floor(val)) > std::uniform_real_distribution<>(0,1)(gen) ? 1 : 0);
    }

    propagateVector(*A.hc, *C.hc, 0.0, C.nnzCols, C.maxHc, C.colsEq1, C.colsGtHalf, C.m / 2, gen);
    for (size_t j = 0; j < C.n; ++j) {
        double val = (*A.hc)[j] + (*B.hc)[j] - (*A.hc)[j] * (*B.hc)[j] * lambda_r;
        (*C.hc)[j] = std::floor(val) + ((val - std::floor(val)) > std::uniform_real_distribution<>(0,1)(gen) ? 1 : 0);
    }

    return C;
}
/*Element-wise Multiplication propagation (see section 4.2 from "MNC: Structure-Exploiting Sparsity Estimation for
Matrix Expressions" paper)*/
// ---------- helpers ---------------------------------------------------------

static std::uint64_t sumU64(const std::vector<std::uint32_t> &v) {
    std::uint64_t s = 0;
    for (auto x : v) s += x;
    return s;
}
// splitmix64 hash (deterministic pseudo-random)
static std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Convert 64-bit hash to uniform double in [0,1)
static double u01_from_u64(std::uint64_t x) {
    // take top 53 bits -> double mantissa
    constexpr double inv2p53 = 1.0 / 9007199254740992.0; // 2^53
    return static_cast<double>(x >> 11) * inv2p53;
}

// Deterministic probabilistic rounding (order-independent)
static std::uint32_t probRoundClampDet(double x, std::uint64_t key,
                                              std::uint32_t hiInclusive) {
    if (!(x > 0.0)) return 0; // handles x<=0 and NaN
    const double f = std::floor(x);
    const double frac = x - f;

    std::uint64_t y = static_cast<std::uint64_t>(f);
    if (frac > 0.0) {
        const double u = u01_from_u64(splitmix64(key));
        if (u < frac) y += 1;
    }
    if (y > hiInclusive) y = hiInclusive;
    return static_cast<std::uint32_t>(y);
}

// ---------- element-wise multiplication propagation -------------------------

MncSketch propagateMul(const MncSketch &A, const MncSketch &B) {
    // element-wise requires same dims
    if (A.m != B.m || A.n != B.n)
        throw std::runtime_error("propagateMul: dimension mismatch");

    if (!A.hr || !A.hc || !B.hr || !B.hc)
        throw std::runtime_error("propagateMul: missing hr/hc vectors");

    const std::size_t m = A.m;
    const std::size_t n = A.n;

    if (A.hr->size() != m || B.hr->size() != m || A.hc->size() != n || B.hc->size() != n)
        throw std::runtime_error("propagateMul: hr/hc vector sizes inconsistent with m/n");

    const auto &hrA = *A.hr;
    const auto &hcA = *A.hc;
    const auto &hrB = *B.hr;
    const auto &hcB = *B.hc;

    const std::uint64_t nnzA = sumU64(hrA);
    const std::uint64_t nnzB = sumU64(hrB);

    MncSketch C;
    C.m = m;
    C.n = n;
    C.hr = std::make_shared<std::vector<std::uint32_t>>(m, 0);
    C.hc = std::make_shared<std::vector<std::uint32_t>>(n, 0);

    // As per paper: extension vectors only if exactly preserved -> not for ⊙
    C.her.reset();
    C.hec.reset();

    // conservative “full diagonal” propagation:
    // only guaranteed if both are full diagonal
    C.isDiagonal = (A.isDiagonal && B.isDiagonal);

    if (nnzA == 0 || nnzB == 0) {
        computeSummary(C);
        return C;
    }

    // λc from column collisions, λr from row collisions (symmetric form)
    long double colDot = 0.0L;
    for (std::size_t j = 0; j < n; ++j)
        colDot += static_cast<long double>(hcA[j]) * static_cast<long double>(hcB[j]);

    long double rowDot = 0.0L;
    for (std::size_t i = 0; i < m; ++i)
        rowDot += static_cast<long double>(hrA[i]) * static_cast<long double>(hrB[i]);

    const long double denom = static_cast<long double>(nnzA) * static_cast<long double>(nnzB);
    const double lambda_c = static_cast<double>(colDot / denom);
    const double lambda_r = static_cast<double>(rowDot / denom);

    // clamp maxima to uint32 range
    const std::uint32_t maxRowNnz =
        static_cast<std::uint32_t>(std::min<std::size_t>(n, std::numeric_limits<std::uint32_t>::max()));
    const std::uint32_t maxColNnz =
        static_cast<std::uint32_t>(std::min<std::size_t>(m, std::numeric_limits<std::uint32_t>::max()));

    // Eq. (15): ⊙ propagation with probabilistic rounding
    // hrC[i] = round( hrA[i] * hrB[i] * λc )
    for (std::size_t i = 0; i < m; ++i) {
        const double est = static_cast<double>(hrA[i]) * static_cast<double>(hrB[i]) * lambda_c;
        (*C.hr)[i] = probRoundClampDet(est, /*key*/0xC0FFEEULL ^ (std::uint64_t)i, maxRowNnz);
    }

    // hcC[j] = round( hcA[j] * hcB[j] * λr )
    for (std::size_t j = 0; j < n; ++j) {
        const double est = static_cast<double>(hcA[j]) * static_cast<double>(hcB[j]) * lambda_r;
        (*C.hc)[j] = probRoundClampDet(est, /*key*/0xBADC0DEULL ^ (std::uint64_t)j, maxColNnz);
    }

    computeSummary(C);
    return C;
}