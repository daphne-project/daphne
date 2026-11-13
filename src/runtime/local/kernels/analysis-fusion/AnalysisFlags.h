#pragma once
#include "ir/daphneir/DataPropertyTypes.h"
#include "runtime/local/datastructures/CSRMatrix.h"
#include "runtime/local/datastructures/DenseMatrix.h"
#include "runtime/local/datastructures/Matrix.h"
#include "util/MurmurHash3.h"
#include "util/UniqueBoundedSet.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

// Some flags to extend our function arguments with
enum class AnalysisFlag { min, max, mean, sparsity, symmetry, numDistinct, numDistinctApprox };

// Template to accept a parameter pack of possible analysis enums.
template <AnalysisFlag... Fs> struct AnalysisFlags {
    // variable template to check if an AnalysisFlag is present.
    template <AnalysisFlag F> constexpr static bool contains = ((F == Fs) || ...);
    template <AnalysisFlag... F> constexpr static bool containsAll = (contains<F> && ...);
    template <AnalysisFlag... F> constexpr static bool containsAny = (contains<F> || ...);
};

// The access pattern that is used to construct the output matrix
// some DenseMatrix <- CSRMatrix operations do not access every index of the output matrix.
// Matrix is considered as Dense.
enum class AccessPattern { Dense, Sparse };

// The type of the output matrix. This must match or it will perform UB.
enum class OutputMatrix { Dense, Sparse, Matrix };

// Use 1024 for k minimum values.
constexpr size_t KMV_K = 1024;

template <typename VT, AccessPattern AP, OutputMatrix OM, AnalysisFlag... Fs> struct AnalAcc {
    // compute the number of elements once
    size_t numElem;

    using flags = AnalysisFlags<Fs...>;

    static_assert(!(flags::template contains<AnalysisFlag::numDistinct> &&
                    flags::template contains<AnalysisFlag::numDistinctApprox>),
                  "Cannot use both numDistinct and numDistinctApprox simultaneously");

    // no_unique_address ensures 0 byte is used if this field is not used (std::monostate)
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::mean>, double, std::monostate>
        sumAcc{};

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::min>, VT, std::monostate> minAcc{};

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::max>, VT, std::monostate> maxAcc{};

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::sparsity>, size_t, std::monostate>
        numNZ{};

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::numDistinct>,
                                             std::unordered_set<VT>, std::monostate> distinct{};

    // K Value
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::numDistinctApprox>, int64_t,
                                             std::monostate> seed;
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::numDistinctApprox>,
                                             UniqueBoundedSet<uint32_t>, std::monostate> uBSet;

    // Dense: queue of values only
    using DenseSymVec = std::vector<std::queue<VT>>;
    // Sparse: queue of (col_index, value) pairs
    using SparseSymVec = std::vector<std::queue<std::pair<size_t, VT>>>;

    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::symmetry>,
                                             std::conditional_t<AP == AccessPattern::Sparse, SparseSymVec, DenseSymVec>,
                                             std::monostate> symVec{};
    [[no_unique_address]] std::conditional_t<flags::template contains<AnalysisFlag::symmetry>, bool, std::monostate>
        isSymmetric;

    AnalAcc()
        requires(flags::template contains<AnalysisFlag::numDistinctApprox>)
        : uBSet(KMV_K) {}

    AnalAcc()
        requires(!flags::template contains<AnalysisFlag::numDistinctApprox>)
    = default;

    // accumulate order-independent statistics
    // implicit assumption: this function called once with 0 is behaviorally equivalent to calling with 0 multiple
    // times. This could be problematic if other measures are considered.
    [[gnu::always_inline]] inline void accumulateScalar(VT val) {
        // mean is computed by first summing up all values. It is possible to overflow.
        if constexpr (flags::template contains<AnalysisFlag::mean>)
            sumAcc += val;
        if constexpr (flags::template contains<AnalysisFlag::min>)
            minAcc = std::min(minAcc, val);
        if constexpr (flags::template contains<AnalysisFlag::max>)
            maxAcc = std::max(maxAcc, val);
        if constexpr (flags::template contains<AnalysisFlag::sparsity>)
            numNZ += (val != 0);
        if constexpr (flags::template contains<AnalysisFlag::numDistinct>)
            distinct.insert(val);
        // Same as the NumDistinctApprox kernel, use K Minimum Values with MurmurHash.
        if constexpr (flags::template contains<AnalysisFlag::numDistinctApprox>) {
            uint32_t hashedValueOut = 0;
            MurmurHash3_x86_32(&val, sizeof(VT), seed, &hashedValueOut);
            uBSet.push(hashedValueOut);
        }
    }
    [[gnu::always_inline]] inline void accumulateSym(size_t r, size_t c, VT val) {
        /*
            For the upper left half, push the values into queues
            For the lower right half, pop the queues to check if a corresponding value was seen.
        */
        if constexpr (flags::template contains<AnalysisFlag::symmetry>) {
            // all values are pushed in order
            if constexpr (AP == AccessPattern::Dense)
                if (isSymmetric) {
                    if (r < c) {
                        symVec[c].push(val);
                    }
                    if (r > c) {
                        isSymmetric = symVec[r].front() == val;
                        symVec[r].pop();
                    }
                }

            // we keep track of the position because zeros are skipped.
            if constexpr (AP == AccessPattern::Sparse) {
                if (val == static_cast<VT>(0))
                    return; // Ignore resulting zeros like the other zeroes that were (implicitly) skipped.
                if (isSymmetric) {
                    if (r < c) {
                        symVec[c].push({r, val});
                    }
                    if (r > c) {
                        if (symVec[r].empty()) {
                            isSymmetric = false;
                        } else {
                            auto [ci, v] = symVec[r].front();
                            if (ci == c) {
                                // a value at a symmetric position was found
                                isSymmetric = v == val;
                                symVec[r].pop();
                            } else {
                                // an entry at a symmetric position was not pushed
                                isSymmetric = false;
                            }
                        }
                    }
                }
            }
        }
    }
    // A convenience function to accumulate both kinds of
    [[gnu::always_inline]] inline void accumulate(size_t r, size_t c, VT val) {
        accumulateScalar(val);
        accumulateSym(r, c, val);
    }

    [[gnu::always_inline]] inline void init(size_t numRows, size_t numCols) {
        // There is no statistics to analyze in an empty matrix
        numElem = numRows * numCols;
        if (numElem == 0) {
            return;
        };
        if constexpr (flags::template contains<AnalysisFlag::min>)
            minAcc = std::numeric_limits<VT>::max();
        if constexpr (flags::template contains<AnalysisFlag::max>)
            maxAcc = std::numeric_limits<VT>::lowest();
        if constexpr (flags::template contains<AnalysisFlag::symmetry>) {
            bool isSquare = numRows == numCols;
            isSymmetric = isSquare;
            symVec.resize(isSquare ? numRows : 0);
        }

        if constexpr (flags::template contains<AnalysisFlag::numDistinctApprox>) {
            seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            uBSet = UniqueBoundedSet<uint32_t>(KMV_K);
        }
    }

    [[gnu::always_inline]] inline void writeResult(DenseMatrix<VT> *res) {
        writeResult(static_cast<Matrix<VT> *>(res));
    }
    [[gnu::always_inline]] inline void writeResult(CSRMatrix<VT> *res) { writeResult(static_cast<Matrix<VT> *>(res)); }

    // Write the result back to the metadata of the resulting matrix
    [[gnu::always_inline]] inline void writeResult(Matrix<VT> *res) {
        // There is no statistics to analyze in an empty matrix
        if (numElem == 0)
            return;

        if constexpr (AP == AccessPattern::Sparse) {
            if constexpr (OM == OutputMatrix::Sparse) {
                if constexpr (flags::template contains<AnalysisFlag::sparsity>) {
                    if (numNZ < numElem)
                        accumulateScalar(static_cast<VT>(0));
                } else {
                    // Fallback to computing from the CSR method if the result is CSRMatrix
                    const auto *csr = static_cast<CSRMatrix<VT> *>(res);
                    if (csr->getNumNonZeros() < numElem)
                        accumulateScalar(static_cast<VT>(0));
                }
            } else if constexpr (OM == OutputMatrix::Dense) {
                // DenseMatrix <- CSRMatrix
                if constexpr (flags::template contains<AnalysisFlag::sparsity>) {
                    if (numNZ < numElem)
                        accumulateScalar(static_cast<VT>(0));
                } else {
                    // Fallback to computing by checking every index if the result is DenseMatrix
                    const auto *denseRes = static_cast<DenseMatrix<VT> *>(res);
                    const size_t numRows = denseRes->getNumRows();
                    const size_t numCols = denseRes->getNumCols();
                    const VT *valuesRes = denseRes->getValues();

                    bool hasZero = false;
                    for (size_t r = 0; r < numRows; r++) {
                        for (size_t c = 0; c < numCols; c++) {
                            if (valuesRes[c] == static_cast<VT>(0)) {
                                hasZero = true;
                                break;
                            }
                        }
                        if (hasZero)
                            break;
                        valuesRes += denseRes->getRowSkip();
                    }
                    if (hasZero)
                        accumulateScalar(static_cast<VT>(0));
                }
            } else if constexpr (OM == OutputMatrix::Matrix) {
                if constexpr (flags::template contains<AnalysisFlag::sparsity>) {
                    if (numNZ < numElem)
                        accumulateScalar(static_cast<VT>(0));
                } else {
                    // Fallback to computing by checking every index if the result is Matrix
                    const size_t numRows = res->getNumRows();
                    const size_t numCols = res->getNumCols();
                    bool hasZero = false;
                    for (size_t r = 0; r < numRows; r++) {
                        for (size_t c = 0; c < numCols; c++) {
                            if (res->get(r, c) == static_cast<VT>(0)) {
                                hasZero = true;
                                break;
                            }
                        }
                        if (hasZero)
                            break;
                    }
                    if (hasZero)
                        accumulateScalar(static_cast<VT>(0));
                }
            } else { // There is no other case.
                __builtin_unreachable();
            }
        }
        if constexpr (flags::template contains<AnalysisFlag::sparsity>)
            res->sparsity = numNZ / (double)(numElem);

        if constexpr (flags::template contains<AnalysisFlag::mean>)
            res->mean = sumAcc / (double)(numElem);

        if constexpr (flags::template contains<AnalysisFlag::min>)
            res->min = minAcc;

        if constexpr (flags::template contains<AnalysisFlag::max>)
            res->max = maxAcc;

        if constexpr (flags::template contains<AnalysisFlag::numDistinct>)
            res->numDistinct = distinct.size();

        if constexpr (flags::template contains<AnalysisFlag::numDistinctApprox>) {
            if (uBSet.size() < KMV_K) {
                res->numDistinct = uBSet.size();
            } else {
                size_t kMinVal = uBSet.top();
                const size_t maxVal = std::numeric_limits<uint32_t>::max();
                double kMinValNormed = static_cast<double>(kMinVal) / static_cast<double>(maxVal);
                res->numDistinct = static_cast<size_t>(static_cast<double>(KMV_K - 1) / kMinValNormed);
            }
        }

        if constexpr (flags::template contains<AnalysisFlag::symmetry>) {
            if constexpr (AP == AccessPattern::Dense)
                res->symmetric = isSymmetric ? BoolOrUnknown::True : BoolOrUnknown::False;
            if constexpr (AP == AccessPattern::Sparse) {
                if (isSymmetric) {
                    for (auto &q : symVec) {
                        if (!q.empty()) {
                            isSymmetric = false;
                            break;
                        }
                    }
                    res->symmetric = isSymmetric ? BoolOrUnknown::True : BoolOrUnknown::False;
                } else {
                    res->symmetric = BoolOrUnknown::False;
                }
            }
        }
    }
};
