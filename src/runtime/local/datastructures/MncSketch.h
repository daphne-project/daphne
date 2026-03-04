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
/*
This MNC Sketch implementatipn is based on the paper: Johanna Sommer, Matthias Boehm,
 Alexandre V. Evfimievski, Berthold Reinwald, Peter J. Haas (2019). 
 "MNC: Structure-Exploiting Sparsity Estimation for Matrix Expressions". 
SIGMOD '19: Proceedings of the 2019 International Conference on Management of Data
*/

#pragma once

#include <iostream>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <numeric>   
#include <random>
#include <algorithm> 
//Changes (Abdullah)
#include <memory>
#include <stdexcept>
#include <limits>

#include <utility>
#include <llvm/ADT/DenseMap.h>

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

/*
MNC Sketch data structure to capture the structure of a matrix to estimate sparsity
*/

struct MncSketch{
    // dimensions
    std::size_t m = 0; // rows
    std::size_t n = 0; // cols

    // core counts
    std::shared_ptr<std::vector<std::uint32_t>> hr;   // nnz per row (size m)
    std::shared_ptr<std::vector<std::uint32_t>> hc;   // nnz per col (size n)

    // Extended counts (optional, only constructed if maxHr or maxHc > 1)
    // her[i]: nnz in row i that lie in columns with hc == 1
    // hec[j]: nnz in column j that lie in rows with hr == 1
    std::shared_ptr<std::vector<std::uint32_t>> her;
    std::shared_ptr<std::vector<std::uint32_t>> hec;

    // Summary statistics
    std::uint32_t maxHr = 0;
    std::uint32_t maxHc = 0;
    std::uint32_t nnzRows = 0;       // # n rows with hr > 0
    std::uint32_t nnzCols = 0;       // # n cols with hc > 0
    std::uint32_t rowsEq1 = 0;       // # n rows with hr == 1, 
    std::uint32_t colsEq1 = 0;       // # n cols with hc == 1
    std::uint32_t rowsGtHalf = 0;    // # n rows with hr > n/2
    std::uint32_t colsGtHalf = 0;    // # n cols with hc > m/2
    bool isDiagonal;         // optional flag if A is (full) diagonal
};
namespace mlir::daphne {
    /*
    Sketch registry to store MNC sketches and retrieve them by their sketch id
    */
    using SketchId = int64_t;
    static constexpr SketchId UNKNOWN_SKETCH_ID = -1;

    class MncSketchRegistry {
        SketchId nextId = 0;
        llvm::DenseMap<SketchId, MncSketch> sketches;
    public:
        SketchId store(MncSketch s) {
            SketchId id = nextId++;
            sketches.try_emplace(id, std::move(s));
            return id;
        }
        const MncSketch* get(SketchId id) const {
            if (id == UNKNOWN_SKETCH_ID)
                return nullptr;
            auto it = sketches.find(id);
            return (it == sketches.end()) ? nullptr : &it->second;
        }   
        void clear() {
            sketches.clear();
            nextId = 0;
        }
    };
    void setActiveMncSketchRegistry(MncSketchRegistry *reg);
    void clearActiveMncSketchRegistry();
    MncSketchRegistry *getActiveMncSketchRegistry();
} // namespace mlir::daphne

inline size_t getMncSizeBytes(const MncSketch& h) {
    size_t size = sizeof(h);
    if (h.hr) size += h.hr->size() * 4;
    if (h.hc) size += h.hc->size() * 4;
    if (h.her) size += h.her->size() * 4;
    if (h.hec) size += h.hec->size() * 4;
    return size;
}