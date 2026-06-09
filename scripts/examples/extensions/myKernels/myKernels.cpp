#include <runtime/local/datastructures/DenseMatrix.h>

#include <immintrin.h> // for the SIMD-enabled kernel
#include <iostream>
#include <stdexcept>

class DaphneContext;

extern "C" {
// Custom sequential sum-kernel.
void mySumSeq(float *res, const DenseMatrix<float> *arg, int kernelId, DaphneContext *ctx) {
    std::cerr << "hello from mySumSeq()" << std::endl;

    const float *valuesArg = arg->getValues();
    *res = 0;
    for (size_t r = 0; r < arg->getNumRows(); r++) {
        for (size_t c = 0; c < arg->getNumCols(); c++)
            *res += valuesArg[c];
        valuesArg += arg->getRowSkip();
    }
}

// Custom SIMD-enabled sum-kernel.
void mySumSIMD(float *res, const DenseMatrix<float> *arg, int kernelId, DaphneContext *ctx) {
    std::cerr << "hello from mySumSIMD()" << std::endl;

    // Validation.
    const size_t numCells = arg->getNumRows() * arg->getNumCols();
    if (numCells % 8)
        throw std::runtime_error("for simplicity, the number of cells must be "
                                 "a multiple of 8");
    if (arg->getNumCols() != arg->getRowSkip())
        throw std::runtime_error("for simplicity, the argument must not be "
                                 "a column segment of another matrix");

    // SIMD accumulation (8x f32).
    const float *valuesArg = arg->getValues();
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i < numCells / 8; i++) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(valuesArg));
        valuesArg += 8;
    }

    // Summation of accumulator elements.
    *res = (reinterpret_cast<float *>(&acc))[0] + (reinterpret_cast<float *>(&acc))[1] +
           (reinterpret_cast<float *>(&acc))[2] + (reinterpret_cast<float *>(&acc))[3] +
           (reinterpret_cast<float *>(&acc))[4] + (reinterpret_cast<float *>(&acc))[5] +
           (reinterpret_cast<float *>(&acc))[6] + (reinterpret_cast<float *>(&acc))[7];
}
}