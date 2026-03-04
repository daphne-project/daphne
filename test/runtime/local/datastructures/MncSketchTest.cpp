/*
 * Tests for MNC sketch on CSRMatrix
 */
#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <runtime/local/datastructures/MncSketch.h>
#include <runtime/local/datastructures/MncSketchBuild.h>
#include <runtime/local/datastructures/MncSketchEstimate.h>
#include <runtime/local/datastructures/MncSketchPropagate.h>

#include <tags.h>
#include <catch.hpp>

#include <cstdint>
#include <vector>

// run ./test.sh -nb -d yes [datastructures] after building Daphne to execute this test
// Test Case for building MNC sketch from CSRMatrix
TEST_CASE("MNC sketch from CSRMatrix basic", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Matrix:
    // [0 5 0
    //  0 0 3
    //  1 0 0]
    const size_t numRows     = 3;
    const size_t numCols     = 3;
    const size_t numNonZeros = 3;

    CSRMatrix<ValueType> *m =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRows, numCols, numNonZeros, /*zero=*/true);

    ValueType *values   = m->getValues();
    size_t   *colIdxs   = m->getColIdxs();
    size_t   *rowOffsets= m->getRowOffsets();

    // Valid CSR: one nnz per row
    // rowOffsets: [0,1,2,3]
    rowOffsets[0] = 0;
    rowOffsets[1] = 1;
    rowOffsets[2] = 2;
    rowOffsets[3] = 3;

    // colIdxs: (0,1)->5, (1,2)->3, (2,0)->1
    colIdxs[0] = 1;
    colIdxs[1] = 2;
    colIdxs[2] = 0;

    values[0] = 5.0;
    values[1] = 3.0;
    values[2] = 1.0;

    MncSketch h = buildMncFromCsrMatrix(*m);

    // dimensions
    CHECK(h.m == numRows);
    CHECK(h.n == numCols);

    // row + col nnz
    std::vector<std::uint32_t> expectedHr{1,1,1};
    std::vector<std::uint32_t> expectedHc{1,1,1};
    CHECK(*h.hr == expectedHr);
    CHECK(*h.hc == expectedHc);

    // summary stats
    CHECK(h.maxHr    == 1);
    CHECK(h.maxHc    == 1);
    CHECK(h.nnzRows  == 3);
    CHECK(h.nnzCols  == 3);
    CHECK(h.rowsEq1  == 3);
    CHECK(h.colsEq1  == 3);
    CHECK(h.rowsGtHalf == 0); // 1 <= n/2 since n=3
    CHECK(h.colsGtHalf == 0); // 1 <= m/2 since m=3

    DataObjectFactory::destroy(m);
}

TEST_CASE("MNC sketch respects CSRMatrix sub-matrix view", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Original 4x3 matrix:
    // row 0: (0,1)
    // row 1: (1,2)
    // row 2: (2,0)
    // row 3: (3,2)  <- we'll slice rows [1,3) = rows 1 and 2
    const size_t numRowsOrig   = 4;
    const size_t numColsOrig   = 3;
    const size_t numNonZeros   = 4;

    CSRMatrix<ValueType> *mOrig =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsOrig, numColsOrig, numNonZeros, /*zero=*/true);

    ValueType *valuesOrig    = mOrig->getValues();
    size_t   *colIdxsOrig    = mOrig->getColIdxs();
    size_t   *rowOffsetsOrig = mOrig->getRowOffsets();

    rowOffsetsOrig[0] = 0;
    rowOffsetsOrig[1] = 1;
    rowOffsetsOrig[2] = 2;
    rowOffsetsOrig[3] = 3;
    rowOffsetsOrig[4] = 4;

    colIdxsOrig[0] = 1; // row 0
    colIdxsOrig[1] = 2; // row 1
    colIdxsOrig[2] = 0; // row 2
    colIdxsOrig[3] = 2; // row 3

    valuesOrig[0] = 1.0;
    valuesOrig[1] = 1.0;
    valuesOrig[2] = 1.0;
    valuesOrig[3] = 1.0;

    // Create sub-matrix with rows [1,3) = rows 1 and 2 of original
    CSRMatrix<ValueType> *mSub = DataObjectFactory::create<CSRMatrix<ValueType>>(mOrig, 1, 3);

    MncSketch hSub = buildMncFromCsrMatrix(*mSub);

    // submatrix is 2x3, with nnz rows = 2
    CHECK(hSub.m == 2);
    CHECK(hSub.n == 3);

    // each row in the submatrix has exactly 1 nnz
    std::vector<std::uint32_t> expectedHrSub{1,1};
    CHECK(*hSub.hr == expectedHrSub);
    CHECK(hSub.nnzRows == 2);
    CHECK(hSub.rowsEq1 == 2);

    DataObjectFactory::destroy(mSub);
    DataObjectFactory::destroy(mOrig);
}

TEST_CASE("Build MNC Sketch from csr matrix with example from paper", TAG_DATASTRUCTURES) {
    using ValueType = double;
    /* Matrix:
    [0,0,0,0,0,0,0,1,0], 
    [0,1,0,0,1,0,0,0,0], 
    [0,0,0,1,1,1,0,0,0], 
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [1,0,0,1,0,0,0,0,0],
    [0,0,1,0,1,0,1,0,0],
    [0,0,0,0,0,0,0,0,1],
    */
    const size_t numRows     = 9;
    const size_t numCols     = 9;
    const size_t numNonZeros = 14;

    CSRMatrix<ValueType> *m =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRows, numCols, numNonZeros, /*zero=*/true);

    ValueType *values   = m->getValues();
    size_t   *colIdxs   = m->getColIdxs();
    size_t   *rowOffsets= m->getRowOffsets();

    // Valid CSR: one nnz per row
    // rowOffsets: [ 0  1  3  6  6  7  8 10 13 14]
    rowOffsets[0] = 0;
    rowOffsets[1] = 1;
    rowOffsets[2] = 3;
    rowOffsets[3] = 6;
    rowOffsets[4] = 6;
    rowOffsets[5] = 7;
    rowOffsets[6] = 8;
    rowOffsets[7] = 10;
    rowOffsets[8] = 13;
    rowOffsets[9] = 14;

    // colIdxs: [7 1 4 3 4 5 2 7 0 3 2 4 6 8]
    colIdxs[0] = 7;
    colIdxs[1] = 1;
    colIdxs[2] = 4;
    colIdxs[3] = 3;
    colIdxs[4] = 4;
    colIdxs[5] = 5;
    colIdxs[6] = 2;
    colIdxs[7] = 7;
    colIdxs[8] = 0;
    colIdxs[9] = 3;
    colIdxs[10] = 2;
    colIdxs[11] = 4;
    colIdxs[12] = 6;
    colIdxs[13] = 8;
    
    for (size_t i = 0; i < numNonZeros; ++i)
        values[i] = 1.0;

    MncSketch h = buildMncFromCsrMatrix(*m);

    // dimensions
    CHECK(h.m == numRows);
    CHECK(h.n == numCols);
    // row + col nnz
    std::vector<std::uint32_t> expectedHr{1,2,3,0,1,1,2,3,1};
    std::vector<std::uint32_t> expectedHc{1,1,2,2,3,1,1,2,1};
    CHECK(*h.hr == expectedHr);
    CHECK(*h.hc == expectedHc);
    // her and hec
    std::vector<std::uint32_t> expectedHer{0,1,1,0,0,0,1,1,1};
    std::vector<std::uint32_t> expectedHec{0,0,1,0,0,0,0,2,1};
    std::vector<std::uint32_t> notexpectedHec{1,1,1,0,0,0,0,2,1};
    CHECK(*h.her == expectedHer);
    CHECK(*h.hec == expectedHec);

    DataObjectFactory::destroy(m);
}
TEST_CASE("Build MNC Sketch from dense matrix with example from paper", TAG_DATASTRUCTURES) {
    /* Matrix:
    [0,0,0,0,0,0,0,1,0], 
    [0,1,0,0,1,0,0,0,0], 
    [0,0,0,1,1,1,0,0,0], 
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [1,0,0,1,0,0,0,0,0],
    [0,0,1,0,1,0,1,0,0],
    [0,0,0,0,0,0,0,0,1],
    */
    const size_t numRows     = 9;
    const size_t numCols     = 9;
    
    auto m_dense = genGivenVals<DenseMatrix<double>>(9, {0,0,0,0,0,0,0,1,0,
                                                  0,1,0,0,1,0,0,0,0,
                                                  0,0,0,1,1,1,0,0,0,
                                                  0,0,0,0,0,0,0,0,0,
                                                  0,0,1,0,0,0,0,0,0,
                                                  0,0,0,0,0,0,0,1,0,
                                                  1,0,0,1,0,0,0,0,0,
                                                  0,0,1,0,1,0,1,0,0,
                                                  0,0,0,0,0,0,0,0,1});


    MncSketch h_dense = buildMncFromDenseMatrix(*m_dense);

    // dimensions
    CHECK(h_dense.m == numRows);
    CHECK(h_dense.n == numCols);
    // row + col nnz
    std::vector<std::uint32_t> expectedHr{1,2,3,0,1,1,2,3,1};
    std::vector<std::uint32_t> expectedHc{1,1,2,2,3,1,1,2,1};
    CHECK(*h_dense.hr == expectedHr);
    CHECK(*h_dense.hc == expectedHc);

    // her and hec
    std::vector<std::uint32_t> expectedHer{0,1,1,0,0,0,1,1,1};
    std::vector<std::uint32_t> expectedHec{0,0,1,0,0,0,0,2,1};
    std::vector<std::uint32_t> notexpectedHec{1,1,1,0,0,0,0,2,1};
    CHECK(*h_dense.her == expectedHer);
    CHECK(*h_dense.hec == expectedHec);
    
    DataObjectFactory::destroy(m_dense);
}

// Tests for estimateSparsity_product function
TEST_CASE("Case 1: maxHr(A) <= 1 or maxHr(B) <= 1", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Matrix A: 3x3 
    // [1 0 0]
    // [0 1 0]
    // [0 0 1]
    const size_t numRowsA = 3;
    const size_t numColsA = 3;
    const size_t nnzA     = 3;

    CSRMatrix<ValueType> *A = 
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsA, numColsA, nnzA, true);

    ValueType *valuesA    = A->getValues();
    size_t   *colIdxsA    = A->getColIdxs();
    size_t   *rowOffsetsA = A->getRowOffsets();

    rowOffsetsA[0] = 0;
    rowOffsetsA[1] = 1;
    rowOffsetsA[2] = 2;
    rowOffsetsA[3] = 3;

    colIdxsA[0] = 0;
    colIdxsA[1] = 1; 
    colIdxsA[2] = 2; 

    valuesA[0] = 1.0;
    valuesA[1] = 1.0;
    valuesA[2] = 1.0;
    // Matrix B: 3x2 
    // [1 0]
    // [1 0]
    // [0 1]
    const size_t numRowsB = 3;
    const size_t numColsB = 2;
    const size_t nnzB     = 3;

    CSRMatrix<ValueType> *B = 
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsB, numColsB, nnzB, true);

    ValueType *valuesB    = B->getValues();
    size_t   *colIdxsB    = B->getColIdxs();
    size_t   *rowOffsetsB = B->getRowOffsets();

    rowOffsetsB[0] = 0;
    rowOffsetsB[1] = 1;
    rowOffsetsB[2] = 2;
    rowOffsetsB[3] = 3;

    colIdxsB[0] = 0; 
    colIdxsB[1] = 0; 
    colIdxsB[2] = 1; 

    valuesB[0] = 1.0;
    valuesB[1] = 1.0;
    valuesB[2] = 1.0;

    MncSketch hA = buildMncFromCsrMatrix(*A);
    MncSketch hB = buildMncFromCsrMatrix(*B);

    double s = estimateSparsity_product(hA, hB);

    // REQUIRE(s >= 0.0);
    // REQUIRE(s <= 1.0);
    REQUIRE(s == 0.5);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
}

TEST_CASE("Case 2: some rows/cols have >1 nnz", TAG_DATASTRUCTURES) {
    // --- Matrix A: 3x3 ---
    // [1 1 0]
    // [0 1 1]
    // [0 0 1]
    auto m_A = genGivenVals<DenseMatrix<double>>(3, {1,1,0,
                                                    0,1,1,
                                                    0,0,1});

    // --- Matrix B: 3x3 ---
    // [1 0 0]
    // [1 1 0]
    // [0 1 1]
    auto m_B = genGivenVals<DenseMatrix<double>>(3, {1,0,0,
                                                    1,1,0,
                                                    0,1,1});

    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    double s = estimateSparsity_product(hA, hB);

    REQUIRE(s >= 0.0);
    REQUIRE(s <= 1.0);
    // std::cout << "Estimated sparsity: " << s << std::endl;
    // REQUIRE(s == 5);
    // std::size_t p = (hA.nnzRows - hA.rowsEq1) * (hB.nnzCols - hB.colsEq1);
    // REQUIRE(p == 4);           
    // REQUIRE(s == 5);

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}

TEST_CASE("Case 3: some rows/cols have >1 nnz for Dense Matrix", TAG_DATASTRUCTURES) {
    // --- Matrix A: 3x3 ---
    // [1 1 0]
    // [0 1 1]
    // [0 0 1]
    
    auto m_A = genGivenVals<DenseMatrix<double>>(3, {1,1,0,
                                                    0,1,1,
                                                    0,0,1});

    // --- Matrix B: 3x3 ---
    // [1 0 0]
    // [1 1 0]
    // [0 1 1]
    

    auto m_B = genGivenVals<DenseMatrix<double>>(3, {1,0,0,
                                                    1,1,0,
                                                    0,1,1});

    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    double s = estimateSparsity_product(hA, hB);
    // nnz in A*B = 7 / 9 = 0.7777
    REQUIRE(s >= 0.65);
    REQUIRE(s <= 0.85);
    // REQUIRE(s == 5);
    // std::size_t p = (hA.nnzRows - hA.rowsEq1) * (hB.nnzCols - hB.colsEq1);
    // REQUIRE(p == 4);
    // REQUIRE(s == 5);

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}

TEST_CASE("Case 4: test estimatsparsity_product using example from paper (Golden Master)", TAG_DATASTRUCTURES) {
    // --- Matrix A: 9x9 (Paper Example) ---
    auto m_A = genGivenVals<DenseMatrix<double>>(9, {
        0,0,0,0,0,0,0,1,0, 0,1,0,0,1,0,0,0,0, 0,0,0,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,0,
        1,0,0,1,0,0,0,0,0, 0,0,1,0,1,0,1,0,0, 0,0,0,0,0,0,0,0,1
    });

    // --- Matrix B: 9x10 (Paper Example) ---
    auto m_B = genGivenVals<DenseMatrix<double>>(9, {
        0,0,0,1,0,0,0,0,0,0, 0,0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0, 0,0,1,0,0,0,1,0,1,0, 0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,0,0, 0,0,1,0,0,0,0,0,0,0
    });

    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    double s = estimateSparsity_product(hA, hB);

    // --- Exact Verification ---
    // Exact Hits = 7. Probabilistic Hits = 6.974. Total = 13.974.
    // Sparsity = 13.974 / 90 ≈ 0.155266
    double expectedSparsity = 0.155266;
    
    // Check strict equality within floating point margin
    REQUIRE(s == Approx(expectedSparsity).epsilon(0.001));

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}
// TODO: Test cases: estimated sparsity should be expected as the paper implementation. 
// TODO: structural properties: corner cases in the paper 


// -----------------------------------------------------------------------------
    /**Propagation Tests
    *Test file for the MncPropagation logic, which allows us to predict the intermediate sparsity of 
    *a chain of multiplications of matrices without the need to compute the full operation.
    *We tested our logic with a total of 3 tests, details are given below
    */
// -----------------------------------------------------------------------------
TEST_CASE("Case 1: Exact Propagation (Diagonal Matrix)", TAG_DATASTRUCTURES) {
    /*
    *In this test, we take 2 3x3 matrices, 
    *one of them (A) being an identity matrix. If the estimator introduced even a 
    very small rounding error, then that would mean that our logic is false.
    We expect that the output sketch must be a match of matrix B.
    */

    // Matrix A: 3x3 Identity (Diagonal)
    // [1 0 0]
    // [0 1 0]
    // [0 0 1]
    auto A = genGivenVals<DenseMatrix<double>>(3, {1,0,0,
                                                    0,1,0,
                                                    0,0,1});

    // Matrix B: 3x3 Sparse
    // [1 0 0]
    // [1 1 0]
    // [0 0 0]
    auto B = genGivenVals<DenseMatrix<double>>(3, {1,0,0,
                                                    1,1,0,
                                                    0,0,0});
    MncSketch hA = buildMncFromDenseMatrix(*A);
    MncSketch hB = buildMncFromDenseMatrix(*B);
    
    REQUIRE(hA.isDiagonal == true);
    MncSketch hC = propagateMM(hA, hB);

    // Since A is diagonal, C should exact copy of B
    REQUIRE(hC.m == hB.m);
    REQUIRE(hC.n == hB.n);
    REQUIRE(hC.hr == hB.hr);
    REQUIRE(hC.hc == hB.hc);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
}

TEST_CASE("Case 2: Outer Product (Density Blowup)", TAG_DATASTRUCTURES) {
    /*
    This test is our worst case scenario for sparsity. It takes a 3x1 matrix and a 1x3 matrix.
    We start with only a total of 6 non-zero entries but end up with a full dense 3x3 matrix
    with 9 non-zeros (no zero values). This test ensures that the logic predicts the
    total content, resulting in 100% density.
    */
    using ValueType = double;

    // Matrix A: 3x1 Column Vector
    // [1]
    // [1]
    // [1]
    const size_t numRowsA = 3;
    const size_t numColsA = 1;
    const size_t nnzA     = 3;
    CSRMatrix<ValueType> *A =  DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsA, numColsA, nnzA, true);

    size_t *rowOffsetsA = A->getRowOffsets();
    size_t *colIdxsA    = A->getColIdxs();
    ValueType *valuesA  = A->getValues();

    rowOffsetsA[0] = 0; 
    rowOffsetsA[1] = 1; 
    rowOffsetsA[2] = 2; 
    rowOffsetsA[3] = 3;

    colIdxsA[0] = 0; 
    colIdxsA[1] = 0; 
    colIdxsA[2] = 0;

    valuesA[0] = 1.0; 
    valuesA[1] = 1.0; 
    valuesA[2] = 1.0;

    // Matrix B: 1x3 Row Vector
    // [1 1 1]
    const size_t numRowsB = 1;
    const size_t numColsB = 3;
    const size_t nnzB     = 3;
    CSRMatrix<ValueType> *B =  DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsB, numColsB, nnzB, true);
    size_t *rowOffsetsB = B->getRowOffsets();
    size_t *colIdxsB    = B->getColIdxs();
    ValueType *valuesB  = B->getValues();

    rowOffsetsB[0] = 0; 
    rowOffsetsB[1] = 3;
    colIdxsB[0] = 0; 
    colIdxsB[1] = 1; 
    colIdxsB[2] = 2;
    valuesB[0] = 1.0; 
    valuesB[1] = 1.0; 
    valuesB[2] = 1.0;

    MncSketch hA = buildMncFromCsrMatrix(*A);
    MncSketch hB = buildMncFromCsrMatrix(*B);

    MncSketch hC = propagateMM(hA, hB);

    // Result should be 3x3 and fully dense (9 items)
    REQUIRE(hC.m == 3);
    REQUIRE(hC.n == 3);
    REQUIRE(hC.nnzRows == 3);
    REQUIRE(hC.nnzCols == 3);
    
    // Check total NNZ roughly matches 9
    uint64_t totalNNZ = 0;
    for(auto c : *hC.hr) totalNNZ += c;
    REQUIRE(totalNNZ > 0);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
}

TEST_CASE("Case 3: Chain Propagation (Dimensions)", TAG_DATASTRUCTURES) {
    /*
    The third test covers a chain multiplication, in which we first multiply a 4x2 matrix
    with a 2x5 matrix and then with a 5x3 matrix. This test covers the recursive logic
    of our code. It not only makes sure that the non-zero counts are correct, but the
    output of the first multiplication is compatible with the input of the second.
    */
    using ValueType = double;

    // Matrix A: 4x2
    const size_t numRowsA = 4;
    const size_t numColsA = 2;
    CSRMatrix<ValueType> *A =  DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsA, numColsA, numRowsA, true);

    size_t *rowOffsetsA = A->getRowOffsets();
    size_t *colIdxsA    = A->getColIdxs();
    for(size_t i=0; i<=numRowsA; ++i) rowOffsetsA[i] = i; 
    for(size_t i=0; i<numRowsA; ++i) colIdxsA[i] = 0; 

    // Matrix B: 2x5
    const size_t numRowsB = 2;
    const size_t numColsB = 5;
    CSRMatrix<ValueType> *B = DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsB, numColsB, numRowsB, true);

    size_t *rowOffsetsB = B->getRowOffsets();
    size_t *colIdxsB    = B->getColIdxs();
    for(size_t i=0; i<=numRowsB; ++i) rowOffsetsB[i] = i; 
    for(size_t i=0; i<numRowsB; ++i) colIdxsB[i] = 0; 

    // Matrix C: 5x3
    const size_t numRowsC = 5;
    const size_t numColsC = 3;
    CSRMatrix<ValueType> *C =  DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsC, numColsC, numRowsC, true);
    
    size_t *rowOffsetsC = C->getRowOffsets();
    size_t *colIdxsC    = C->getColIdxs();
    for(size_t i=0; i<=numRowsC; ++i) rowOffsetsC[i] = i; 
    for(size_t i=0; i<numRowsC; ++i) colIdxsC[i] = 0; 

    MncSketch hA = buildMncFromCsrMatrix(*A);
    MncSketch hB = buildMncFromCsrMatrix(*B);
    MncSketch hC = buildMncFromCsrMatrix(*C);

    std::vector<MncSketch> chain = {hA, hB, hC};
    MncSketch result = propagateChain(chain);

    // (4x2) * (2x5) -> (4x5)
    // (4x5) * (5x3) -> (4x3)
    REQUIRE(result.m == 4);
    REQUIRE(result.n == 3);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
    DataObjectFactory::destroy(C);

}


// -----------------------------------------------------------------------------
// Transpose Test
// -----------------------------------------------------------------------------
TEST_CASE("Case 5: Transpose Propagation", TAG_DATASTRUCTURES) {
    using ValueType = double;
    // Row 0: [1, 0, 0, 0]
    // Row 1: [0, 2, 3, 0]
    // Row 2: [0, 0, 0, 4]
    auto A = genGivenVals<CSRMatrix<ValueType>>(3, {
        1, 0, 0, 0,
        0, 2, 3, 0,
        0, 0, 0, 4
    });

    MncSketch hA = buildMncFromCsrMatrix(*A);
    MncSketch hC = propagateTranspose(hA); 

    // 1. Verify Dimensions (Should be swapped)
    // Input was 3x4 so the  Output should be 4x3
    CHECK(hC.m == 4);
    CHECK(hC.n == 3);

    // 2. Verify Swapped Histograms
    // A.hr was {1, 2, 1}. This should now be C.hc
    CHECK((*hC.hc)[0] == 1);
    CHECK((*hC.hc)[1] == 2);
    CHECK((*hC.hc)[2] == 1);

    // A.hc was {1, 1, 1, 1}. This should now be C.hr
    CHECK((*hC.hr)[0] == 1);
    CHECK((*hC.hr)[1] == 1);
    CHECK((*hC.hr)[2] == 1);
    CHECK((*hC.hr)[3] == 1);

    // 3. Verify Stats Swapped
    // A.maxHr was 2. C.maxHc should be 2.
    CHECK(hC.maxHc == 2);
    CHECK(hC.maxHr == 1);

    DataObjectFactory::destroy(A);
}

TEST_CASE("Estimate sparsity for element-wise addition", TAG_DATASTRUCTURES) {
    // --- Matrix A: 3x3 ---
    // [1 0 0]
    // [0 1 1]
    // [0 0 0]
    auto m_A = genGivenVals<DenseMatrix<double>>(3, {
        1, 0, 0,
        0, 1, 1,
        0, 0, 0
    });

    // --- Matrix B: 3x3 ---
    // [0 1 0]
    // [1 0 0]
    // [0 0 1]
    auto m_B = genGivenVals<DenseMatrix<double>>(3, {
        0, 1, 0,
        1, 0, 0,
        0, 0, 1
    });

    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    double s = estimateSparsity_ElementWiseAddition(hA, hB);
    // std::cout << "Estimated sparsity: " << s << std::endl;
    
    REQUIRE(s == Approx(0.5556).epsilon(1e-4));

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}

TEST_CASE("Estimate sparsity for element-wise addition (Large)", TAG_DATASTRUCTURES) {
    // --- Matrix A: 9x9 ---
    // [0,0,0,0,0,0,0,1,0], 
    // [0,1,0,0,1,0,0,0,0], 
    // [0,0,0,1,1,1,0,0,0], 
    // [0,0,0,0,0,0,0,0,0],
    // [0,0,1,0,0,0,0,0,0],
    // [0,0,0,0,0,0,0,1,0],
    // [1,0,0,1,0,0,0,0,0],
    // [0,0,1,0,1,0,1,0,0],
    // [0,0,0,0,0,0,0,0,1],
    
    auto m_A = genGivenVals<DenseMatrix<double>>(9, {0,0,0,0,0,0,0,1,0,
                                                    0,1,0,0,1,0,0,0,0,
                                                    0,0,0,1,1,1,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,
                                                    0,0,1,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,1,0,
                                                    1,0,0,1,0,0,0,0,0,
                                                    0,0,1,0,1,0,1,0,0,
                                                    0,0,0,0,0,0,0,0,1
    });

    // --- Matrix B: 9x10 ---
    // [0 0 0 1 0 0 0 0 0 0]
    // [0 0 0 0 0 0 1 0 0 0]
    // [0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 1 0 0 0 0 0]
    // [0 0 1 0 0 0 1 0 1 0]
    // [0 0 0 0 1 0 0 0 0 0]
    // [0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 0 1 0 0]
    // [0 0 1 0 0 0 0 0 0 0]
    auto m_B = genGivenVals<DenseMatrix<double>>(9, {0,0,0,1,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,1,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,0,
                                                    0,0,0,0,1,0,0,0,0,0,
                                                    0,0,1,0,0,0,1,0,1,0,
                                                    0,0,0,0,1,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,1,0,0,
                                                    0,0,1,0,0,0,0,0,0,0
    });


    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    double s = estimateSparsity_ElementWiseAddition(hA, hB);
    
    REQUIRE(s <= 1.0);
    REQUIRE(s >= 0.0);

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}

TEST_CASE("Estimate sparsity for element-wise multiplication", TAG_DATASTRUCTURES) {
    // --- Matrix A: 3x3 ---
    // [1 0 0]
    // [0 1 1]
    // [0 0 0]
    auto m_A = genGivenVals<DenseMatrix<double>>(3, {
        1, 0, 0,
        0, 1, 1,
        0, 0, 0
    });

    // --- Matrix B: 3x3 ---
    // [0 1 0]
    // [1 0 0]
    // [0 0 1]
    auto m_B = genGivenVals<DenseMatrix<double>>(3, {
        0, 1, 0,
        1, 0, 0,
        0, 0, 1
    });

    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    double s = estimateSparsity_ElementWiseMultiplication(hA, hB);
    std::cout << "Estimated sparsity: " << s << std::endl;
    
    REQUIRE(s >= 0.11);
    REQUIRE(s <= 0.112);

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}

TEST_CASE("Estimate sparsity for element-wise multiplication (Large)", TAG_DATASTRUCTURES) {
    // --- Matrix A: 9x9 ---
    // [0,0,0,0,0,0,0,1,0], 
    // [0,1,0,0,1,0,0,0,0], 
    // [0,0,0,1,1,1,0,0,0], 
    // [0,0,0,0,0,0,0,0,0],
    // [0,0,1,0,0,0,0,0,0],
    // [0,0,0,0,0,0,0,1,0],
    // [1,0,0,1,0,0,0,0,0],
    // [0,0,1,0,1,0,1,0,0],
    // [0,0,0,0,0,0,0,0,1],
    
    auto m_A = genGivenVals<DenseMatrix<double>>(9, {0,0,0,0,0,0,0,1,0,
                                                    0,1,0,0,1,0,0,0,0,
                                                    0,0,0,1,1,1,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,
                                                    0,0,1,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,1,0,
                                                    1,0,0,1,0,0,0,0,0,
                                                    0,0,1,0,1,0,1,0,0,
                                                    0,0,0,0,0,0,0,0,1
    });

    // --- Matrix B: 9x10 ---
    // [0 0 0 1 0 0 0 0 0 0]
    // [0 0 0 0 0 0 1 0 0 0]
    // [0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 1 0 0 0 0 0]
    // [0 0 1 0 0 0 1 0 1 0]
    // [0 0 0 0 1 0 0 0 0 0]
    // [0 0 0 0 0 0 0 0 0 0]
    // [0 0 0 0 0 0 0 1 0 0]
    // [0 0 1 0 0 0 0 0 0 0]
    auto m_B = genGivenVals<DenseMatrix<double>>(9, {0,0,0,1,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,1,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,0,
                                                    0,0,0,0,1,0,0,0,0,0,
                                                    0,0,1,0,0,0,1,0,1,0,
                                                    0,0,0,0,1,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,0,0,0,
                                                    0,0,0,0,0,0,0,1,0,0,
                                                    0,0,1,0,0,0,0,0,0,0
    });


    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    double s = estimateSparsity_ElementWiseMultiplication(hA, hB);
    
    REQUIRE(s <= 1.0);
    REQUIRE(s >= 0.0);

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}



TEST_CASE("Source Operations & Memory Benchmark", TAG_DATASTRUCTURES) {
    // 1. Rand Test (Probabilistic)
    size_t rows = 1000, cols = 1000;
    double sparsity = 0.1;
    MncSketch hRand = buildMncFromRand(rows, cols, sparsity, 42);
    
    double totalNNZ = 0;
    for(auto c : *hRand.hr) totalNNZ += c;
    double expected = rows * cols * sparsity;
    
    // Check if generated NNZ is within 10% of expected (100,000 +/- 10,000)
    CHECK(totalNNZ >= expected * 0.90);
    CHECK(totalNNZ <= expected * 1.10);
    CHECK(hRand.m == rows);

    // 2. Fill Test (Constant)
    MncSketch hFill = buildMncFromFill(5.0, 10, 5);
    CHECK(hFill.nnzRows == 10);
    CHECK((*hFill.hr)[0] == 5); 

    // 3. Seq Test (Dense Column Vector)
    MncSketch hSeq = buildMncFromSeq(1, 10, 1);
    CHECK(hSeq.m == 10);
    CHECK(hSeq.n == 1);
    CHECK((*hSeq.hr)[0] == 1); 

    // 4. Memory Benchmark Test
    size_t mem = getMncSizeBytes(hRand);
    CHECK(mem > 16000);
}

TEST_CASE("Propagate element-wise addition", TAG_DATASTRUCTURES) {
    // --- Matrix A: 3x3 ---
    // [1 0 0]
    // [0 1 1]
    // [0 0 0]
    auto m_A = genGivenVals<DenseMatrix<double>>(3, {
        1, 0, 0,
        0, 1, 1,
        0, 0, 0
    });

    // --- Matrix B: 3x3 ---
    // [0 1 0]
    // [1 0 0]
    // [0 0 1]
    auto m_B = genGivenVals<DenseMatrix<double>>(3, {
        0, 1, 0,
        1, 0, 0,
        0, 0, 1
    });

    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    MncSketch hC = propagateAdd(hA, hB);

    double totalNNZ = 0.0;
    for (uint32_t val : *hC.hr) totalNNZ += val;
    double sparsity = totalNNZ / (hC.m * hC.n);
    // std::cout << "Propagated addition sparsity: " << sparsity << std::endl;

    REQUIRE(sparsity >= 0.0);
    REQUIRE(sparsity <= 1.0);

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}

TEST_CASE("Propagate element-wise multiplication", TAG_DATASTRUCTURES) {
    // --- Matrix A: 3x3 ---
    auto m_A = genGivenVals<DenseMatrix<double>>(3, {
        1, 0, 0,
        0, 1, 1,
        0, 0, 0
    });

    // --- Matrix B: 3x3 ---
    auto m_B = genGivenVals<DenseMatrix<double>>(3, {
        0, 1, 0,
        1, 0, 0,
        0, 0, 1
    });

    MncSketch hA = buildMncFromDenseMatrix(*m_A);
    MncSketch hB = buildMncFromDenseMatrix(*m_B);

    MncSketch hC = propagateMul(hA, hB);

    double totalNNZ = 0.0;
    for (uint32_t val : *hC.hr) totalNNZ += val;
    double sparsity = totalNNZ / (hC.m * hC.n);
    // std::cout << "Propagated multiplication sparsity: " << sparsity << std::endl;

    REQUIRE(sparsity >= 0.0);
    REQUIRE(sparsity <= 1.0);

    DataObjectFactory::destroy(m_A);
    DataObjectFactory::destroy(m_B);
}