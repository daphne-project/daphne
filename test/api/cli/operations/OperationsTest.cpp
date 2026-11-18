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

#include <api/cli/StatusCode.h>
#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>
#include <vector>

const std::string dirPath = "test/api/cli/operations/";

#define MAKE_TEST_CASE(name, count)                                                                                    \
    TEST_CASE(name, TAG_OPERATIONS) {                                                                                  \
        for (unsigned i = 1; i <= count; i++) {                                                                        \
            DYNAMIC_SECTION(name "_" << i << ".daphne") { compareDaphneToRefSimple(dirPath, name, i); }                \
        }                                                                                                              \
    }

MAKE_TEST_CASE("aggMax", 1)
MAKE_TEST_CASE("aggMin", 1)
MAKE_TEST_CASE("bin", 2)
MAKE_TEST_CASE("cbind", 1)
MAKE_TEST_CASE("createFrame", 2)
MAKE_TEST_CASE("ctable", 1)
MAKE_TEST_CASE("fill", 1)
MAKE_TEST_CASE("gemv", 1)
MAKE_TEST_CASE("groupSum", 1)
MAKE_TEST_CASE("idxMax", 1)
MAKE_TEST_CASE("idxMin", 1)
MAKE_TEST_CASE("innerJoin", 1)
MAKE_TEST_CASE("isNan", 1)
MAKE_TEST_CASE("isSymmetric", 1)
MAKE_TEST_CASE("lower", 1)
MAKE_TEST_CASE("mean", 1)
MAKE_TEST_CASE("oneHot", 1)
MAKE_TEST_CASE("operator_at", 2)
MAKE_TEST_CASE("operator_eq", 3)
MAKE_TEST_CASE("operator_gt", 1)
MAKE_TEST_CASE("operator_lt", 1)
MAKE_TEST_CASE("operator_minus", 4)
MAKE_TEST_CASE("operator_plus", 2)
MAKE_TEST_CASE("operator_slash", 1)
MAKE_TEST_CASE("operator_times", 1)
MAKE_TEST_CASE("order", 1)
MAKE_TEST_CASE("rbind", 1)
MAKE_TEST_CASE("recode", 4)
MAKE_TEST_CASE("replace", 1)
MAKE_TEST_CASE("reshape", 3)
MAKE_TEST_CASE("reverse", 1)
MAKE_TEST_CASE("semiJoin", 1)
MAKE_TEST_CASE("seq", 2)
MAKE_TEST_CASE("solve", 1)
MAKE_TEST_CASE("sqrt", 1)
MAKE_TEST_CASE("sum", 1)
MAKE_TEST_CASE("syrk", 1)
MAKE_TEST_CASE("transpose", 1)
MAKE_TEST_CASE("upper", 1)

// Checks if the result of both `t(X) @ X` and `X @ t(X)`, when using physical operator selection (default args), is:
// (1) equal to the result of the same expressions when not using physical operator selection (up to round-off errors),
// and (2) symmetric (there should not even be round-off errors).
TEST_CASE("operator_at_symmetry", TAG_OPERATIONS) {
    // We use a few different shapes for the matrix X, since an incorrect implementation might still produce symmetric
    // results for some shapes (before the commit that introduced this test case, `X @ t(X)` was lowered to the
    // matmul-kernel, which didn't produce symmetric results for some input shapes). We ensure that the result is
    // symmetric by lowering to the syrk-kernel. To check if the result is (not only symmetric but also) correct (up to
    // round-off errors), we compare it to the result without the rewrite from MatMulOp to SyrkOp, by passing
    // `--no-phy-op-selection`.

    std::string scriptFilePath = dirPath + "operator_at_symmetry.daphne";

    std::string checkSymArgFalse = "checkSymmetry=false";
    std::string checkSymArgTrue = "checkSymmetry=true";

    std::string sep = "-----\n";
    std::string outExpSym = "t(X) @ X is symmetric\nX @ t(X) is symmetric\n" + sep;

    std::vector<size_t> choiceSize = {1, 3, 57, 100, 256};
    for (size_t numRows : choiceSize)
        for (size_t numCols : choiceSize) {
            DYNAMIC_SECTION("" << numRows << "x" << numCols) {
                std::string numRowsArg = "numRows=" + std::to_string(numRows);
                std::string numColsArg = "numCols=" + std::to_string(numCols);

                // In all variable names:
                // "1" stands for "with physical operator selection"
                // "2" stands for "without physical operator selection"

                // Run DAPHNE with physical operator selection (default args).
                std::stringstream out1, err1;
                int status1 = runDaphne(out1, err1, scriptFilePath.c_str(), numRowsArg.c_str(), numColsArg.c_str(),
                                        checkSymArgTrue.c_str());

                // Run DAPHNE without physical operator selection.
                std::stringstream out2, err2;
                int status2 = runDaphne(out2, err2, "--no-phy-op-selection", scriptFilePath.c_str(), numRowsArg.c_str(),
                                        numColsArg.c_str(), checkSymArgFalse.c_str());

                // Both runs of DAPHNE must succeed.
                // Just CHECK (don't REQUIRE) success, such that in case of a failure, the following checks run and
                // provide useful messages.
                CHECK(status1 == StatusCode::SUCCESS);
                CHECK(status2 == StatusCode::SUCCESS);

                std::string outStr1 = out1.str();
                std::string outStr2 = out2.str();

                // Extract the first section of the output, which contains the results of the two expressions.
                size_t pos1 = outStr1.find(sep);
                size_t pos2 = outStr2.find(sep);
                REQUIRE(pos1 > 0);                  // results output must no be empty
                REQUIRE(pos2 > 0);                  // results output must no be empty
                REQUIRE(pos1 != std::string::npos); // separator must be present
                REQUIRE(pos2 != std::string::npos); // separator must be present
                // The result of `t(X) @ X` must be the same (up to round-off errors) with and without physical operator
                // selection. The same must hold for `X @ t(X)`. An approximate check is achieved through the limited
                // precision when printing the matrices in DaphneDSL.
                CHECK(outStr1.substr(0, pos1) == outStr2.substr(0, pos2));

                // Only for the results with physical operator selection:
                // - Extract the second section of the output, which contains the results of the symmetry check.
                // - The results of both `t(X) @ X` and `X @ t(X)` must be symmetric.
                CHECK(outStr1.substr(pos1 + sep.size(), outExpSym.size()) == outExpSym);

                // We don't check the contents of the third section of the output, which contains the eigen values and
                // eigen vectors, since we are not interested in the concrete data; we just want to see if their
                // calculation succeeds.

                // The error output of both runs of DAPHNE must be empty.
                // Don't check empty(), because then catch2 doesn't display the error output.
                CHECK(err1.str() == "");
                CHECK(err2.str() == "");
            }
        }
}

TEST_CASE("operator_at_symmetry_inference", TAG_OPERATIONS) {
    // The compiler infers at compile-time that `t(X) @ X` and `X @ t(X)` are symmetric (a matrix product with the syrk
    // shape is symmetric for any real X). We check this by dumping the IR after property inference and looking for the
    // inferred symmetric property on the matmul results. This is the compile-time counterpart to the run-time
    // isSymmetric() check above.
    std::string scriptFilePath = dirPath + "operator_at_symmetry.daphne";

    std::stringstream out, err;
    int status = runDaphne(out, err, "--explain", "property_inference", scriptFilePath.c_str(), "numRows=3",
                           "numCols=4", "checkSymmetry=false");

    CHECK(status == StatusCode::SUCCESS);
    // Both matmul results (t(X) @ X and X @ t(X)) must be inferred symmetric. Counting "symmetric[true]" hits can't
    // tell whether both paths worked: each type is printed at its definition and again at every use, so one symetric
    // result already gives two hits. Count the matMul definition lines that carry the property instead, one per
    // result, so >= 2 means both got it.
    const std::string dump = err.str();
    size_t symMatMulDefs = 0;
    std::stringstream dumpLines(dump);
    for (std::string line; std::getline(dumpLines, line);)
        if (line.find("= \"daphne.matMul\"") != std::string::npos && line.find("symmetric[true]") != std::string::npos)
            ++symMatMulDefs;
    CHECK(symMatMulDefs >= 2);
}

TEST_CASE("operator_at_non_syrk_not_symmetric", TAG_OPERATIONS) {
    // Negative counterpart to operator_at_symmetry_inference: a plain product of two distinct matrices `X @ Y` has no
    // syrk shape and is not symmetric in general, so inference must NOT annotate its result symmetric[true]. Guards
    // against an over-eager MatMulOp::inferSymmetric that returns True for any product.
    std::string scriptFilePath = dirPath + "operator_at_non_syrk.daphne";

    std::stringstream out, err;
    int status = runDaphne(out, err, "--explain", "property_inference", scriptFilePath.c_str());

    CHECK(status == StatusCode::SUCCESS);
    CHECK_THAT(err.str(), !Catch::Contains("symmetric[true]"));
}

TEST_CASE("operator_at_square_not_symmetric", TAG_OPERATIONS) {
    // Same operand, but no transpose: `X @ X` is not a syrk shape and not symmetric in general, so it must NOT be
    // annotated symmetric[true]. Guards the `transa != transb` check that the distinct-operand X @ Y case never hits.
    std::string scriptFilePath = dirPath + "operator_at_square.daphne";

    std::stringstream out, err;
    int status = runDaphne(out, err, "--explain", "property_inference", scriptFilePath.c_str());

    CHECK(status == StatusCode::SUCCESS);
    CHECK_THAT(err.str(), !Catch::Contains("symmetric[true]"));
}

// Numerically proves the trace idiom `sum(diagVector(X @ Y)) -> sum(X * t(Y))`
// end-to-end through the full compiler. The `.daphne` script extracts the
// diagonal of the product so the rewrite fires; the `.ref.daphne` script sums
// the same trace by masking the product with an identity matrix, a chain the
// rewrite cannot match. Their outputs must agree byte for byte, which guards
// the transpose shape and operand pairing that the rewrite creates (a dropped
// transpose or wrong shape would make the emitted element-wise product
// dimensionally incoherent rather than throw).
TEST_CASE("rewrite_trace", TAG_OPERATIONS) { compareDaphneToSelfRefSimple(dirPath, "rewrite_trace", 1); }

// Numerically proves the row-scaling rewrite `diag(v) @ X -> X * v` end-to-end
// through the full compiler. The `.daphne` script uses diagMatrix() so the
// rewrite fires; the `.ref.daphne` script forms the same product via a plain
// diagonal matrix literal, which the rewrite leaves untouched. Their outputs
// must match byte for byte, which guards the operand-order and broadcast-axis
// invariants that a wrong rewrite would silently violate (the element-wise
// kernel would miscompute rather than throw).
TEST_CASE("rewrite_row_scale", TAG_OPERATIONS) { compareDaphneToSelfRefSimple(dirPath, "rewrite_row_scale", 1); }

// Numerically proves the column-scaling rewrite `X @ diag(v) -> X * t(v)`
// end-to-end through the full compiler. The `.daphne` script uses diagMatrix()
// so the rewrite fires; the `.ref.daphne` script forms the same product via a
// plain diagonal matrix literal, which the rewrite leaves untouched. Their
// outputs must match byte for byte, which guards the transpose, operand-order,
// and broadcast-axis invariants that a wrong rewrite would silently violate
// (a dropped transpose would scale rows instead of columns, or the element-wise
// kernel would take the wrong broadcast branch).
TEST_CASE("rewrite_col_scale", TAG_OPERATIONS) { compareDaphneToSelfRefSimple(dirPath, "rewrite_col_scale", 1); }

// Numerically proves the scalar-factor rewrite `sum(s * X) -> s * sum(X)`
// end-to-end through the full compiler. The `.daphne` script scales the matrix
// before aggregating so the rewrite fires; the `.ref.daphne` script scales the
// aggregated scalar instead, a chain the rewrite cannot match. Their outputs
// must agree byte for byte, which guards that hoisting the factor preserves the
// result (a dropped factor or wrong aggregate would change the printed value). An
// integer matrix is used because the regrouping is exact only over integers, the
// condition under which the rewrite fires.
TEST_CASE("rewrite_sum_scalar_factor", TAG_OPERATIONS) {
    compareDaphneToSelfRefSimple(dirPath, "rewrite_sum_scalar_factor", 1);
}

// Numerically proves the row-aggregate identity `sum(X, 0) -> X` for an n x 1
// matrix end-to-end through the full compiler. The `.daphne` script aggregates a
// single-column matrix so the rewrite fires; the `.ref.daphne` script prints the
// matrix directly, giving no aggregate to match. Their outputs must agree byte
// for byte, which guards that dropping the aggregate preserves the values.
TEST_CASE("rewrite_row_agg_dim1", TAG_OPERATIONS) { compareDaphneToSelfRefSimple(dirPath, "rewrite_row_agg_dim1", 1); }

// Numerically proves the repeated-add fold `X + X + X -> X * 3` end-to-end. The
// values are near 2^62 so the sum overflows si64; the rewrite must preserve the
// modular wraparound, so the `.daphne` chain of adds and the `.ref` plain
// multiply must print the same byte pattern. This is the one property a reviewer
// cannot check by inspection, so it is proved through the full compiler.
TEST_CASE("rewrite_repeated_add", TAG_OPERATIONS) { compareDaphneToSelfRefSimple(dirPath, "rewrite_repeated_add", 1); }

// Numerically proves the broadcast-minimization rewrite
// `(M1 + s1) + (M2 + s2) -> (M1 + M2) + (s1 + s2)` end-to-end. The scalars sum
// past 2^63 so the scalar add overflows si64; integer addition is associative in
// Z/2^64, so the `.daphne` and already-regrouped `.ref` forms must print the same
// wrapped result.
TEST_CASE("rewrite_broadcast_min", TAG_OPERATIONS) {
    compareDaphneToSelfRefSimple(dirPath, "rewrite_broadcast_min", 1);
}
