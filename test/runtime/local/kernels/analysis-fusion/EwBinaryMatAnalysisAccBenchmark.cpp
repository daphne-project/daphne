#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/kernels/RandMatrix.h>
#include <runtime/local/kernels/analysis-fusion/AnalysisFlags.h>
#include <runtime/local/kernels/analysis-fusion/EwBinaryMatAnalAcc.h>
#include <runtime/local/kernels/analysis-fusion/Util.h>

#include <catch.hpp>
#include <tags.h>

#include <chrono>
#include <cstdint>
#include <fstream>

#define RUNONEP(OP, SP, VT, ...)                                                                                       \
    do {                                                                                                               \
        using DT = MatrixRepr<VT, SP>;                                                                                 \
        const BinaryOpCode opcode = BinaryOpCode::OP;                                                                  \
        for (size_t n : {1024, 2048, 4096}) {                                                                          \
            for (int seed = 0; seed < 10; seed++) {                                                                    \
                DT *lhs = nullptr;                                                                                     \
                DT *rhs = nullptr;                                                                                     \
                RandMatrix<DT, VT>::apply(lhs, n, n, -100, 100, SP, seed, nullptr);                                    \
                RandMatrix<DT, VT>::apply(rhs, n, n, -100, 100, SP, seed * 2, nullptr);                                \
                {                                                                                                      \
                    DT *res = nullptr;                                                                                 \
                    auto t0 = std::chrono::high_resolution_clock::now();                                               \
                    ewBinaryMatAnalAcc<DT, DT, DT, AnalysisFlags<__VA_ARGS__>>(opcode, res, lhs, rhs, nullptr);        \
                    auto t1 = std::chrono::high_resolution_clock::now();                                               \
                    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();                             \
                    DataObjectFactory::destroy(res);                                                                   \
                    csv << #OP << "," << #VT << "," << "\"" #__VA_ARGS__ "\"" << "," << n << "," << (SP) << "," << ns  \
                        << "\n";                                                                                       \
                }                                                                                                      \
                DataObjectFactory::destroy(lhs);                                                                       \
                DataObjectFactory::destroy(rhs);                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    } while (0);

#define RUNONE(OP, SP, ...)                                                                                            \
    RUNONEP(OP, SP, double, __VA_ARGS__);                                                                              \
    RUNONEP(OP, SP, int64_t, __VA_ARGS__);

#define RUNALLSP(OP, ...)                                                                                              \
    RUNONE(OP, 0.001, __VA_ARGS__)                                                                                     \
    RUNONE(OP, 0.01, __VA_ARGS__)                                                                                      \
    RUNONE(OP, 0.05, __VA_ARGS__)                                                                                      \
    RUNONE(OP, 0.1, __VA_ARGS__)                                                                                       \
    RUNONE(OP, 0.2, __VA_ARGS__)                                                                                       \
    RUNONE(OP, 0.25, __VA_ARGS__)                                                                                      \
    RUNONE(OP, 1.0, __VA_ARGS__)

#define RUNALL(...)                                                                                                    \
    RUNALLSP(ADD, __VA_ARGS__)                                                                                         \
    RUNALLSP(MUL, __VA_ARGS__)

TEST_CASE("EwBinaryMatAnalysisAcc Benchmark", "[AnalysisAccBench][EwBinaryMatAnalysisAccBench]") {
    std::ofstream csv("/tmp/benchmark/EwBinaryMatAnalysisAcc.csv");
    csv << "op,element_type,analysis,n,sparsity,time_ns\n";
    RUNALL(AnalysisFlag::min);
    RUNALL(AnalysisFlag::min, AnalysisFlag::max);
    RUNALL(AnalysisFlag::numDistinct);
    RUNALL(AnalysisFlag::numDistinctApprox);
    RUNALL(AnalysisFlag::sparsity);
    RUNALL(AnalysisFlag::symmetry);
    RUNALL(AnalysisFlag::min, AnalysisFlag::max, AnalysisFlag::mean, AnalysisFlag::sparsity, AnalysisFlag::symmetry,
           AnalysisFlag::numDistinctApprox);
    csv.close();
}