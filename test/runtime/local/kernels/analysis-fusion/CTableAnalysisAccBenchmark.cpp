#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/RandMatrix.h>
#include <runtime/local/kernels/analysis-fusion/AnalysisFlags.h>
#include <runtime/local/kernels/analysis-fusion/CTableAnalAcc.h>
#include <runtime/local/kernels/analysis-fusion/Util.h>

#include <catch.hpp>
#include <tags.h>

#include <chrono>
#include <cstdint>
#include <fstream>

#define RUNONEP(RES, SP, VTW, ...)                                                                                     \
    do {                                                                                                               \
        using DTRes = RES<VTW>;                                                                                        \
        using DTCoord = DenseMatrix<int64_t>;                                                                          \
        for (size_t n : {1024,2048,4096}) {                                                                                      \
            for (int seed = 0; seed < 10; seed++) {                                                                    \
                DTCoord *lhs = nullptr;                                                                                \
                DTCoord *rhs = nullptr;                                                                                \
                RandMatrix<DTCoord, int64_t>::apply(lhs, n, 1, 0, 100, SP, seed, nullptr);                             \
                RandMatrix<DTCoord, int64_t>::apply(rhs, n, 1, 0, 100, SP, seed * 2, nullptr);                         \
                {                                                                                                      \
                    DTRes *res = nullptr;                                                                              \
                    auto t0 = std::chrono::high_resolution_clock::now();                                               \
                    ctableAnalAcc<DTRes, DTCoord, DTCoord, VTW, AnalysisFlags<__VA_ARGS__>>(                           \
                        res, lhs, rhs, static_cast<VTW>(1), -1, -1, nullptr);                                          \
                    auto t1 = std::chrono::high_resolution_clock::now();                                               \
                    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();                             \
                    DataObjectFactory::destroy(res);                                                                   \
                    csv << #RES << "," << #VTW << "," << "\"" #__VA_ARGS__ "\"" << "," << n << "," << (SP) << ","      \
                        << ns << "\n";                                                                                 \
                }                                                                                                      \
                DataObjectFactory::destroy(lhs);                                                                       \
                DataObjectFactory::destroy(rhs);                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    } while (0);

#define RUNONE(RES, SP, ...)                                                                                           \
    RUNONEP(RES, SP, double, __VA_ARGS__);                                                                             \
    RUNONEP(RES, SP, int64_t, __VA_ARGS__);

#define RUNALLSP(RES, ...)                                                                                             \
    RUNONE(RES, 0.001, __VA_ARGS__)                                                                                    \
    RUNONE(RES, 0.01, __VA_ARGS__)                                                                                     \
    RUNONE(RES, 0.05, __VA_ARGS__)                                                                                     \
    RUNONE(RES, 0.1, __VA_ARGS__)                                                                                      \
    RUNONE(RES, 0.2, __VA_ARGS__)                                                                                      \
    RUNONE(RES, 0.25, __VA_ARGS__)                                                                                     \
    RUNONE(RES, 1.0, __VA_ARGS__)

// both result representations
#define RUNALL(...)                                                                                                    \
    RUNALLSP(DenseMatrix, __VA_ARGS__)                                                                                 \
    RUNALLSP(CSRMatrix, __VA_ARGS__)

TEST_CASE("CTableAnalysisAcc Benchmark", "[AnalysisAccBench][CTableAnalysisAccBench]") {
    std::ofstream csv("/tmp/benchmark/CTableAnalysisAcc.csv");
    csv << "result_repr,element_type,analysis,n,sparsity,time_ns\n";
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