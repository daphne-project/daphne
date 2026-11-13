#include "runtime/local/kernels/analysis-fusion/AnalysisFlags.h"
#include <catch.hpp>
template <typename DT, AnalysisFlag... Fs> void checkAnalysisResult(const DT *arg, const DT *exp) {
    using anal_t = AnalysisFlags<Fs...>;
    if constexpr (anal_t::template contains<AnalysisFlag::mean>) {
        CHECK(arg->mean.has_value() == exp->mean.has_value());
        if (arg->mean.has_value())
            CHECK(std::abs(arg->mean.value() - exp->mean.value()) < 1e-2);
    }
    if constexpr (anal_t::template contains<AnalysisFlag::min>) {
        CHECK(arg->min.has_value() == exp->min.has_value());
        if (arg->min.has_value())
            CHECK(arg->min.value() == exp->min.value());
    }
    if constexpr (anal_t::template contains<AnalysisFlag::max>) {
        CHECK(arg->max.has_value() == exp->max.has_value());
        if (arg->max.has_value())
            CHECK(arg->max.value() == exp->max.value());
    }

    if constexpr (anal_t::template contains<AnalysisFlag::sparsity>)
        CHECK(std::abs(arg->sparsity - exp->sparsity) < 1e-10);

    if constexpr (anal_t::template contains<AnalysisFlag::symmetry>)
        CHECK(arg->symmetric == exp->symmetric);

    if constexpr (anal_t::template contains<AnalysisFlag::numDistinct>) {
        CHECK(arg->numDistinct.has_value() == exp->numDistinct.has_value());
        CHECK(arg->numDistinct.value() == exp->numDistinct.value());
    }

    if constexpr (anal_t::template contains<AnalysisFlag::numDistinctApprox>) {
        CHECK(arg->numDistinct.has_value() == exp->numDistinct.has_value());
        const double exp_cnt = static_cast<double>(exp->numDistinct.value());
        const double arg_cnt = static_cast<double>(arg->numDistinct.value());
        CHECK(std::abs(exp_cnt - arg_cnt) / exp_cnt < 0.25);
    }
}

/*
    For the `rand()` (RandMatrix kernel) operation, the Daphne compiler uses CSR for sparsities lower than 0.25
    For testing, we use the same threshold.
*/
template <typename VT, double Sparsity>
using MatrixRepr = std::conditional_t<(Sparsity < 0.25), CSRMatrix<VT>, DenseMatrix<VT>>;