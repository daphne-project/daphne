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

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/kernels/AggAll.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwUnaryMat.h>
#include <runtime/local/kernels/RandMatrix.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/kernels/analysis-fusion/EwUnaryMatAnalysis.h>

#include <tags.h>

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch.hpp>

#include <cstdint>
#include <string>

#define TEST_NAME(opName) "EwUnaryMatAnalysisBench (" opName ")"
#define DATA_TYPES DenseMatrix, CSRMatrix, Matrix
#define VALUE_TYPES int32_t, double

static const std::vector<size_t> SIZES = {
    64,
    256,
    1024,
};
static const std::vector<double> SPARSITIES = {0.0, 0.001, 0.01, 0.1};
static const std::vector<UnaryOpCode> OPCODES = {UnaryOpCode::MINUS, UnaryOpCode::EXP, UnaryOpCode::FLOOR};

TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Sparse: fused vs naive"), "[EwBenchmark]", (CSRMatrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;

    for (size_t n : SIZES) {
        for (double sparsity : SPARSITIES) {
            DT *arg = nullptr;
            RandMatrix<DT, VT>::apply(arg, n, n, VT(-100), VT(100), sparsity, n, nullptr);

            for (UnaryOpCode opcode : OPCODES) {
                // generate a random matrix to be used for all kernel combinations

                std::string section = "n=" + std::to_string(n) + " sparsity=" + std::to_string(sparsity) +
                                      " op=" + std::string(unary_op_codes[static_cast<size_t>(opcode)]);
                SECTION(section) {
                    // baseline: unfused
                    BENCHMARK_ADVANCED(("unfused: kernel only"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMat<DT, DT>::apply(opcode, res, arg, nullptr);
                            return res;
                        });
                        DataObjectFactory::destroy(res);
                    };

                    // naive consecutive kernels
                    BENCHMARK_ADVANCED(("unfused: kernel + mean"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMat<DT, DT>::apply(opcode, res, arg, nullptr);
                            return aggAll<double, DT>(AggOpCode::MEAN, res, nullptr);
                        });
                        DataObjectFactory::destroy(res);
                    };

                    BENCHMARK_ADVANCED(("unfused: kernel + min"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMat<DT, DT>::apply(opcode, res, arg, nullptr);
                            return aggAll<VT, DT>(AggOpCode::MIN, res, nullptr);
                        });
                        DataObjectFactory::destroy(res);
                    };

                    BENCHMARK_ADVANCED(("unfused: kernel + mean + min"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMat<DT, DT>::apply(opcode, res, arg, nullptr);
                            return aggAll<VT, DT>(AggOpCode::MIN, res, nullptr) +
                                   aggAll<double, DT>(AggOpCode::MEAN, res, nullptr);
                        });
                        DataObjectFactory::destroy(res);
                    };

                    // Fused benchmarks
                    BENCHMARK_ADVANCED(("fused: kernel + mean"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::mean>>::apply(opcode, res, arg,
                                                                                                 nullptr);
                            return res;
                        });
                        DataObjectFactory::destroy(res);
                    };

                    BENCHMARK_ADVANCED(("fused: kernel + min"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::min>>::apply(opcode, res, arg,
                                                                                                nullptr);
                            return res;
                        });
                        DataObjectFactory::destroy(res);
                    };

                    BENCHMARK_ADVANCED(("fused: kernel + mean + min"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::mean, AnalysisFlag::min>>::apply(
                                opcode, res, arg, nullptr);
                            return res;
                        });
                        DataObjectFactory::destroy(res);
                    };

                    BENCHMARK_ADVANCED(("fused: kernel + sparsity"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::sparsity>>::apply(opcode, res, arg,
                                                                                                     nullptr);
                            return res;
                        });
                        DataObjectFactory::destroy(res);
                    };

                    BENCHMARK_ADVANCED(("fused: kernel + symmetry"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::symmetry>>::apply(opcode, res, arg,
                                                                                                     nullptr);
                            return res;
                        });
                        DataObjectFactory::destroy(res);
                    };

                    BENCHMARK_ADVANCED(("fused: kernel + numDistinct"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::numDistinct>>::apply(opcode, res,
                                                                                                        arg, nullptr);
                            return res;
                        });
                        DataObjectFactory::destroy(res);
                    };

                    BENCHMARK_ADVANCED(("fused: kernel + all analyses"))(Catch::Benchmark::Chronometer meter) {
                        DT *res = nullptr;
                        meter.measure([&] {
                            EwUnaryMatAnalysis<
                                DT, DT,
                                AnalysisFlags<AnalysisFlag::min, AnalysisFlag::mean, AnalysisFlag::sparsity,
                                              AnalysisFlag::symmetry, AnalysisFlag::numDistinct>>::apply(opcode, res,
                                                                                                         arg, nullptr);
                            return res;
                        });
                        DataObjectFactory::destroy(res);
                    };
                }
            }
            DataObjectFactory::destroy(arg);
        }
    }
}
TEMPLATE_PRODUCT_TEST_CASE(TEST_NAME("Dense: fused vs naive"), "[EwBenchmark]", (DenseMatrix, Matrix), (VALUE_TYPES)) {
    using DT = TestType;
    using VT = typename DT::VT;
    const auto sparsity = 1.;

    for (size_t n : SIZES) {
        // generate a random matrix to be used for all kernel combinations
        DT *arg = nullptr;
        RandMatrix<DT, VT>::apply(arg, n, n, VT(-100), VT(100), sparsity, n, nullptr);

        for (UnaryOpCode opcode : OPCODES) {
            std::string section =
                "n=" + std::to_string(n) + " op=" + std::string(unary_op_codes[static_cast<size_t>(opcode)]);
            SECTION(section) {
                // baseline: unfused
                BENCHMARK_ADVANCED(("unfused: kernel only"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMat<DT, DT>::apply(opcode, res, arg, nullptr);
                        return res;
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("unfused: kernel + mean"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMat<DT, DT>::apply(opcode, res, arg, nullptr);
                        return aggAll<double, DT>(AggOpCode::MEAN, res, nullptr);
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("unfused: kernel + min"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMat<DT, DT>::apply(opcode, res, arg, nullptr);
                        return aggAll<VT, DT>(AggOpCode::MIN, res, nullptr);
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("unfused: kernel + mean + min"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMat<DT, DT>::apply(opcode, res, arg, nullptr);
                        return aggAll<double, DT>(AggOpCode::MEAN, res, nullptr) +
                               aggAll<VT, DT>(AggOpCode::MIN, res, nullptr);
                    });
                    DataObjectFactory::destroy(res);
                };

                // Fused benchmarks
                BENCHMARK_ADVANCED(("fused: kernel + mean"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::mean>>::apply(opcode, res, arg, nullptr);
                        return res;
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("fused: kernel + min"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::min>>::apply(opcode, res, arg, nullptr);
                        return res;
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("fused: kernel + mean + min"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::mean, AnalysisFlag::min>>::apply(
                            opcode, res, arg, nullptr);
                        return res;
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("fused: kernel + sparsity"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::sparsity>>::apply(opcode, res, arg,
                                                                                                 nullptr);
                        return res;
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("fused: kernel + symmetry"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::symmetry>>::apply(opcode, res, arg,
                                                                                                 nullptr);
                        return res;
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("fused: kernel + numDistinct"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMatAnalysis<DT, DT, AnalysisFlags<AnalysisFlag::numDistinct>>::apply(opcode, res, arg,
                                                                                                    nullptr);
                        return res;
                    });
                    DataObjectFactory::destroy(res);
                };

                BENCHMARK_ADVANCED(("fused: kernel + all analyses"))(Catch::Benchmark::Chronometer meter) {
                    DT *res = nullptr;
                    meter.measure([&] {
                        EwUnaryMatAnalysis<
                            DT, DT,
                            AnalysisFlags<AnalysisFlag::min, AnalysisFlag::mean, AnalysisFlag::sparsity,
                                          AnalysisFlag::symmetry, AnalysisFlag::numDistinct>>::apply(opcode, res, arg,
                                                                                                     nullptr);
                        return res;
                    });
                    DataObjectFactory::destroy(res);
                };
            }
        }
        DataObjectFactory::destroy(arg);
    }
}