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

#include <string>

#include <cstdlib>
#include <cstring>

const std::string dirPath = "test/api/python/";

#define MAKE_TEST_CASE(name)                                                                                           \
    TEST_CASE(name ".py", TAG_DAPHNELIB) {                                                                             \
        const std::string prefix = dirPath + name;                                                                     \
        compareDaphneToDaphneLib(prefix + ".py", prefix + ".daphne");                                                  \
    }
#define MAKE_TEST_CASE_ENVVAR(name, envVar)                                                                            \
    TEST_CASE(name ".py", TAG_DAPHNELIB) {                                                                             \
        const char *depAvail = std::getenv(envVar);                                                                    \
        if (depAvail == nullptr) {                                                                                     \
            FAIL("this test case requires environment variable " envVar                                                \
                 " to be set to either 0 or 1, but it is unset");                                                      \
        }                                                                                                              \
        if (!strcmp(depAvail, "1")) {                                                                                  \
            const std::string prefix = dirPath + name;                                                                 \
            compareDaphneToDaphneLib(prefix + ".py", prefix + ".daphne");                                              \
        } else if (!strcmp(depAvail, "0")) {                                                                           \
            SUCCEED("this test case is skipped since environment variable " envVar " is 0");                           \
        } else {                                                                                                       \
            FAIL("this test case requires environment variable " envVar                                                \
                 " to be set to either 0 or 1, but it is something else");                                             \
        }                                                                                                              \
    }
#define MAKE_TEST_CASE_SCALAR(name)                                                                                    \
    TEST_CASE(name ".py", TAG_DAPHNELIB) {                                                                             \
        const std::string prefix = dirPath + name;                                                                     \
        compareDaphneToDaphneLibScalar(prefix + ".py", prefix + ".daphne");                                            \
    }
#define MAKE_TEST_CASE_PARAMETRIZED(name, argument)                                                                    \
    TEST_CASE((std::string(name) + ".py, " + std::string(argument)).c_str(), TAG_DAPHNELIB) {                          \
        const std::string prefix = dirPath + name;                                                                     \
        compareDaphneToDaphneLib(prefix + ".py", prefix + ".daphne", argument);                                        \
    }
#define MAKE_TEST_CASE_STR(name, str)                                                                                  \
    TEST_CASE(name ".py", TAG_DAPHNELIB) {                                                                             \
        const std::string prefix = dirPath + name;                                                                     \
        compareDaphneLibToStr(str, prefix + ".py");                                                                    \
    }

MAKE_TEST_CASE("data_transfer_numpy_array_float64_1d")
MAKE_TEST_CASE("data_transfer_numpy_array_float64_1d_vector")
MAKE_TEST_CASE("data_transfer_numpy_array_float64_1d_shared_memory")
MAKE_TEST_CASE("data_transfer_numpy_array_float64_2d")
MAKE_TEST_CASE("data_transfer_numpy_array_float64_2d_shared_memory")
MAKE_TEST_CASE("data_transfer_numpy_array_float64_large_ones")
MAKE_TEST_CASE("data_transfer_numpy_array_float64_large_ones_shared_memory")
MAKE_TEST_CASE("data_transfer_numpy_array_float64_large_zeros")
MAKE_TEST_CASE("data_transfer_numpy_array_float64_large_zeros_shared_memory")
MAKE_TEST_CASE("data_transfer_numpy_array_int64_1d")
MAKE_TEST_CASE("data_transfer_numpy_array_int64_1d_vector")
MAKE_TEST_CASE("data_transfer_numpy_array_string_1d")
MAKE_TEST_CASE("data_transfer_numpy_array_string_1d_vector")
MAKE_TEST_CASE("data_transfer_numpy_array_string_2d")
MAKE_TEST_CASE_STR("data_transfer_numpy_array_fixedstr16_bytes_shared_memory", "DenseMatrix(2x2, FixedStr)\n")
MAKE_TEST_CASE_STR("data_transfer_numpy_array_fixedstr16_unicode_shared_memory", "DenseMatrix(3x1, FixedStr)\n")

MAKE_TEST_CASE("data_transfer_python_list_float64_1d")
MAKE_TEST_CASE("data_transfer_python_list_float64_1d_shared_memory")
MAKE_TEST_CASE("data_transfer_python_list_float64_2d")
MAKE_TEST_CASE("data_transfer_python_list_float64_2d_shared_memory")
MAKE_TEST_CASE("data_transfer_python_list_float64_large_ones")
MAKE_TEST_CASE("data_transfer_python_list_float64_large_ones_shared_memory")
MAKE_TEST_CASE("data_transfer_python_list_float64_large_zeros")
MAKE_TEST_CASE("data_transfer_python_list_float64_large_zeros_shared_memory")
MAKE_TEST_CASE("data_transfer_python_list_int64_1d")
MAKE_TEST_CASE("data_transfer_python_list_string_1d")
MAKE_TEST_CASE("data_transfer_python_list_string_2d")

MAKE_TEST_CASE("data_transfer_pandas_series_float64")
MAKE_TEST_CASE("data_transfer_pandas_series_float64_shared_memory")
MAKE_TEST_CASE("data_transfer_pandas_series_float64_large_ones")
MAKE_TEST_CASE("data_transfer_pandas_series_float64_large_ones_shared_memory")
MAKE_TEST_CASE("data_transfer_pandas_series_float64_large_zeros")
MAKE_TEST_CASE("data_transfer_pandas_series_float64_large_zeros_shared_memory")
MAKE_TEST_CASE("data_transfer_pandas_series_int64")
MAKE_TEST_CASE("data_transfer_pandas_series_int64_shared_memory")
MAKE_TEST_CASE("data_transfer_pandas_series_string")

MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_float64_1d", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_float64_1d_shared_memory", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_float64_2d", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_float64_2d_shared_memory", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_float64_large_ones", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_float64_large_ones_shared_memory", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_float64_large_zeros", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_float64_large_zeros_shared_memory", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_tensor_int64_1d", "DAPHNE_DEP_AVAIL_PYTORCH")

MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_float64_1d", "DAPHNE_DEP_AVAIL_TENSFORFLOW")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_float64_1d_shared_memory", "DAPHNE_DEP_AVAIL_TENSFORFLOW")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_float64_2d", "DAPHNE_DEP_AVAIL_TENSFORFLOW")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_float64_2d_shared_memory", "DAPHNE_DEP_AVAIL_TENSFORFLOW")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_float64_large_ones", "DAPHNE_DEP_AVAIL_TENSFORFLOW")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_float64_large_ones_shared_memory",
                      "DAPHNE_DEP_AVAIL_TENSFORFLOW")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_float64_large_zeros", "DAPHNE_DEP_AVAIL_TENSFORFLOW")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_float64_large_zeros_shared_memory",
                      "DAPHNE_DEP_AVAIL_TENSFORFLOW")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_tensor_int64_1d", "DAPHNE_DEP_AVAIL_TENSFORFLOW")

MAKE_TEST_CASE("data_transfer_numpy_1")
MAKE_TEST_CASE("data_transfer_numpy_2")
MAKE_TEST_CASE("data_transfer_numpy_3")
MAKE_TEST_CASE("data_transfer_pandas_1")
MAKE_TEST_CASE("data_transfer_pandas_2")
MAKE_TEST_CASE("data_transfer_pandas_3_series")
MAKE_TEST_CASE("data_transfer_pandas_4_sparse_dataframe")
MAKE_TEST_CASE("data_transfer_pandas_5_categorical_dataframe")
MAKE_TEST_CASE_ENVVAR("data_transfer_pytorch_1", "DAPHNE_DEP_AVAIL_PYTORCH")
MAKE_TEST_CASE_ENVVAR("data_transfer_tensorflow_1", "DAPHNE_DEP_AVAIL_TENSFORFLOW")

MAKE_TEST_CASE("frame_innerJoin")
MAKE_TEST_CASE("frame_setColLabels")
MAKE_TEST_CASE("frame_setColLabelsPrefix")
MAKE_TEST_CASE("frame_to_matrix")
MAKE_TEST_CASE("random_matrix_generation")
MAKE_TEST_CASE("random_matrix_sum")
MAKE_TEST_CASE("random_matrix_addition")
MAKE_TEST_CASE("random_matrix_subtraction")
MAKE_TEST_CASE("random_matrix_mult")
MAKE_TEST_CASE("random_matrix_div")
MAKE_TEST_CASE("random_matrix_functions")
MAKE_TEST_CASE("context_datagen")
MAKE_TEST_CASE("scalar_ops")
MAKE_TEST_CASE("scalar_ewunary")
MAKE_TEST_CASE("scalar_ewbinary")
MAKE_TEST_CASE("frame_dimensions")
MAKE_TEST_CASE("frame_reorg")
MAKE_TEST_CASE("frame_cartesian")
MAKE_TEST_CASE("matrix_dimensions")
MAKE_TEST_CASE("matrix_ewunary")
MAKE_TEST_CASE("matrix_ewbinary")
MAKE_TEST_CASE("matrix_outerbinary")
MAKE_TEST_CASE("matrix_agg")
MAKE_TEST_CASE("matrix_reorg")
MAKE_TEST_CASE("matrix_other")
MAKE_TEST_CASE("matrix_preprocessing")
MAKE_TEST_CASE("matrix_indexing_1")
MAKE_TEST_CASE("matrix_indexing_2")
MAKE_TEST_CASE("matrix_ifelse_1")
MAKE_TEST_CASE("matrix_asType_1")
MAKE_TEST_CASE_STR("matrix_indexing_3", "90\n")
MAKE_TEST_CASE_SCALAR("numpy_matrix_ops")
MAKE_TEST_CASE_SCALAR("numpy_matrix_ops_extended")
MAKE_TEST_CASE("numpy_matrix_ops_replace")

// Tests for DaphneLib complex control flow.
MAKE_TEST_CASE_PARAMETRIZED("if_else_simple", "param=3.8")
MAKE_TEST_CASE_PARAMETRIZED("if_else_simple", "param=0.1")
MAKE_TEST_CASE_PARAMETRIZED("if_only_simple", "param=3.8")
MAKE_TEST_CASE_PARAMETRIZED("if_only_simple", "param=0.1")
MAKE_TEST_CASE_PARAMETRIZED("if_else_2_outputs", "param=3.8")
MAKE_TEST_CASE_PARAMETRIZED("if_else_2_outputs", "param=0.1")
MAKE_TEST_CASE_PARAMETRIZED("if_else_complex", "param=3.8")
MAKE_TEST_CASE_PARAMETRIZED("if_else_complex", "param=10.0")
MAKE_TEST_CASE_PARAMETRIZED("for_loop_simple", "param=1")
MAKE_TEST_CASE_PARAMETRIZED("for_loop_simple", "param=10")
MAKE_TEST_CASE_PARAMETRIZED("for_loop_with_step", "param=1")
MAKE_TEST_CASE_PARAMETRIZED("for_loop_with_step", "param=2")
MAKE_TEST_CASE_PARAMETRIZED("for_loop_use_iterable", "param=1")
MAKE_TEST_CASE_PARAMETRIZED("for_loop_use_iterable", "param=10")
MAKE_TEST_CASE_PARAMETRIZED("for_loop_2_outputs", "param=0")
MAKE_TEST_CASE_PARAMETRIZED("for_loop_2_outputs", "param=1")
// skipping the next test for now as it is not supported by Daphne yet
// (manipulating frame read from a file inside loop)
// MAKE_TEST_CASE("for_loop_with_frame")
MAKE_TEST_CASE("while_loop_simple")
MAKE_TEST_CASE("while_loop_complex_cond")
MAKE_TEST_CASE_PARAMETRIZED("while_loop_2_outputs", "param=0")
MAKE_TEST_CASE_PARAMETRIZED("while_loop_2_outputs", "param=1")
MAKE_TEST_CASE("do_while_loop_simple")
MAKE_TEST_CASE("do_while_loop_complex_cond")
MAKE_TEST_CASE_PARAMETRIZED("do_while_loop_2_outputs", "param=0")
MAKE_TEST_CASE_PARAMETRIZED("do_while_loop_2_outputs", "param=1")
MAKE_TEST_CASE_PARAMETRIZED("nested_control_flow_1", "param=0.1")
MAKE_TEST_CASE_PARAMETRIZED("nested_control_flow_1", "param=3.8")
MAKE_TEST_CASE_PARAMETRIZED("nested_control_flow_2", "param=0.1")
MAKE_TEST_CASE_PARAMETRIZED("nested_control_flow_2", "param=3.8")
MAKE_TEST_CASE("nested_control_flow_3")
MAKE_TEST_CASE("user_def_func_simple")
MAKE_TEST_CASE_PARAMETRIZED("user_def_func_1_input_3_outputs", "param=0")
MAKE_TEST_CASE_PARAMETRIZED("user_def_func_1_input_3_outputs", "param=1")
MAKE_TEST_CASE_PARAMETRIZED("user_def_func_1_input_3_outputs", "param=2")
MAKE_TEST_CASE("user_def_func_multiple_functions")
MAKE_TEST_CASE("user_def_func_multiple_calls")
MAKE_TEST_CASE("user_def_func_with_scalar")
MAKE_TEST_CASE("user_def_func_3_inputs")
// skipping the next test for now as it is not supported by Daphne yet
// (manipulating variables in a nested block inside a function)
// MAKE_TEST_CASE_PARAMETRIZED("user_def_func_with_condition", "param=0.1")
// MAKE_TEST_CASE_PARAMETRIZED("user_def_func_with_condition", "param=3.8")
// MAKE_TEST_CASE("user_def_func_with_for_loop")
// MAKE_TEST_CASE("user_def_func_with_while_loop")
