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

#ifndef TEST_TAGS_H
#define TEST_TAGS_H

// The following tags are intended to be used as the second argument to the
// TEST_CASE macros of catch2. You can easily combine tags by using multiple
// tag macros separated by whitespace, e.g., if TAG_A is "[a]" and TAG_B is
// "[b]", then TAG_A TAG_B is "[a]" "[b]", which is equivalent to "[a][b]".

#define TAG_ALGORITHMS "[algorithms]"
#define TAG_BUILD "[build]"
#define TAG_CAST "[cast]"
#define TAG_CODEGEN "[codegen]"
#define TAG_CONFIG "[config]"
#define TAG_CONTROLFLOW "[controlflow]"
#define TAG_DAPHNELIB "[daphnelib]"
#define TAG_DATASTRUCTURES "[datastructures]"
#define TAG_DISTRIBUTED "[distributed]"
#define TAG_DNN "[dnn]"
#define TAG_EXTENSIBILITY "[extensibility]"
#define TAG_FRAME_LITERAL "[frameliterals]"
#define TAG_FUNCTIONS "[functions]"
#define TAG_IMPORT "[import]"
#define TAG_INDEXING "[indexing]"
#define TAG_INFERENCE "[inference]"
#define TAG_IO "[io]"
#define TAG_KERNELS "[kernels]"
#define TAG_LISTS "[lists]"
#define TAG_LITERALS "[literals]"
#define TAG_MATMUL "[matmul]"
#define TAG_MATRIX_LITERAL "[matrixliterals]"
#define TAG_OPERATIONS "[operations]"
#define TAG_PARSER "[parser]"
#define TAG_SCOPING "[scoping]"
#define TAG_SCRIPTARGS "[scriptargs]"
#define TAG_SECONDORDER "[secondorder]"
#define TAG_SQL "[sql]"
#define TAG_SYNTAX "[syntax]"
#define TAG_TERNARY "[ternary]"
#define TAG_UTIL "[util]"
#define TAG_VECTORIZED "[vectorized]"

#endif // TEST_TAGS_H
