/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * Modifications 2024 The DAPHNE Consortium.
 */

// This code has been manually translated from Apache SystemDS (and
// significantly adapted).

#include <api/cli/Utils.h>

#include <tags.h>

#include <catch.hpp>

#include <sstream>
#include <string>


void runKmeans(int num_clusters, int max_points, double noise_level, int dimensions, double max_values, int expected_result) {
    const std::string scriptFileName =
        ("test/api/cli/algorithms/new_kmeans.daphne");
    std::stringstream out;
    std::stringstream err;
    const std::string arg1 = "arg1=" + std::to_string(num_clusters);
    const std::string arg2 = "arg2=" + std::to_string(max_points);
    const std::string arg3 = "arg3=" + std::to_string(noise_level);
    const std::string arg4 = "arg4=" + std::to_string(dimensions);
    const std::string arg5 = "arg5=" + std::to_string(max_values);
    int status = runDaphne(out, err, scriptFileName.c_str(), arg1.c_str(), arg2.c_str(), arg3.c_str(), arg4.c_str(), arg5.c_str());
    CHECK(status == StatusCode::SUCCESS);
    int output = std::stod(out.str());
    CHECK(output == expected_result);
}

// Basic Functionality

TEST_CASE("Kmeans - Multiple Clusters moderate noise", TAG_ALGORITHMS) {
    runKmeans(3, 100, 0.05, 3, 20.0, 0);
}

TEST_CASE("Kmeans - Many Clusters moderate noise", TAG_ALGORITHMS) {
    runKmeans(10, 20, 0.05, 3, 10.0, 0);
}


// Edge Cases

TEST_CASE("Kmeans - Single cluster", TAG_ALGORITHMS) {
    runKmeans(1, 100, 0.05, 3, 10.0, 0);
}

TEST_CASE("Kmeans - High noise", TAG_ALGORITHMS) {
    runKmeans(2, 10, 0.25, 3, 10.0, 0);
}

TEST_CASE("Kmeans - No noise", TAG_ALGORITHMS) {
    runKmeans(5, 50, 0.0, 3, 10.0, 0); 
}

