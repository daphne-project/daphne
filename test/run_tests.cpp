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

#define CATCH_CONFIG_MAIN // make catch2 generate a main-function
#include <catch.hpp>

#include "run_tests.h"

#include "runtime/local/kernels/CreateDaphneContext.h"
#include <api/cli/DaphneUserConfig.h>
#ifdef USE_CUDA
#include "runtime/local/kernels/CUDA/CreateCUDAContext.h"
#endif

#include "spdlog/cfg/env.h"

std::unique_ptr<DaphneContext> setupContextAndLogger() {
    if (not logger) {
        logger = std::make_unique<DaphneLogger>(user_config);
        user_config.log_ptr->registerLoggers();
        spdlog::cfg::load_env_levels();
    }

    DaphneContext *dctx_;
    createDaphneContext(dctx_, reinterpret_cast<uint64_t>(&user_config), reinterpret_cast<uint64_t>(&dispatchMapping),
                        reinterpret_cast<uint64_t>(&Statistics::instance()),
                        reinterpret_cast<uint64_t>(&PropertyLogger::instance()),
                        reinterpret_cast<uint64_t>(&StringRefCounter::instance()));

#ifdef USE_CUDA
    CUDA::createCUDAContext(dctx_);
#endif

    return std::unique_ptr<DaphneContext>(dctx_);
}

// Nothing to do here, the individual test cases are in separate cpp-files.
