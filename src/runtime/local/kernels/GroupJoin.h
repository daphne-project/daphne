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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_GROUPJOIN_H
#define SRC_RUNTIME_LOCAL_KERNELS_GROUPJOIN_H

#include "ir/daphneir/Daphne.h"
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <cstddef>
#include <cstdint>

// TODO This entire implementation of the group-join is very inefficient and
// there are numerous opportunities for improvement. However, currently, we
// just need it to work.

// ****************************************************************************
// Utility function
// ****************************************************************************
// TODO Maybe this should be a kernel on its own.

using AggValue = std::variant<int64_t, uint64_t, double, std::string>;

struct AggState {
    const char *col;
    mlir::daphne::GroupEnum func;
    ValueTypeCode inputType;
    ValueTypeCode outputType;
    std::string label;
    AggValue value = int64_t{0};
    size_t count = 0;         // for COUNT and AVG
    bool initialized = false; // for MIN and MAX
};

struct JoinState {
    size_t lhsRow;
    bool matched;
    std::vector<AggState> aggStates;

    JoinState(size_t lhsRow, std::vector<AggState> aggStates)
        : lhsRow(lhsRow), matched(false), aggStates(std::move(aggStates)) {}
};

template <typename VTAgg> void updateAggState(AggState &aggState, VTAgg value) {
    using mlir::daphne::GroupEnum;
    switch (aggState.func) {
    case GroupEnum::COUNT:
        aggState.value = std::get<uint64_t>(aggState.value) + uint64_t{1};
        break;
    case GroupEnum::SUM:
        aggState.value += value;
        break;
    case GroupEnum::MIN:
        if (!aggState.initialized) {
            aggState.value = value;
            aggState.initialized = true;
        }
        aggState.value = std::min(aggState.value, value);
        break;
    case GroupEnum::MAX:
        if (!aggState.initialized) {
            aggState.value = value;
            aggState.initialized = true;
        }
        aggState.value = std::max(aggState.value, value);
        break;
    case GroupEnum::AVG:
        aggState.value += value;
        aggState.count++;
        break;
    }
}

template <typename VTAgg>
bool groupJoinColIf(ValueTypeCode vtcAgg, AggState &aggState, const Frame *rhs, size_t rhsRow, DCTX(ctx)) {
    if (vtcAgg != ValueTypeUtils::codeFor<VTAgg>)
        return false;
    updateAggState<VTAgg>(aggState, rhs->getColumn<VTAgg>(aggState.col)->get(rhsRow, 0));
    return true;
}

inline AggValue initAggValue(mlir::daphne::GroupEnum func, ValueTypeCode inputType) {
    using mlir::daphne::GroupEnum;
    switch (func) {
    case GroupEnum::COUNT:
        return uint64_t(0);
    case GroupEnum::AVG:
        return double(0);
    case GroupEnum::SUM:
    case GroupEnum::MIN:
    case GroupEnum::MAX:
        switch (inputType) {
        case ValueTypeCode::SI64:
            return int64_t(0);
        case ValueTypeCode::UI64:
            return uint64_t(0);
        case ValueTypeCode::F64:
            return double(0);
        default:
            throw std::runtime_error("unsupported aggregate column value type in groupJoin");
        }
    }
    throw std::runtime_error("unsupported aggregation function in groupJoin");
}

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <typename VTLhsTid>
void groupJoin(
    // results
    Frame *&res, DenseMatrix<VTLhsTid> *&lhsTid,
    // input frames
    const Frame *lhs, const Frame *rhs,
    // input column names
    const char *lhsOn, const char *rhsOn,
    // input aggs
    const char **rhsCols, size_t numRhsCols, mlir::daphne::GroupEnum *rhsAggFuncs, size_t numRhsAggFuncs,
    // context
    DCTX(ctx)) {
    // Find out the value types of the columns to process.
    ValueTypeCode vtcLhsOn = lhs->getColumnType(lhsOn);
    ValueTypeCode vtcRhsOn = rhs->getColumnType(rhsOn);
    if (numRhsCols != numRhsAggFuncs)
        throw std::runtime_error("number of aggregate columns and aggregate functions must match");
    if (vtcLhsOn != ValueTypeUtils::codeFor<int64_t> || vtcRhsOn != ValueTypeUtils::codeFor<int64_t>)
        throw std::runtime_error("groupJoin currently supports only int64_t join key columns");

    std::unordered_map<int64_t, JoinState> ht;
    auto argLhs = lhs->getColumn<int64_t>(lhsOn);
    auto argRhs = rhs->getColumn<int64_t>(rhsOn);

    std::vector<AggState> aggStates(numRhsCols);
    for (size_t i = 0; i < numRhsCols; i++) {
        AggState aggState;
        aggState.col = rhsCols[i];
        aggState.inputType = rhs->getColumnType(rhsCols[i]);
        aggState.func = rhsAggFuncs[i];
        aggState.label = mlir::daphne::stringifyGroupEnum(aggState.func).str() + "(" + std::string(aggState.col) + ")";

        switch (aggState.func) {
        case mlir::daphne::GroupEnum::COUNT:
            aggState.outputType = ValueTypeCode::UI64;
            break;
        case mlir::daphne::GroupEnum::SUM:
        case mlir::daphne::GroupEnum::MIN:
        case mlir::daphne::GroupEnum::MAX:
            aggState.outputType = rhs->getColumnType(rhsCols[i]);
            break;
        case mlir::daphne::GroupEnum::AVG:
            aggState.outputType = ValueTypeCode::F64;
            break;
        }
        aggState.value = initAggValue(aggState.func, aggState.inputType);
        aggStates[i] = aggState;
    }

    // ================================================================
    // Build phase
    const size_t numArgLhs = lhs->getNumRows();
    for (size_t i = 0; i < numArgLhs; i++)
        ht.emplace(argLhs->get(i, 0), JoinState(i, aggStates));
    // ================================================================

    // ================================================================
    // Probe phase
    const size_t numArgRhs = rhs->getNumRows();
    for (size_t r = 0; r < numArgRhs; r++) {
        auto it = ht.find(argRhs->get(r, 0));
        if (it == ht.end())
            continue;

        it->second.matched = true;
        for (size_t a = 0; a < it->second.aggStates.size(); a++) {
            AggState &aggState = it->second.aggStates[a];
            const bool updated = groupJoinColIf<int64_t>(aggState.inputType, aggState, rhs, r, ctx) ||
                                 groupJoinColIf<uint64_t>(aggState.inputType, aggState, rhs, r, ctx) ||
                                 groupJoinColIf<double>(aggState.inputType, aggState, rhs, r, ctx);
            if (!updated)
                throw std::runtime_error("unsupported aggregate column value type in groupJoin");
        }
    }
    // ================================================================
    // Combine into final res frame

    // res->setLabels(labels.data());
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_GROUPJOIN_H
