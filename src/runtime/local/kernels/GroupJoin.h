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

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cstddef>
#include <cstdint>

namespace group_join {

inline ValueTypeCode aggResultType(mlir::daphne::GroupEnum op, ValueTypeCode inputType) {
    using mlir::daphne::GroupEnum;
    if (op == GroupEnum::COUNT)
        return ValueTypeCode::UI64;
    if (op == GroupEnum::AVG)
        return ValueTypeCode::F64;
    return inputType;
}

// aggregateColumn aggregates one numeric rhs column into its output column.
template <typename VT>
void aggTypedColumn(mlir::daphne::GroupEnum op, const VT *rhsAggColumn, const std::vector<bool> &rhsMatched,
                    const std::vector<size_t> &rhsGroup, size_t numGroups, const std::vector<bool> &lhsMatched,
                    const std::vector<size_t> &groupToOut, void *outColumn) {
    using mlir::daphne::GroupEnum;
    switch (op) {
    case GroupEnum::SUM: {
        std::vector<VT> acc(numGroups, VT{0});
        for (size_t r = 0; r < rhsGroup.size(); r++)
            if (rhsMatched[r])
                acc[rhsGroup[r]] += rhsAggColumn[r];
        auto *out = static_cast<VT *>(outColumn);
        for (size_t g = 0; g < numGroups; g++)
            if (lhsMatched[g])
                out[groupToOut[g]] = acc[g];
        break;
    }
    case GroupEnum::MIN: {
        std::vector<VT> acc(numGroups, std::numeric_limits<VT>::max());
        for (size_t r = 0; r < rhsGroup.size(); r++)
            if (rhsMatched[r])
                acc[rhsGroup[r]] = std::min(acc[rhsGroup[r]], rhsAggColumn[r]);
        auto *out = static_cast<VT *>(outColumn);
        for (size_t g = 0; g < numGroups; g++)
            if (lhsMatched[g])
                out[groupToOut[g]] = acc[g];
        break;
    }
    case GroupEnum::MAX: {
        std::vector<VT> acc(numGroups, std::numeric_limits<VT>::lowest());
        for (size_t r = 0; r < rhsGroup.size(); r++)
            if (rhsMatched[r])
                acc[rhsGroup[r]] = std::max(acc[rhsGroup[r]], rhsAggColumn[r]);
        auto *out = static_cast<VT *>(outColumn);
        for (size_t g = 0; g < numGroups; g++)
            if (lhsMatched[g])
                out[groupToOut[g]] = acc[g];
        break;
    }
    case GroupEnum::AVG: {
        std::vector<double> sum(numGroups, 0.0);
        std::vector<uint64_t> cnt(numGroups, 0);
        for (size_t r = 0; r < rhsGroup.size(); r++)
            if (rhsMatched[r]) {
                sum[rhsGroup[r]] += static_cast<double>(rhsAggColumn[r]);
                ++cnt[rhsGroup[r]];
            }
        auto *out = static_cast<double *>(outColumn);
        for (size_t g = 0; g < numGroups; g++)
            if (lhsMatched[g])
                out[groupToOut[g]] = sum[g] / static_cast<double>(cnt[g]);
        break;
    }
    default:
        throw std::runtime_error("unsupported aggregation function");
    }
}

inline void aggregateColumn(mlir::daphne::GroupEnum op, ValueTypeCode inputType, const void *inRaw,
                            const std::vector<bool> &rhsMatched, const std::vector<size_t> &rhsGroup, size_t numGroups,
                            const std::vector<bool> &lhsMatched, const std::vector<size_t> &groupToOut,
                            void *outColumn) {
    switch (inputType) {
    case ValueTypeCode::SI64:
        aggTypedColumn<int64_t>(op, static_cast<const int64_t *>(inRaw), rhsMatched, rhsGroup, numGroups, lhsMatched,
                                groupToOut, outColumn);
        break;
    case ValueTypeCode::UI64:
        aggTypedColumn<uint64_t>(op, static_cast<const uint64_t *>(inRaw), rhsMatched, rhsGroup, numGroups, lhsMatched,
                                 groupToOut, outColumn);
        break;
    case ValueTypeCode::F64:
        aggTypedColumn<double>(op, static_cast<const double *>(inRaw), rhsMatched, rhsGroup, numGroups, lhsMatched,
                               groupToOut, outColumn);
        break;
    default:
        throw std::runtime_error("unsupported agg column type");
    }
}

}

template <typename VTLhsTid>
void groupJoin(Frame *&res, DenseMatrix<VTLhsTid> *&lhsTid, const Frame *lhs, const Frame *rhs, const char *lhsOn,
               const char *rhsOn, const char **rhsCols, size_t numRhsCols, mlir::daphne::GroupEnum *rhsAggFuncs,
               size_t numRhsAggFuncs, DCTX(ctx)) {
    if (numRhsCols != numRhsAggFuncs)
        throw std::runtime_error("number of aggregate columns and aggregate functions must match");

    // Find join keys
    // TODO: currently, only int64_t is supported as join key
    const size_t lhsKeyIdx = lhs->getColumnIdx(lhsOn);
    const size_t rhsKeyIdx = rhs->getColumnIdx(rhsOn);
    if (lhs->getColumnType(lhsKeyIdx) != ValueTypeUtils::codeFor<int64_t> ||
        rhs->getColumnType(rhsKeyIdx) != ValueTypeUtils::codeFor<int64_t>)
        throw std::runtime_error("groupJoin currently supports only int64_t join key columns");

    auto *lhsKeys = static_cast<const int64_t *>(lhs->getColumnRaw(lhsKeyIdx));
    auto *rhsKeys = static_cast<const int64_t *>(rhs->getColumnRaw(rhsKeyIdx));
    size_t numLhsRows = lhs->getNumRows();
    size_t numRhsRows = rhs->getNumRows();

    // Map each unique lhs key to its row index (= group id).
    std::unordered_map<int64_t, size_t> keyToGroup;
    keyToGroup.reserve(numLhsRows);
    for (size_t i = 0; i < numLhsRows; i++) {
        auto [_, exists] = keyToGroup.emplace(lhsKeys[i], i);
        if (!exists) {
            throw std::runtime_error("lhs join key need to be unique");
        }
    }

    // Assign each matched rhs row to a group and mark matched groups.
    std::vector<bool> rhsMatched(numRhsRows, false);
    std::vector<size_t> rhsGroup(numRhsRows);
    std::vector<bool> lhsMatched(numLhsRows, false);
    size_t numMatchedGroups = 0;
    for (size_t r = 0; r < numRhsRows; r++) {
        auto it = keyToGroup.find(rhsKeys[r]);
        if (it == keyToGroup.end())
            continue;
        rhsMatched[r] = true;
        rhsGroup[r] = it->second;
        if (!lhsMatched[it->second]) {
            lhsMatched[it->second] = true;
            numMatchedGroups++;
        }
    }

    // Result schema: key column + one column per aggregate.
    std::vector<ValueTypeCode> schema;
    std::vector<std::string> labels;
    schema.reserve(numRhsCols + 1);
    labels.reserve(numRhsCols + 1);
    schema.push_back(lhs->getColumnType(lhsKeyIdx));
    labels.emplace_back(lhsOn);

    std::vector<ValueTypeCode> aggInputType(numRhsCols, ValueTypeCode::INVALID);
    for (size_t c = 0; c < numRhsCols; c++) {
        mlir::daphne::GroupEnum op = rhsAggFuncs[c];
        std::string colLabel = rhsCols[c];
        // guard from SUM(*), MIN(*), MAX(*)
        if (op != mlir::daphne::GroupEnum::COUNT) {
            if (colLabel.empty())
                throw std::runtime_error("only COUNT() may use '*' as aggregate column in groupJoin");
            aggInputType[c] = rhs->getColumnType(rhs->getColumnIdx(colLabel.c_str()));
        }
        schema.push_back(group_join::aggResultType(op, aggInputType[c]));
        labels.push_back(mlir::daphne::stringifyGroupEnum(op).str() + "(" + colLabel + ")");
    }

    // Allocate outputs.
    res = DataObjectFactory::create<Frame>(numMatchedGroups, schema.size(), schema.data(), labels.data(), false);
    lhsTid = DataObjectFactory::create<DenseMatrix<VTLhsTid>>(numMatchedGroups, 1, false);

    // Fill lhs matched records and first column of res frame
    std::vector<size_t> groupToOut(numLhsRows);
    auto *resKeys = static_cast<int64_t *>(res->getColumnRaw(0));
    VTLhsTid *tid = lhsTid->getValues();
    size_t outRow = 0;
    for (size_t g = 0; g < numLhsRows; g++) {
        if (!lhsMatched[g])
            continue;
        groupToOut[g] = outRow;
        resKeys[outRow] = lhsKeys[g];
        tid[outRow] = static_cast<VTLhsTid>(g);
        outRow++;
    }

    // One pass per aggregate column.
    for (size_t c = 0; c < numRhsCols; c++) {
        const mlir::daphne::GroupEnum op = rhsAggFuncs[c];
        auto outColumn = res->getColumnRaw(c + 1);
        // handle COUNT explicitly
        if (op == mlir::daphne::GroupEnum::COUNT) {
            std::vector<uint64_t> cnt(numLhsRows, 0);
            for (size_t r = 0; r < numRhsRows; r++)
                if (rhsMatched[r])
                    ++cnt[rhsGroup[r]];
            auto *out = static_cast<uint64_t *>(outColumn);
            for (size_t g = 0; g < numLhsRows; g++)
                if (lhsMatched[g])
                    out[groupToOut[g]] = cnt[g];
            continue;
        }
        group_join::aggregateColumn(op, aggInputType[c], rhs->getColumnRaw(rhs->getColumnIdx(rhsCols[c])), rhsMatched,
                                    rhsGroup, numLhsRows, lhsMatched, groupToOut, outColumn);
    }
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_GROUPJOIN_H
