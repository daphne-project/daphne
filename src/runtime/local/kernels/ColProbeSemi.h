/*
 * Copyright 2025 The DAPHNE Consortium
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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/Column.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/HashSet.h>

#include <absl/container/flat_hash_set.h>

#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <typename DTResLHSPos, typename HS, typename VTData> struct ColProbeSemi {
    static void apply(const DTResLHSPos *&resLhsPos, const VTData *hashSet, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <typename DTResLHSPos, typename HS, typename VTData> void colBuildSemi(const VTData *hashSet, DCTX(ctx)) {
    ColProbeSemi<DTResLHSPos, HS, VTData>::apply(hashSet, ctx);
}

template <typename VTPos, typename HashFn, typename VTData>
struct ColProbeSemi<VTPos, HashSet<VTData, HashFn>, VTData> {
    static void apply(Column<VTPos> *&resLhsPos, HashSet<VTData, HashFn> *rhsHashSet, const Column<VTData> *lhsData,
                      DCTX(_)) {
        const size_t numRes = lhsData->getNumRows();
        if (resLhsPos == nullptr)
            resLhsPos = DataObjectFactory::create<Column<VTPos>>(numRes, false);

        const VTData *valuesLhsData = lhsData->getValues();
        VTPos *valuesResLhsPos = resLhsPos->getValues();
        size_t posRes = 0;
        for (size_t r = 0; r < numRes; r++)
            if (rhsHashSet->has(valuesLhsData[r]))
                valuesResLhsPos[posRes++] = r;

        resLhsPos->shrinkNumRows(posRes);
    }
};
