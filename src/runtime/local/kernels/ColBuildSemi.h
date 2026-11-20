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

template <typename HS, typename VTData> struct ColBuildSemi {
    static void apply(const VTData *hashSet, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <typename HS, typename VTData> void colBuildSemi(const VTData *hashSet, DCTX(ctx)) {
    ColBuildSemi<HS, VTData>::apply(hashSet, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Column <- Column, Column
// ----------------------------------------------------------------------------

template <typename HashFn, typename VTData> struct ColBuildSemi<HashSet<VTData, HashFn>, VTData> {
    static void apply(HashSet<VTData, HashFn> *&resHashSet, const Column<VTData> *rhsData, DCTX(_)) {
        const size_t numRhsData = rhsData->getNumRows();

        if (resHashSet == nullptr)
            resHashSet = DataObjectFactory::create<HashSet<VTData, HashFn>>(numRhsData);

        // Build phase.
        const VTData *valuesRhsData = rhsData->getValues();
        for (size_t r = 0; r < numRhsData; r++)
            resHashSet->insert(valuesRhsData[r]);
    }
};
