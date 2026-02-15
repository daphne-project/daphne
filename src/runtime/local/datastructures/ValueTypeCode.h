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

#ifndef SRC_RUNTIME_LOCAL_DATASTRUCTURES_VALUETYPECODE_H
#define SRC_RUNTIME_LOCAL_DATASTRUCTURES_VALUETYPECODE_H

#include <cinttypes>

/**
 * @brief A run-time representation for value types.
 *
 * Each of these represents one value type from DaphneIR and the underlying
 * C++ type to use. A `ValueTypeCode` is meant to be used in situations when
 * the value type cannot be known at compile-time.
 */
enum class ValueTypeCode : uint8_t {
    SI8,
    SI32,
    SI64, // signed integers (intX_t)
    UI8,
    UI32,
    UI64, // unsigned integers (uintx_t)
    F32,
    F64,        // floating point (float, double)
    STR,        // std::string
    FIXEDSTR16,
    BOOL, // fixed-size string (length 16)
    INVALID,   // only for JSON enum conversion
    // TODO Support bool as well, but poses some challenges (e.g. sizeof).
    //    UI1 // boolean (bool)
};

#endif // SRC_RUNTIME_LOCAL_DATASTRUCTURES_VALUETYPECODE_H