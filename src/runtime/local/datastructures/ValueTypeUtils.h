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

#pragma once

#include <runtime/local/datastructures/FixedSizeStringValueType.h>
#include <runtime/local/datastructures/ValueTypeCode.h>

#include <iostream>
#include <string>

#include <cinttypes>
#include <cstddef>

// Intended for use with TEMPLATE_TEST_CASE in the test cases, but fits nicely
// here where everything else value-type-related resides, as that helps to keep
// changes to the list of supported data types local.
#define ALL_VALUE_TYPES int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t, float, double

#define ALL_STRING_VALUE_TYPES std::string, FixedStr16

struct ValueTypeUtils {

    static size_t sizeOf(ValueTypeCode type);

    static void printValue(std::ostream &os, ValueTypeCode type, const void *array, size_t pos);

    template <typename ValueType> static const ValueTypeCode codeFor;

    template <typename ValueType> static const ValueType defaultValue;

    template <typename ValueType> static const std::string cppNameFor;

    template <typename ValueType> static const std::string irNameFor;

    static const std::string cppNameForCode(ValueTypeCode type);

    static const std::string irNameForCode(ValueTypeCode type);
};

template <> const ValueTypeCode ValueTypeUtils::codeFor<bool>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<int8_t>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<int32_t>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<int64_t>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<uint8_t>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<uint32_t>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<uint64_t>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<float>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<double>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<std::string>;
template <> const ValueTypeCode ValueTypeUtils::codeFor<FixedStr16>;

template <> const std::string ValueTypeUtils::cppNameFor<int8_t>;
template <> const std::string ValueTypeUtils::cppNameFor<int32_t>;
template <> const std::string ValueTypeUtils::cppNameFor<int64_t>;
template <> const std::string ValueTypeUtils::cppNameFor<uint8_t>;
template <> const std::string ValueTypeUtils::cppNameFor<uint32_t>;
template <> const std::string ValueTypeUtils::cppNameFor<uint64_t>;
template <> const std::string ValueTypeUtils::cppNameFor<float>;
template <> const std::string ValueTypeUtils::cppNameFor<double>;
template <> const std::string ValueTypeUtils::cppNameFor<bool>;
template <> const std::string ValueTypeUtils::cppNameFor<char *>;

template <> const std::string ValueTypeUtils::irNameFor<int8_t>;
template <> const std::string ValueTypeUtils::irNameFor<int32_t>;
template <> const std::string ValueTypeUtils::irNameFor<int64_t>;
template <> const std::string ValueTypeUtils::irNameFor<uint8_t>;
template <> const std::string ValueTypeUtils::irNameFor<uint32_t>;
template <> const std::string ValueTypeUtils::irNameFor<uint64_t>;
template <> const std::string ValueTypeUtils::irNameFor<float>;
template <> const std::string ValueTypeUtils::irNameFor<double>;

template <> const int8_t ValueTypeUtils::defaultValue<int8_t>;
template <> const int32_t ValueTypeUtils::defaultValue<int32_t>;
template <> const int64_t ValueTypeUtils::defaultValue<int64_t>;
template <> const uint8_t ValueTypeUtils::defaultValue<uint8_t>;
template <> const uint32_t ValueTypeUtils::defaultValue<uint32_t>;
template <> const uint64_t ValueTypeUtils::defaultValue<uint64_t>;
template <> const float ValueTypeUtils::defaultValue<float>;
template <> const double ValueTypeUtils::defaultValue<double>;
template <> const std::string ValueTypeUtils::defaultValue<std::string>;
template <> const FixedStr16 ValueTypeUtils::defaultValue<FixedStr16>;
template <> const char *ValueTypeUtils::defaultValue<const char *>;
