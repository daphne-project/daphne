/*
 * Copyright 2024 The DAPHNE Consortium
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

#include "ChunkedTensor.h"

template <typename ValueType> void ChunkedTensor<ValueType>::printValue(std::ostream &os, ValueType val) const {
    os << val;
}

// Convert to an integer to print uint8_t values as numbers
// even if they fall into the range of special ASCII characters.
template <> [[maybe_unused]] void ChunkedTensor<uint8_t>::printValue(std::ostream &os, uint8_t val) const {
    os << static_cast<uint32_t>(val);
}

template <> [[maybe_unused]] void ChunkedTensor<int8_t>::printValue(std::ostream &os, int8_t val) const {
    os << static_cast<int32_t>(val);
}

// explicitly instantiate to satisfy linker
template class ChunkedTensor<double>;
template class ChunkedTensor<float>;
template class ChunkedTensor<int>;
template class ChunkedTensor<long>;
template class ChunkedTensor<unsigned int>;
template class ChunkedTensor<unsigned long>;
#if defined(__APPLE__) && defined(__aarch64__)
template class ChunkedTensor<long long>;
template class ChunkedTensor<unsigned long long>;
#endif
