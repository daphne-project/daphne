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

#include <runtime/local/io/DaphneSerializer.h>

#include "CSRMatrix.h"

template <typename ValueType> size_t CSRMatrix<ValueType>::serialize(std::vector<char> &buf) const {
    return DaphneSerializer<CSRMatrix<ValueType>>::serialize(this, buf);
}

// explicitly instantiate to satisfy linker
template class CSRMatrix<double>;
template class CSRMatrix<float>;
template class CSRMatrix<int>;
template class CSRMatrix<long>;
template class CSRMatrix<signed char>;
template class CSRMatrix<unsigned char>;
template class CSRMatrix<unsigned int>;
template class CSRMatrix<unsigned long>;
#if defined(__APPLE__) && defined(__aarch64__)
template class CSRMatrix<long long>;
template class CSRMatrix<unsigned long long>;
#endif
