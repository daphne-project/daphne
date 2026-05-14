/*
 *  Copyright 2023 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <mlir/IR/Types.h>

#include <ostream>
#include <vector>

/**
 * @brief Prints the name of the given MLIR type to the given stream.
 *
 * @param os The stream to print to.
 * @param t The MLIR type to print.
 */
std::ostream &operator<<(std::ostream &os, mlir::Type t);

/**
 * @brief Prints the names of the given MLIR types to the given stream, separated by comma and space.
 *
 * @param os The stream to print to.
 * @param ts The MLIR types to print.
 */
std::ostream &operator<<(std::ostream &os, std::vector<mlir::Type> ts);