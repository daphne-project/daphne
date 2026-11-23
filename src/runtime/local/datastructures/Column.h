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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <cstddef>

/**
 * @brief A data structure that represents a single column in columnar query processing, all elements have the same
 * value type and are stored contiguously in memory.
 */
template <typename ValueType> class Column : public Structure {
    // `using`, so that we do not need to prefix each occurrence of these fields from the super-classes.
    using Structure::numCols;
    using Structure::numRows;

    std::shared_ptr<ValueType[]> values{};

    // Grant DataObjectFactory access to the private constructors and destructors.
    template <class DataType, typename... ArgTypes> friend DataType *DataObjectFactory::create(ArgTypes...);
    template <class DataType> friend void DataObjectFactory::destroy(const DataType *obj);

    Column(size_t numRows, bool zero)
        : Structure(numRows, 1), values(std::shared_ptr<ValueType[]>(new ValueType[numRows])) {
        if (zero)
            std::fill(values.get(), values.get() + numRows, ValueTypeUtils::defaultValue<ValueType>);
    }

    Column(size_t numRows, std::shared_ptr<ValueType[]> &values) : Structure(numRows, 1), values(values) {}

    ~Column() override = default;

    void printValue(std::ostream &os, ValueType val) const {
        if constexpr (std::is_same<ValueType, int8_t>::value || std::is_same<ValueType, uint8_t>::value)
            os << static_cast<uint32_t>(val);
        else
            os << val;
    }

  public:
    template <typename NewValueType> using WithValueType = Column<NewValueType>;

    /**
     * @brief The common type of all values in this column.
     */
    using VT = ValueType;

    std::optional<double> lowerBound;
    std::optional<double> upperBound;

    static std::string getName() { return "Column"; }

    size_t getNumDims() const override { return 1; }

    size_t getNumItems() const override { return this->numRows; }

    void print(std::ostream &os) const override {
        os << "Column(" << numRows << ", " << ValueTypeUtils::cppNameFor<ValueType> << ')' << std::endl;

        for (size_t r = 0; r < numRows; r++) {
            printValue(os, values.get()[r]);
            os << std::endl;
        }
    }

    Column<ValueType> *sliceRow(size_t rl, size_t ru) const override {
        throw std::runtime_error("slicing has not been implemented for Column yet");
    }

    Column<ValueType> *sliceCol(size_t cl, size_t cu) const override {
        throw std::runtime_error("slicing has not been implemented for Column yet");
    }

    Column<ValueType> *slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        throw std::runtime_error("slicing has not been implemented for Column yet");
    }

    size_t serialize(std::vector<char> &buf) const override {
        throw std::runtime_error("serialization has not been implemented for Column yet");
    }

    void shrinkNumRows(size_t numRows) {
        if (numRows > this->numRows)
            throw std::runtime_error("Column (shrinkNumRows): number of rows can only be shrunk");
        // TODO Here we could reduce the allocated size of the values array.
        this->numRows = numRows;
    }

    const ValueType *getValues() const { return values.get(); }

    ValueType *getValues() { return values.get(); }

    std::shared_ptr<ValueType[]> getValuesSharedPtr() const { return values; }

    bool operator==(const Column<ValueType> &rhs) const {
        // Note that we do not use the generic `get` interface here since this operator is meant to be used for writing
        // tests for, besides others, those generic interfaces.

        if (this == &rhs)
            return true;

        const size_t numRows = this->getNumRows();

        if (numRows != rhs.getNumRows())
            return false;

        const ValueType *valuesLhs = this->getValues();
        const ValueType *valuesRhs = rhs.getValues();

        if (valuesLhs == valuesRhs)
            return true;

        for (size_t r = 0; r < numRows; r++)
            if (valuesLhs[r] != valuesRhs[r])
                return false;

        return true;
    }
};

template <typename ValueType> std::ostream &operator<<(std::ostream &os, const Column<ValueType> &obj) {
    obj.print(os);
    return os;
}