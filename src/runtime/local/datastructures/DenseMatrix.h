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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Matrix.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#include <cstddef>
#include <cstring>

/**
 * @brief A dense matrix implementation.
 *
 * This matrix implementation is backed by a single dense array of values. The
 * values are arranged in row-major fashion. That is, the array contains all
 * values in the first row, followed by all values in the second row, etc.
 *
 * Each instance of this class might represent a sub-matrix of another
 * `DenseMatrix`. Thus, in general, the row skip (see `getRowSkip()`) needs to
 * be added to a pointer to a particular cell in the `values` array in order to
 * obtain a pointer to the corresponding cell in the next row.
 */
template <typename ValueType> class DenseMatrix : public Matrix<ValueType> {
    // `using`, so that we do not need to prefix each occurrence of these
    // fields from the super-classes.
    using Matrix<ValueType>::numRows;
    using Matrix<ValueType>::numCols;

    const bool is_view;
    size_t rowSkip;

    // ToDo: handle this through MDO
    std::shared_ptr<ValueType[]> values{};
    size_t bufferSize;

    size_t lastAppendedRowIdx;
    size_t lastAppendedColIdx;

    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template <class DataType, typename... ArgTypes> friend DataType *DataObjectFactory::create(ArgTypes...);
    template <class DataType> friend void DataObjectFactory::destroy(const DataType *obj);

    /**
     * @brief Creates a `DenseMatrix` and allocates enough memory for the
     * specified maximum size in the `values` array.
     *
     * @param maxNumRows The maximum number of rows.
     * @param numCols The exact number of columns.
     * @param zero Whether the allocated memory of the `values` array shall be
     * initialized to zeros (`true`), or be left uninitialized (`false`).
     */
    DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, IAllocationDescriptor *allocInfo = nullptr);

    /**
     * @brief Creates a `DenseMatrix` around an existing array of values without
     * copying the data.
     *
     * @param numRows The exact number of rows.
     * @param numCols The exact number of columns.
     * @param values A `std::shared_ptr` to an existing array of values.
     */
    DenseMatrix(size_t numRows, size_t numCols, std::shared_ptr<ValueType[]> &values);

    /**
     * @brief Creates a `DenseMatrix` around a sub-matrix of another
     * `DenseMatrix` without copying the data.
     *
     * @param src The other dense matrix.
     * @param rowLowerIncl Inclusive lower bound for the range of rows to
     * extract.
     * @param rowUpperExcl Exclusive upper bound for the range of rows to
     * extract.
     * @param colLowerIncl Inclusive lower bound for the range of columns to
     * extract.
     * @param colUpperExcl Exclusive upper bound for the range of columns to
     * extract.
     */
    DenseMatrix(const DenseMatrix<ValueType> *src, int64_t rowLowerIncl, int64_t rowUpperExcl, int64_t colLowerIncl,
                int64_t colUpperExcl);
    DenseMatrix(const DenseMatrix<ValueType> *src, int64_t rowLowerIncl, int64_t rowUpperExcl)
        : DenseMatrix(src, rowLowerIncl, rowUpperExcl, 0, src->numCols){};

    /**
     * @brief Creates a `DenseMatrix` around an existing array of values without
     * copying the data.
     *
     * @param numRows The exact number of rows.
     * @param numCols The exact number of columns.
     * @param values A `std::shared_ptr` to an existing array of values.
     */
    DenseMatrix(size_t numRows, size_t numCols, const DenseMatrix<ValueType> *src);

    ~DenseMatrix() override = default;

    [[nodiscard]] size_t pos(size_t rowIdx, size_t colIdx, bool rowSkipOverride = false) const {
        if (rowIdx >= numRows)
            throw std::runtime_error("rowIdx is out of bounds");
        if (colIdx >= numCols)
            throw std::runtime_error("colIdx is out of bounds");
        return rowIdx * (rowSkipOverride ? numCols : rowSkip) + colIdx;
    }

    void fillZeroUntil(size_t rowIdx, size_t colIdx) {
        if (rowSkip == numCols || lastAppendedRowIdx == rowIdx) {
            const size_t startPosIncl = pos(lastAppendedRowIdx, lastAppendedColIdx) + 1;
            const size_t endPosExcl = pos(rowIdx, colIdx);

            if (startPosIncl < endPosExcl)
                std::fill(values.get() + startPosIncl, values.get() + endPosExcl,
                          ValueTypeUtils::defaultValue<ValueType>);
        } else {
            auto v = values.get() + lastAppendedRowIdx * rowSkip;
            std::fill(v + lastAppendedColIdx + 1, v + numCols, ValueTypeUtils::defaultValue<ValueType>);

            v += rowSkip;

            for (size_t r = lastAppendedRowIdx + 1; r < rowIdx; r++) {
                std::fill(v, v + numCols, ValueTypeUtils::defaultValue<ValueType>);
                v += rowSkip;
            }
            if (colIdx)
                std::fill(v, v + colIdx - 1, ValueTypeUtils::defaultValue<ValueType>);
        }
    }

    void printValue(std::ostream &os, ValueType val) const;

    void alloc_shared_values(std::shared_ptr<ValueType[]> src = nullptr, size_t offset = 0);

    /**
     * @brief The getValuesInternal method fetches a pointer to an allocation of
     * values. Optionally a sub range can be specified.
     *
     * This method is called by the public getValues() methods either in const
     * or non-const fashion. The const version returns a data pointer that is
     * meant to be read-only. Read only access updates the list of up-to-date
     * allocations (the latest_versions "list"). This way several copies of the
     * data in various allocations can be kept without invalidating their copy.
     * If the read write version of getValues() is used, the latest_versions
     * list is cleared because a write to an allocation is assumed, which
     * renders the other allocations of the same data out of sync.
     *
     * @param alloc_desc An instance of an IAllocationDescriptor derived class
     * that is used to specify what type of allocation is requested. If no
     * allocation descriptor is provided, a host allocation (plain main memory)
     * is assumed by default.
     *
     * @param range An optional range describing which rows and columns of a
     * matrix-like structure are requested. By default this is null and means
     * all rows and columns.
     * @return A tuple of three values is returned:
     *         1: bool - is the returend allocation in the latest_versions list
     *         2: size_t - the ID of the data placement (a structure relating an
     * allocation to a range) 3: ValueType* - the pointer to the actual data
     */

    auto getValuesInternal(const IAllocationDescriptor *alloc_desc = nullptr,
                           const Range *range = nullptr) -> std::tuple<bool, size_t, ValueType *>;

    [[nodiscard]] size_t offset() const { return this->row_offset * rowSkip + this->col_offset; }

    ValueType *startAddress() const { return isPartialBuffer() ? values.get() + offset() : values.get(); }

  public:
    template <typename NewValueType> using WithValueType = DenseMatrix<NewValueType>;

    static std::string getName() { return "DenseMatrix"; }

    [[nodiscard]] bool isPartialBuffer() const {
        return bufferSize != this->getNumRows() * this->getRowSkip() * sizeof(ValueType);
    }

    void shrinkNumRows(size_t numRows) {
        if (numRows > this->numRows)
            throw std::runtime_error("DenseMatrix (shrinkNumRows): number of "
                                     "rows can only be shrunk");
        // TODO Here we could reduce the allocated size of the values array.
        this->numRows = numRows;
    }

    [[nodiscard]] size_t getRowSkip() const { return rowSkip; }

    [[nodiscard]] bool isView() const { return is_view; }

    /**
     * @brief Fetch a pointer to the data held by this structure meant for
     * read-only access.
     *
     * A difference is made between read-only and read-write access because with
     * read-only access, the data can be cached in several memory spaces at the
     * same time.
     *
     * @param alloc_desc An allocation descriptor describing which type of
     * memory is requested (e.g. main memory in the current system, memory in an
     * accelerator card or memory in another host)
     *
     * @param range A Range object describing optionally requesting a sub range
     * of a data structure.
     * @return A pointer to the data in the requested memory space
     */
    const ValueType *getValues(const IAllocationDescriptor *alloc_desc = nullptr, const Range *range = nullptr) const {
        auto [isLatest, id, ptr] = const_cast<DenseMatrix<ValueType> *>(this)->getValuesInternal(alloc_desc, range);
        if (!isLatest)
            this->mdo->addLatest(id);
        return ptr;
    }

    /**
     * @brief Fetch a pointer to the data held by this structure meant for
     * read-write access.
     *
     * A difference is made between read-only and read-write access. With
     * read-write access, all copies in various memory spaces will be
     * invalidated because data is assumed to change.
     *
     * @param alloc_desc An allocation descriptor describing which type of
     * memory is requested (e.g. main memory in the current system, memory in an
     * accelerator card or memory in another host)
     *
     * @param range A Range object describing optionally requesting a sub range
     * of a data structure.
     * @return A pointer to the data in the requested memory space
     */
    ValueType *getValues(IAllocationDescriptor *alloc_desc = nullptr, const Range *range = nullptr) {
        auto [isLatest, id, ptr] = const_cast<DenseMatrix<ValueType> *>(this)->getValuesInternal(alloc_desc, range);
        if (!isLatest)
            this->mdo->setLatest(id);
        return ptr;
    }

    std::shared_ptr<ValueType[]> getValuesSharedPtr() const { return values; }

    ValueType get(size_t rowIdx, size_t colIdx) const override {
        return getValues()[pos(rowIdx, colIdx, isPartialBuffer())];
    }

    void set(size_t rowIdx, size_t colIdx, ValueType value) override {
        auto vals = getValues();
        vals[pos(rowIdx, colIdx)] = value;
    }

    void prepareAppend() override {
        // The matrix might be empty.
        if (numRows != 0 && numCols != 0)
            values.get()[0] = ValueType(ValueTypeUtils::defaultValue<ValueType>);
        lastAppendedRowIdx = 0;
        lastAppendedColIdx = 0;
    }

    void append(size_t rowIdx, size_t colIdx, ValueType value) override {
        // Set all cells since the last one that was appended to zero.
        fillZeroUntil(rowIdx, colIdx);
        // Set the specified cell.
        values.get()[pos(rowIdx, colIdx)] = value;
        // Update append state.
        lastAppendedRowIdx = rowIdx;
        lastAppendedColIdx = colIdx;
    }

    void finishAppend() override {
        // The matrix might be empty.
        if ((numRows != 0 && numCols != 0) &&
            ((lastAppendedRowIdx + 1 < numRows) || (lastAppendedColIdx + 1 < numCols)))
            append(numRows - 1, numCols - 1, ValueType(ValueTypeUtils::defaultValue<ValueType>));
    }

    void print(std::ostream &os) const override {
        os << "DenseMatrix(" << numRows << 'x' << numCols << ", " << ValueTypeUtils::cppNameFor<ValueType> << ')'
           << std::endl;

        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                printValue(os, get(r, c));
                if (c < numCols - 1)
                    os << ' ';
            }
            os << std::endl;
        }
    }

    DenseMatrix<ValueType> *sliceRow(size_t rl, size_t ru) const override { return slice(rl, ru, 0, numCols); }

    DenseMatrix<ValueType> *sliceCol(size_t cl, size_t cu) const override { return slice(0, numRows, cl, cu); }

    DenseMatrix<ValueType> *slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        return DataObjectFactory::create<DenseMatrix<ValueType>>(this, rl, ru, cl, cu);
    }

    [[nodiscard]] size_t getBufferSize() const { return bufferSize; }

    bool operator==(const DenseMatrix<ValueType> &rhs) const {
        // Note that we do not use the generic `get` interface to matrices here
        // since this operator is meant to be used for writing tests for,
        // besides others, those generic interfaces.

        if (this == &rhs)
            return true;

        const size_t numRows = this->getNumRows();
        const size_t numCols = this->getNumCols();

        if (numRows != rhs.getNumRows() || numCols != rhs.getNumCols())
            return false;

        const ValueType *valuesLhs = this->getValues();
        const ValueType *valuesRhs = rhs.getValues();

        const size_t rowSkipLhs = this->getRowSkip();
        const size_t rowSkipRhs = rhs.getRowSkip();

        if (valuesLhs == valuesRhs && rowSkipLhs == rowSkipRhs)
            return true;

        for (size_t r = 0; r < numRows; ++r) {
            for (size_t c = 0; c < numCols; ++c) {
                if (*(valuesLhs + c) != *(valuesRhs + c)){
                    std::cout << "Mismatch at (" << r << ", " << c << "): "
                              << *(valuesLhs + c) << " != " << *(valuesRhs + c) << std::endl;
                    return false;
                }
            }
            valuesLhs += rowSkipLhs;
            valuesRhs += rowSkipRhs;
        }
        return true;
    }

    size_t serialize(std::vector<char> &buf) const override;
};

template <typename ValueType> std::ostream &operator<<(std::ostream &os, const DenseMatrix<ValueType> &obj) {
    obj.print(os);
    return os;
}

/*
    Helper struct for DenseMatrix with strings, represents a char buffer.
    Needs to keep numCells from original DenseMatrix to manage modifications
   from views.
*/
struct CharBuf {
    char *strings;
    char *currentTop;

    size_t capacity;
    size_t numCells;

    CharBuf(size_t capacity_, size_t numCells_) : capacity(capacity_), numCells(numCells_) {
        strings = new char[capacity];
        currentTop = strings;
    }
    ~CharBuf() { delete[] strings; }

    void expandStringBuffer(const size_t toFit, const char **vals, size_t numRows, size_t rowSkip,
                            const size_t valsSize) {
        size_t strBufSize = getSize();

        size_t largerStrCapacity = (capacity * 2) > toFit ? (capacity * 2) : toFit;
        char *largerStrings = new char[largerStrCapacity];
        memcpy(largerStrings, strings, strBufSize);

        auto start = vals[0];
        const size_t numCols = numCells / numRows;
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                size_t offset = vals[c] - start;
                vals[c] = &largerStrings[offset];
            }
            vals += rowSkip;
        }

        delete[] strings;
        strings = largerStrings;
        capacity = largerStrCapacity;
        currentTop = &largerStrings[strBufSize];
    }

    size_t getSize() { return currentTop - strings; }
};

template <> class DenseMatrix<const char *> : public Matrix<const char *> {
    // `using`, so that we do not need to prefix each occurrence of these
    // fields from the super-classes.
    using Matrix<const char *>::numRows;
    using Matrix<const char *>::numCols;

    size_t rowSkip;
    std::shared_ptr<const char *[]> values{};
    std::shared_ptr<CharBuf> strBuf;

    std::shared_ptr<const char *> cuda_ptr{};
    uint32_t deleted = 0;

    size_t lastAppendedRowIdx;
    size_t lastAppendedColIdx;

    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template <class DataType, typename... ArgTypes> friend DataType *DataObjectFactory::create(ArgTypes...);
    template <class DataType> friend void DataObjectFactory::destroy(const DataType *obj);

    DenseMatrix(size_t maxNumRows, size_t numCols, bool zero, size_t strBufCapacity = 1024,
                ALLOCATION_TYPE type = ALLOCATION_TYPE::HOST);

    DenseMatrix(size_t numRows, size_t numCols, std::shared_ptr<const char *[]> &strings, size_t strBufCapacity = 1024,
                std::shared_ptr<const char *> cuda_ptr_ = nullptr);

    DenseMatrix(const DenseMatrix<const char *> *src, int64_t rowLowerIncl, int64_t rowUpperExcl, int64_t colLowerIncl,
                int64_t colUpperExcl);

    ~DenseMatrix() override = default;

    /**
     * @brief Calculates the position within linear memory given row/col indices
     *
     * @param rowIdx
     * @param colIdx
     * @param rowSkipOverride use numCols instead of rowSkip if get() is used on
     * a partial buffer
     * @return linearized position
     */
    [[nodiscard]] size_t pos(size_t rowIdx, size_t colIdx, bool rowSkipOverride = false) const {
        if (rowIdx >= numRows)
            throw std::runtime_error("rowIdx is out of bounds");
        if (colIdx >= numCols)
            throw std::runtime_error("colIdx is out of bounds");
        return rowIdx * (rowSkipOverride ? numCols : rowSkip) + colIdx;
    }

    void appendZerosRange(const char **valsStartPos, const size_t length) {
        memset(strBuf->currentTop, '\0', length * sizeof(char));
        for (size_t val = 0; val < length; val++) {
            valsStartPos[val] = &strBuf->currentTop[val];
        }
        strBuf->currentTop += length;
    }

    void fillZeroUntil(size_t rowIdx, size_t colIdx) {
        auto vals = values.get();
        if (rowSkip == numCols || lastAppendedRowIdx == rowIdx) {
            const size_t startPosIncl = pos(lastAppendedRowIdx, lastAppendedColIdx) + 1;
            const size_t endPosExcl = pos(rowIdx, colIdx);
            if (startPosIncl < endPosExcl) {
                appendZerosRange(&vals[startPosIncl], endPosExcl - startPosIncl);
            }
        } else {
            auto v = vals + lastAppendedRowIdx * rowSkip;
            appendZerosRange(&v[lastAppendedColIdx + 1], numCols - lastAppendedColIdx - 1);
            v += rowSkip;

            for (size_t r = lastAppendedRowIdx + 1; r < rowIdx; r++) {
                appendZerosRange(v, numCols);
                v += rowSkip;
            }

            if (colIdx)
                appendZerosRange(v, colIdx - 1);
        }
    }

    void printValue(std::ostream &os, const char *val) const;

    void alloc_shared_values(std::shared_ptr<const char *[]> src = nullptr, size_t offset = 0);

    void alloc_shared_strings(std::shared_ptr<CharBuf> src = nullptr, size_t strBufferCapacity = 1024);

    void alloc_shared_cuda_buffer(std::shared_ptr<const char *> src = nullptr, size_t offset = 0);

  public:
    void shrinkNumRows(size_t numRows) {
        if (numRows > this->numRows)
            throw std::runtime_error("DenseMatrix (shrinkNumRows): number of "
                                     "rows can only be shrunk");
        // TODO Here we could reduce the allocated size of the values array.
        this->numRows = numRows;
    }

    [[nodiscard]] size_t getRowSkip() const { return rowSkip; }

    const char **getValues() const {
        if (!values)
            const_cast<DenseMatrix *>(this)->alloc_shared_values();
        return values.get();
    }

    const char **getValues() {
        if (!values)
            alloc_shared_values();
        return values.get();
    }

    std::shared_ptr<CharBuf> getStrBufSharedPtr() const { return strBuf; }

    CharBuf *getStrBuf() const {
        if (!strBuf)
            const_cast<DenseMatrix *>(this)->alloc_shared_strings();
        return strBuf.get();
    }

    CharBuf *getStrBuf() {
        if (!strBuf)
            alloc_shared_strings();
        return strBuf.get();
    }

    std::shared_ptr<const char *[]> getValuesSharedPtr() const { return values; }

    const char *get(size_t rowIdx, size_t colIdx) const override { return getValues()[pos(rowIdx, colIdx, false)]; }

    void set(size_t rowIdx, size_t colIdx, const char *value) override {
        auto vals = getValues();
        size_t currentPos = pos(rowIdx, colIdx);
        auto currentVal = vals[currentPos];
        size_t currentSize = strBuf.get()->getSize();
        int32_t diff = strlen(value) - strlen(currentVal);

        if (currentSize + diff > strBuf->capacity) {
            strBuf.get()->expandStringBuffer(currentSize + diff, vals, numRows, rowSkip, getNumItems());
            currentVal = vals[currentPos];
        }

        if (diff && (currentPos + 1 < getStrBuf()->numCells)) {
            const char *from = values[currentPos + 1];
            const char *to = from + diff;
            size_t length = strBuf->currentTop - from;
            memmove(const_cast<char *>(to), from, length);
        }
        memcpy(const_cast<char *>(currentVal), value, strlen(value) + 1);

        if (diff) {
            for (size_t offset = currentPos + 1; offset < getStrBuf()->numCells; offset++)
                vals[offset] += diff;
        }

        strBuf->currentTop += diff;
    }

    void prepareAppend() override {
        // the matrix might be empty
        if (numRows != 0 && numCols != 0)
            values.get()[0] = "\0";
        lastAppendedRowIdx = 0;
        lastAppendedColIdx = 0;
        strBuf->currentTop = strBuf.get()->strings;
    }

    void append(size_t rowIdx, size_t colIdx, const char *value) override {
        // Set all cells since the last one that was appended to zero.
        fillZeroUntil(rowIdx, colIdx);
        auto vals = getValues();
        // Set the specified cell.
        size_t length = strlen(value) + 1;
        size_t currentSize = strBuf.get()->getSize();

        if (currentSize + length > strBuf->capacity)
            strBuf.get()->expandStringBuffer(currentSize + length, vals, numRows, rowSkip, getNumRows() * getNumCols());

        memcpy(strBuf->currentTop, value, length);
        vals[pos(rowIdx, colIdx)] = strBuf->currentTop;

        strBuf->currentTop += length;
        // Update append state.
        lastAppendedRowIdx = rowIdx;
        lastAppendedColIdx = colIdx;
    }

    void finishAppend() override {
        // numRows/numCols are unsigned and can underflow
        if ((numRows != 0 && numCols != 0) &&
            ((lastAppendedRowIdx + 1 < numRows) || (lastAppendedColIdx + 1 < numCols)))
            append(numRows - 1, numCols - 1, "\0");
    }

    void print(std::ostream &os) const override {
        os << "DenseMatrix(" << numRows << 'x' << numCols << ", " << ValueTypeUtils::cppNameFor<const char *> << ')'
           << std::endl;
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                printValue(os, get(r, c));
                if (c < numCols - 1)
                    os << ' ';
            }
            os << std::endl;
        }
    }

    DenseMatrix<const char *> *sliceRow(size_t rl, size_t ru) const override { return slice(rl, ru, 0, numCols); }

    DenseMatrix<const char *> *sliceCol(size_t cl, size_t cu) const override { return slice(0, numRows, cl, cu); }

    DenseMatrix<const char *> *slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        return DataObjectFactory::create<DenseMatrix<const char *>>(this, rl, ru, cl, cu);
    }

    float printBufferSize() const { return static_cast<float>(numRows * numCols) / (1048576); }

    bool operator==(const DenseMatrix<const char *> &M) const {
        if (getNumRows() == 0 || getNumCols() == 0 || !strBuf || !values)
            throw std::runtime_error("DenseMatrix (operator==): invalid "
                                     "matrix. DenseMatrix must not be empty");
        for (size_t r = 0; r < getNumRows(); r++)
            for (size_t c = 0; c < getNumCols(); c++)
                if (strcmp(M.getValues()[M.pos(r, c)], values.get()[pos(r, c)]))
                    return false;
        return true;
    }

    size_t serialize(std::vector<char> &buf) const override { throw std::runtime_error("Not implemented"); }
};
