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

#pragma once

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <vector>

#include <cstddef>

/**
 * @brief An ordered sequence of homogeneous-typed elements of any data/value
 * type.
 *
 * Most importantly, a list can store data objects (matrices/frames).
 */
template <typename DataType> class List : public Structure {
    /**
     * @brief The elements of this list.
     */
    std::vector<const DataType *> elements;

    // Grant DataObjectFactory access to the private constructors and
    // destructors.
    template <class DataType_, typename... ArgTypes> friend DataType_ *DataObjectFactory::create(ArgTypes...);
    template <class DataType_> friend void DataObjectFactory::destroy(const DataType_ *obj);

    /**
     * @brief Creates a new empty list.
     */
    List()
        : Structure(0, 1){
              // nothing to do
          };

    /**
     * @brief Creates a new list containing the same elements as the given list.
     *
     * @param other The other list.
     */
    List(const List<DataType> *other) : Structure(other->length(), 1) {
        for (const DT *elem : other->elements) {
            // We must increase the reference counter of each element we put
            // into this list to prevent the element from being freed as long as
            // this list exists.
            elem->increaseRefCounter();
            this->elements.push_back(elem);
        }
    }

  public:
    virtual ~List() {
        // Decrease reference counters of each element by 1.
        // If the reference counter becomes 0, destroy the element.
        for (const DataType *element : elements)
            DataObjectFactory::destroy(element);
    };

    /**
     * @brief The common type of all elements in this list.
     */
    using DT = DataType;

    size_t getNumDims() const override { return 1; }

    size_t getNumItems() const override { return this->numRows; }

    void print(std::ostream &os) const override {
        os << "List(" << elements.size() << ", " << DataType::getName() << ", "
           << ValueTypeUtils::cppNameFor<typename DataType::VT> << ')' << std::endl;
        for (size_t i = 0; i < elements.size(); i++)
            elements[i]->print(os);
    }

    Structure *sliceRow(size_t rl, size_t ru) const override {
        throw std::runtime_error("sliceRow is not supported for List yet");
    }

    Structure *sliceCol(size_t cl, size_t cu) const override {
        throw std::runtime_error("sliceCol is not supported for List yet");
    }

    Structure *slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        throw std::runtime_error("slice is not supported for List yet");
    }

    size_t serialize(std::vector<char> &buf) const override {
        throw std::runtime_error("serialize is not supported for List yet");
    }

    /**
     * @brief Returns the number of elements in this list.
     *
     * @return The number of elements in this list.
     */
    size_t length() const { return elements.size(); }

    /**
     * @brief Appends the given element to the end of this list.
     *
     * @param element The element to append.
     */
    void append(const DataType *element) {
        // We must increase the reference counter of the new element to prevent
        // it from being freed as long as this list exists.
        element->increaseRefCounter();
        elements.push_back(element);
    }

    /**
     * @brief Removes the element at the given position from this list and
     * returns it.
     *
     * @param idx The position of the element to remove.
     * @return The removed element.
     */
    const DataType *remove(size_t idx) {
        if (idx >= elements.size())
            throw std::runtime_error("trying to remove element at position " + std::to_string(idx) +
                                     " from a list with " + std::to_string(elements.size()) + " elements");
        const DataType *element = elements[idx];
        elements.erase(elements.begin() + idx);
        // Note that we do not decrease the reference counter of the element. It
        // must not be freed here, since we return it.
        return element;
    }

    /**
     * @brief Replaces the element at the given position with the provided element.
     *
     * @param idx The position of the element to replace.
     * @param element The new element to insert.
     * @return The old element that was replaced.
     */
    const DataType *replace(size_t idx, const DataType* element) {
        if (idx >= elements.size())
            throw std::runtime_error("trying to replace element at position " + std::to_string(idx) +
                                    " in a list with " + std::to_string(elements.size()) + " elements");
        const DataType* oldElement = elements[idx];
        // Increase ref counter for the new element before replacing
        element->increaseRefCounter();
        // Replace the element
        elements[idx] = element;
        // Note that we do not decrease the reference counter of the element. It
        // must not be freed here, since we return it.       
        return oldElement;
    }

    /**
     * @brief Returns a copy of the element at the given position.
     *
     * @param idx The position of the element to return.
     * @return The element at the given position.
     */
    const DataType *getElementInList(size_t idx) const {
        if (idx >= elements.size())
            throw std::runtime_error("trying to access element at position " + std::to_string(idx) +
                                    " from a list with " + std::to_string(elements.size()) + " elements");
        // Increase ref counter for the element before returning it
        elements[idx]->increaseRefCounter();
        return elements[idx];
    }
};

template <typename DataType> std::ostream &operator<<(std::ostream &os, const List<DataType> &obj) {
    obj.print(os);
    return os;
}
