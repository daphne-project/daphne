#pragma once

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Structure.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <concepts>
#include <ostream>
#include <stdexcept>
#include <vector>

template <typename H, typename KeyType>
concept IHashFn = requires(H &h, const KeyType &k) {
    { h.fit() } -> std::convertible_to<size_t>;
    { h(k) } -> std::convertible_to<size_t>;
};

template <typename KeyType, typename HashFn>
    requires IHashFn<HashFn, KeyType>
class HashSet : public Structure {
    template <class DataType, typename... ArgTypes> friend DataType *DataObjectFactory::create(ArgTypes...);
    template <class DataType_> friend void DataObjectFactory::destroy(const DataType_ *obj);

  private:
    std::vector<KeyType> keys; // TODO(#511): Do we want to swap out std set / map here instead of our own hashFn

    // TODO: currently we simply use boolean map to denote empty state
    // could revisit this if removing this flags is neccessary (use some special value as empty)
    std::vector<bool> flags;
    size_t keyCount = 0;
    size_t cap;
    HashFn hasher;

    // TODO(#511): decide what numRows / numCols mean for a hash set.
    HashSet(size_t capacity, HashFn &&hashFn) : Structure(0, 0) {
        hasher = std::move(hashFn);
        cap = hashFn.fit(capacity);
        keys.resize(cap + 1, false); // we need more than cap 1 slot for empty check
        flags.resize(cap + 1, false);
    };

    size_t next(size_t i) const { return i + 1 == cap ? 0 : i + 1; };

  public:
    void insert(KeyType k) {
        if (keyCount + 1 == cap) {
            throw std::runtime_error("insert more than hashset capacity");
        }
        size_t i = hasher(k);
        while (flags[i])
            i = next(i);

        keyCount++;
        keys[i] = k;
        flags[i] = true;
    };

    bool has(KeyType k) const {
        size_t i = hasher(k);
        while (flags[i] && keys[i] != k)
            i = next(i);
        return flags[i];
    };

    size_t getNumItems() const override { return keyCount; }

    void print(std::ostream &os) const override {
        os << "HashSet(" << keyCount << ", " << ValueTypeUtils::cppNameFor<KeyType> << ")\n";
        for (size_t i = 0; i < cap; i++) {
            if (flags[i])
                os << keys[i] << ", ";
        }
    }

    size_t getNumDims() const override { throw std::runtime_error("getNumDims is not supported for HashSet yet"); }

    HashSet *sliceRow(size_t rl, size_t ru) const override {
        throw std::runtime_error("sliceRow is not supported for HashSet yet");
    };

    HashSet *sliceCol(size_t cl, size_t cu) const override {
        throw std::runtime_error("sliceCol is not supported for HashSet yet");
    }

    HashSet *slice(size_t rl, size_t ru, size_t cl, size_t cu) const override {
        throw std::runtime_error("slice is not supported for HashSet yet");
    }

    size_t serialize(std::vector<char> &buf) const override {
        throw std::runtime_error("serialize is not supported for HashSet yet");
    }
};
