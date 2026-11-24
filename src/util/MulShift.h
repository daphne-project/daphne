#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

inline constexpr uint64_t kFib64 = 0x9E3779B97F4A7C15ull;

// mulShift performs hashing on typed T key on 2^l buckets
template <typename T> inline size_t mulShift(T k, unsigned l) {
    uint64_t h = (uint64_t)k * kFib64;
    return l == 0 ? 0 : (size_t)(h >> (64 - l));
}

template <typename T> struct MulShift {
  private:
    size_t l;
    std::hash<T> hasher;

  public:
    MulShift(size_t cap) {
        size_t c = 1;
        l = 0;
        while (c < cap) {
            c <<= 1;
            l++;
        }
    };

    // fit returns the appropriate capacity for given wanted buckets
    static size_t fit(size_t want) {
        size_t c = 1;
        while (c < want) {
            c <<= 1;
        }
        return c;
    }

    size_t operator()(T k) const { return mulShift(hasher(k), l); }
};
