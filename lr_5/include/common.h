#pragma once

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdlib>

constexpr size_t N = 10000000;

using src_t = uint8_t;
using dst_t = uint16_t;

inline void* aligned_alloc_(size_t align, size_t size) {
    void* ptr = nullptr;
    ::posix_memalign(&ptr, align, size);
    return ptr;
}

inline void aligned_free_(void* ptr) {
    free(ptr);
}

class Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_;
public:
    Timer() : start_(clock::now()) {}
    void reset() { start_ = clock::now(); }
    double elapsed_ms() const {
        auto end = clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

template<typename T>
bool verify(const T* expected, const T* actual, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (expected[i] != actual[i]) {
            std::cerr << "Mismatch at index " << i << ": expected "
                      << (int)expected[i] << ", got " << (int)actual[i] << std::endl;
            return false;
        }
    }
    return true;
}