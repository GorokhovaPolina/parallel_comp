#define _POSIX_C_SOURCE 199309L  // для clock_gettime на macOS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

//  функции измерения времени (наносекунды)
static inline uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

//  тест 1 - последовательный обход
double test_sequential(int *arr, size_t n, int repeats) {
    uint64_t total_ns = 0;
    volatile int dummy; // для предотвращения оптимизации

    for (int r = 0; r < repeats; r++) {
        int *data = (int*)malloc(n * sizeof(int));
        if (!data) return -1.0;
        for (size_t i = 0; i < n; i++) data[i] = i & 0xFF;

        uint64_t start = ns_now();
        int sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += data[i];
        }
        uint64_t end = ns_now();
        total_ns += (end - start);
        dummy = sum;
        free(data);
    }
    return (double)total_ns / (double)(n * repeats);
}

//  тест 2 - случайный обход
double test_random(int *arr, size_t n, int repeats) {
    uint64_t total_ns = 0;
    volatile int dummy;
    unsigned int seed = 12345; // фиксированный seed для воспроизводимости

    for (int r = 0; r < repeats; r++) {
        int *data = (int*)malloc(n * sizeof(int));
        if (!data) return -1.0;
        for (size_t i = 0; i < n; i++) data[i] = i & 0xFF;

        uint64_t start = ns_now();
        int sum = 0;
        for (size_t i = 0; i < n; i++) {
            size_t idx = rand_r(&seed) % n;
            sum += data[idx];
        }
        uint64_t end = ns_now();
        total_ns += (end - start);
        dummy = sum;
        free(data);
    }
    return (double)total_ns / (double)(n * repeats);
}

//  тест 3: случайный обход с предвычисленным массивом индексов
double test_random_index(int *arr, size_t n, int repeats) {
    uint64_t total_ns = 0;
    volatile int dummy;
    unsigned int seed = 12345;

    for (int r = 0; r < repeats; r++) {
        int *data = (int*)malloc(n * sizeof(int));
        int *indices = (int*)malloc(n * sizeof(int));
        if (!data || !indices) {
            free(data); free(indices);
            return -1.0;
        }
        for (size_t i = 0; i < n; i++) data[i] = i & 0xFF;
        for (size_t i = 0; i < n; i++) {
            indices[i] = rand_r(&seed) % n;
        }

        uint64_t start = ns_now();
        int sum = 0;
        for (size_t i = 0; i < n; i++) {
            size_t idx = indices[i];
            sum += data[idx];
        }
        uint64_t end = ns_now();
        total_ns += (end - start);
        dummy = sum;
        free(data);
        free(indices);
    }
    return (double)total_ns / (double)(n * repeats);
}

int main() {
    struct Range {
        const char *name;
        size_t start_bytes;
        size_t end_bytes;
        size_t step_bytes;
        int repeats;
    };
    struct Range ranges[] = {
        {"L1/L2",      1024,      2*1024*1024,   1024,       5},
        {"L2/L3",      512*1024, 32*1024*1024,  512*1024,   5},
        {"RAM",         5*1024*1024, 150*1024*1024, 5*1024*1024, 3}
    };
    int num_ranges = sizeof(ranges) / sizeof(ranges[0]);

    printf("# size_bytes, sequential_ns, random_ns, random_index_ns\n");

    for (int r = 0; r < num_ranges; r++) {
        size_t start = ranges[r].start_bytes;
        size_t end   = ranges[r].end_bytes;
        size_t step  = ranges[r].step_bytes;
        int repeats  = ranges[r].repeats;

        for (size_t bytes = start; bytes <= end; bytes += step) {
            size_t n = bytes / sizeof(int);
            if (n == 0) continue;

            int *dummy_arr = NULL;
            double t_seq = test_sequential(dummy_arr, n, repeats);
            double t_rand = test_random(dummy_arr, n, repeats);
            double t_rand_idx = test_random_index(dummy_arr, n, repeats);

            if (t_seq < 0 || t_rand < 0 || t_rand_idx < 0) {
                fprintf(stderr, "Ошибка выделения памяти для размера %zu байт\n", bytes);
                continue;
            }

            printf("%zu, %.2f, %.2f, %.2f\n", bytes, t_seq, t_rand, t_rand_idx);
            fflush(stdout);
        }
    }
    return 0;
}