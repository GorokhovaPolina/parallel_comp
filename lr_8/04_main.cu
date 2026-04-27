#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

void matmul_cpu(const float *A, const float *B, float *C, int n);
void run_naive(const float *A, const float *B, float *C, int n);
void run_row_cache(const float *A, const float *B, float *C, int n);
void run_col_cache(const float *A, const float *B, float *C, int n);
void run_tiled(const float *A, const float *B, float *C, int n, int S);

bool compare(const float *C1, const float *C2, int n) {
    for (int i = 0; i < n*n; i++) {
        if (std::fabs(C1[i] - C2[i]) > 1e-2) {
            std::cout << "Mismatch at " << i << ": " << C1[i] << " vs " << C2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void init_matrix(float *M, int n) {
    for (int i = 0; i < n*n; i++) {
        M[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

int main(int argc, char **argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int S_tile = (argc > 2) ? std::atoi(argv[2]) : 16;
    std::cout << "Matrix size: " << N << "x" << N << ", tile size S=" << S_tile << std::endl;

    // size_t size = N * N * sizeof(float);
    float *hA = new float[N*N];
    float *hB = new float[N*N];
    float *hC_cpu = new float[N*N];
    float *hC_gpu = new float[N*N];

    std::srand(42);
    init_matrix(hA, N);
    init_matrix(hB, N);

    // CPU-эталон
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul_cpu(hA, hB, hC_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "CPU time: " << cpu_time << " s\n";

    auto test_gpu = [&](const std::string &name, auto func, auto... args) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        func(hA, hB, hC_gpu, N, args...);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        double t = ms / 1000.0;
        double flops = 2.0 * N * N * N;
        double gflops = flops / t / 1e9;

        cudaError_t err = cudaGetLastError();
        bool ok = compare(hC_cpu, hC_gpu, N);

        std::cout << std::left << std::setw(20) << name
                  << " time: " << std::setw(8) << t << " s"
                  << " perf: " << std::setw(8) << gflops << " Gflops"
                  << " match: " << (ok ? "yes" : "NO")
                  << " err: " << cudaGetErrorString(err) << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    test_gpu("Naive (16x16)", run_naive);
    if (N <= 1024) {
        test_gpu("Row cache", run_row_cache);
        test_gpu("Col cache", run_col_cache);
    }
    if (S_tile > 0)
        test_gpu("Tiled S=" + std::to_string(S_tile), run_tiled, S_tile);

    delete[] hA;
    delete[] hB;
    delete[] hC_cpu;
    delete[] hC_gpu;

    return 0;
}
