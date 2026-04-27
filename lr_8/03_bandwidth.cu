#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <chrono>

using namespace std::chrono;

void test_copy(const char* label, void* dst, void* src, size_t bytes, cudaMemcpyKind kind, bool sync=false) {
    auto start = high_resolution_clock::now();
    cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        return;
    }
    if (sync) cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    double time_ms = duration<double, std::milli>(end - start).count();
    double bw_gb = (bytes / (1024.0*1024.0*1024.0)) / (time_ms / 1000.0);
    std::cout << label << " : " << time_ms << " ms, " << bw_gb << " GB/s\n";
}

int main() {
    const size_t N = 256 * 1024 * 1024; // 256 MB
    const size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_a_pinned, *h_b_pinned;
    cudaMallocHost(&h_a_pinned, bytes);
    cudaMallocHost(&h_b_pinned, bytes);

    float *d_a, *d_b;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    memset(h_a, 0, bytes);
    memset(h_b, 0, bytes);
    memset(h_a_pinned, 0, bytes);
    memset(h_b_pinned, 0, bytes);

    test_copy("Host->Host (pageable)   ", h_b, h_a, bytes, cudaMemcpyHostToHost);
    test_copy("Host->Device (pageable) ", d_a, h_a, bytes, cudaMemcpyHostToDevice);
    test_copy("Host->Device (pinned)   ", d_a, h_a_pinned, bytes, cudaMemcpyHostToDevice);
    test_copy("Device->Host (pageable) ", h_b, d_a, bytes, cudaMemcpyDeviceToHost);
    test_copy("Device->Host (pinned)   ", h_b_pinned, d_a, bytes, cudaMemcpyDeviceToHost);
    test_copy("Device->Device          ", d_b, d_a, bytes, cudaMemcpyDeviceToDevice, true);

    free(h_a);
    free(h_b);
    cudaFreeHost(h_a_pinned);
    cudaFreeHost(h_b_pinned);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
