#include <cuda.h>
#include <cuda_runtime.h>

__global__ void VecAddKernel(const float *a, const float *b, float *c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}

extern "C" void vec_add_cuda(const float *a, const float *b, float *c, int n) {
    size_t size = n * sizeof(float);
    float *d_a = 0, *d_b = 0, *d_c = 0;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threads(512);
    dim3 blocks((n + threads.x - 1) / threads.x);

    VecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
