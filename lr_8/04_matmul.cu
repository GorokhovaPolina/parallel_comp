#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void matmul_row_cache(const float *A, const float *B, float *C, int n) {
    extern __shared__ float sharedA[];
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (col < n)
        sharedA[col] = A[row * n + col];
    __syncthreads();
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += sharedA[k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void matmul_col_cache(const float *A, const float *B, float *C, int n) {
    extern __shared__ float sharedB[];
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < n)
        sharedB[row] = B[row * n + col];
    __syncthreads();
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * sharedB[k];
        C[row * n + col] = sum;
    }
}

// Блочное умножение (tiling) с заданным размером блока S
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n, int S) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * S * by;
    int aEnd = aBegin + n - 1;
    int aStep = S;
    int bBegin = S * bx;
    int bStep = S * n;
    float sum = 0.0f;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[32][32]; // предполагаем S <= 32
        __shared__ float Bs[32][32];
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];
        __syncthreads();
        for (int k = 0; k < S; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    int row = by * S + ty;
    int col = bx * S + tx;
    if (row < n && col < n)
        C[row * n + col] = sum;
}

void run_naive(const float *hA, const float *hB, float *hC, int n) {
    size_t size = n * n * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size); cudaMalloc(&dB, size); cudaMalloc(&dC, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    int blockSize = 16; // фиксированный, т.к. naive не использует shared memory
    dim3 threads(blockSize, blockSize);
    dim3 blocks((n + blockSize - 1)/blockSize, (n + blockSize - 1)/blockSize);
    matmul_naive<<<blocks, threads>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void run_row_cache(const float *hA, const float *hB, float *hC, int n) {
    size_t size = n * n * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size); cudaMalloc(&dB, size); cudaMalloc(&dC, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 threads(n);
    dim3 blocks(n, 1);
    int shmSize = n * sizeof(float);
    matmul_row_cache<<<blocks, threads, shmSize>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void run_col_cache(const float *hA, const float *hB, float *hC, int n) {
    size_t size = n * n * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size); cudaMalloc(&dB, size); cudaMalloc(&dC, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 threads(n);
    dim3 blocks(n, 1);
    int shmSize = n * sizeof(float);
    matmul_col_cache<<<blocks, threads, shmSize>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

void run_tiled(const float *hA, const float *hB, float *hC, int n, int S) {
    size_t size = n * n * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size); cudaMalloc(&dB, size); cudaMalloc(&dC, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 threads(S, S);
    dim3 blocks(n/S, n/S); // предполагается, что n делится на S
    matmul_tiled<<<blocks, threads>>>(dA, dB, dC, n, S);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
