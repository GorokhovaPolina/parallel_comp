#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA device count: " << deviceCount << "\n\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability:            " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory (GB):      " << std::fixed << std::setprecision(2)
                  << prop.totalGlobalMem / (1024.0*1024.0*1024.0) << "\n";
        std::cout << "  Constant memory (KB):          " << prop.totalConstMem / 1024.0 << "\n";
        std::cout << "  Shared memory per block (KB):  " << prop.sharedMemPerBlock / 1024.0 << "\n";
        std::cout << "  Registers per block:           " << prop.regsPerBlock << "\n";
        std::cout << "  Warp size:                     " << prop.warpSize << "\n";
        std::cout << "  Max threads per block:         " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Number of multiprocessors:     " << prop.multiProcessorCount << "\n";
        std::cout << "  Core clock rate (MHz):         " << prop.clockRate / 1000.0 << "\n";
        std::cout << "  Memory clock rate (MHz):       " << prop.memoryClockRate / 1000.0 << "\n";
        std::cout << "  L2 cache size (KB):            " << prop.l2CacheSize / 1024.0 << "\n";
        std::cout << "  Memory bus width (bit):        " << prop.memoryBusWidth << "\n";
        std::cout << "  Max block dim:                 " << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << "\n";
        std::cout << "  Max grid dim:                  " << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "\n\n";
    }
    return 0;
}
