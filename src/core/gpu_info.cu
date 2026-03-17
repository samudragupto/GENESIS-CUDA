#include "gpu_info.cuh"
#include "cuda_utils.cuh"
#include <cstdio>

void printGPUInfo() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("\n=== GPU Device Info ===\n");
    printf("  CUDA Devices Found: %d\n", device_count);
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("\n  Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  SM Count: %d\n", prop.multiProcessorCount);
        printf("  Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    }
    printf("========================\n\n");
}

int getGPUCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

int getComputeCapabilityMajor(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major;
}

int getComputeCapabilityMinor(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.minor;
}

size_t getGPUMemoryBytes(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.totalGlobalMem;
}