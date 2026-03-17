#ifndef MEMORY_OPTIMIZER_CUH
#define MEMORY_OPTIMIZER_CUH

#include <cuda_runtime.h>
#include <cstddef>

#define ALIGN_UP(x, alignment) (((x) + (alignment) - 1) & ~((alignment) - 1))
#define ALIGN_128(x) ALIGN_UP(x, 128)
#define COALESCED_SIZE(count, type) ALIGN_128((count) * sizeof(type))
#define WARP_SIZE 32

struct MemoryStats {
    size_t total_allocated;
    size_t total_freed;
    size_t peak_usage;
    size_t current_usage;
    int allocation_count;
    int free_count;
};

class AlignedAllocator {
public:
    size_t alignment;
    MemoryStats stats;

    void init(size_t align = 128);

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    template<typename T>
    T* allocateArray(int count);

    template<typename T>
    void deallocateArray(T* ptr);

    MemoryStats getStats() const;
    void printStats() const;
};

class PinnedMemoryPool {
public:
    void* pool;
    size_t pool_size;
    size_t used;

    void init(size_t size);
    void destroy();

    void* allocate(size_t bytes);
    void reset();
};

__device__ __forceinline__ int warpReduceSum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceSumF(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceMin(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ int warpBroadcast(int val, int src_lane) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}

__device__ __forceinline__ float warpBroadcastF(float val, int src_lane) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}

__device__ __forceinline__ int warpScan(int val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) val += n;
    }
    return val;
}

__device__ __forceinline__ int laneId() {
    return threadIdx.x & (WARP_SIZE - 1);
}

__device__ __forceinline__ int warpId() {
    return threadIdx.x / WARP_SIZE;
}

size_t computeOptimalSharedMem(int block_size, size_t per_thread_bytes);
int computeOptimalBlockSize(void* kernel_func, size_t shared_mem_per_block,
                            int max_block_size);
void prefetchToDevice(void* ptr, size_t bytes, int device, cudaStream_t stream = 0);
void prefetchToHost(void* ptr, size_t bytes, cudaStream_t stream = 0);

#endif