#include "memory_optimizer.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstring>

void AlignedAllocator::init(size_t align) {
    alignment = align;
    memset(&stats, 0, sizeof(stats));
}

void* AlignedAllocator::allocate(size_t bytes) {
    size_t aligned_bytes = ALIGN_UP(bytes, alignment);
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, aligned_bytes));

    stats.total_allocated += aligned_bytes;
    stats.current_usage += aligned_bytes;
    stats.allocation_count++;
    if (stats.current_usage > stats.peak_usage) {
        stats.peak_usage = stats.current_usage;
    }

    return ptr;
}

void AlignedAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    cudaFree(ptr);
    stats.free_count++;
}

template<typename T>
T* AlignedAllocator::allocateArray(int count) {
    size_t bytes = COALESCED_SIZE(count, T);
    return (T*)allocate(bytes);
}

template<typename T>
void AlignedAllocator::deallocateArray(T* ptr) {
    deallocate((void*)ptr);
}

MemoryStats AlignedAllocator::getStats() const {
    return stats;
}

void AlignedAllocator::printStats() const {
    printf("\n=== Memory Allocator Stats ===\n");
    printf("  Allocations: %d | Frees: %d\n", stats.allocation_count, stats.free_count);
    printf("  Total Allocated: %.2f MB\n", (float)stats.total_allocated / (1024.0f * 1024.0f));
    printf("  Current Usage: %.2f MB\n", (float)stats.current_usage / (1024.0f * 1024.0f));
    printf("  Peak Usage: %.2f MB\n", (float)stats.peak_usage / (1024.0f * 1024.0f));
    printf("==============================\n");
}

void PinnedMemoryPool::init(size_t size) {
    pool_size = size;
    used = 0;
    CUDA_CHECK(cudaMallocHost(&pool, size));
}

void PinnedMemoryPool::destroy() {
    if (pool) {
        cudaFreeHost(pool);
        pool = nullptr;
    }
}

void* PinnedMemoryPool::allocate(size_t bytes) {
    size_t aligned = ALIGN_UP(bytes, 64);
    if (used + aligned > pool_size) return nullptr;
    void* ptr = (char*)pool + used;
    used += aligned;
    return ptr;
}

void PinnedMemoryPool::reset() {
    used = 0;
}

size_t computeOptimalSharedMem(int block_size, size_t per_thread_bytes) {
    size_t requested = block_size * per_thread_bytes;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t max_shared = prop.sharedMemPerBlock;
    return (requested < max_shared) ? requested : max_shared;
}

int computeOptimalBlockSize(void* kernel_func, size_t shared_mem_per_block,
                             int max_block_size) {
    int min_grid = 0;
    int opt_block = 0;

    cudaOccupancyMaxPotentialBlockSize(&min_grid, &opt_block,
        (void(*)(void))kernel_func, shared_mem_per_block, max_block_size);

    return opt_block;
}

void prefetchToDevice(void* ptr, size_t bytes, int device, cudaStream_t stream) {
    (void)ptr;
    (void)bytes;
    (void)device;
    (void)stream;
}

void prefetchToHost(void* ptr, size_t bytes, cudaStream_t stream) {
    (void)ptr;
    (void)bytes;
    (void)stream;
}

template float* AlignedAllocator::allocateArray<float>(int);
template int* AlignedAllocator::allocateArray<int>(int);
template void AlignedAllocator::deallocateArray<float>(float*);
template void AlignedAllocator::deallocateArray<int>(int*);