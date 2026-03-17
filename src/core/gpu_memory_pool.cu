#include "gpu_memory_pool.cuh"
#include "cuda_utils.cuh"
#include <cstdio>

void GPUMemoryPool::initialize(size_t total_bytes) {
    pool_size = total_bytes;
    used = 0;
    max_blocks = 1024;
    num_blocks = 0;

    CUDA_CHECK(cudaMalloc(&pool_base, total_bytes));
    blocks = new MemBlock[max_blocks];

    for (int i = 0; i < max_blocks; i++) {
        blocks[i].ptr = nullptr;
        blocks[i].size = 0;
        blocks[i].in_use = false;
    }
}

void GPUMemoryPool::destroy() {
    if (pool_base) {
        cudaFree(pool_base);
        pool_base = nullptr;
    }
    if (blocks) {
        delete[] blocks;
        blocks = nullptr;
    }
    pool_size = 0;
    used = 0;
    num_blocks = 0;
}

void* GPUMemoryPool::allocate(size_t bytes) {
    bytes = alignUp(bytes, POOL_ALIGNMENT);

    if (used + bytes > pool_size) {
        return nullptr;
    }

    if (num_blocks >= max_blocks) {
        return nullptr;
    }

    void* ptr = (char*)pool_base + used;

    blocks[num_blocks].ptr = ptr;
    blocks[num_blocks].size = bytes;
    blocks[num_blocks].in_use = true;
    num_blocks++;

    used += bytes;

    size_t remaining = pool_size - used;
    if (remaining > POOL_ALIGNMENT) {
        (void)remaining;
    }

    return ptr;
}

void GPUMemoryPool::deallocate(void* ptr) {
    for (int i = 0; i < num_blocks; i++) {
        if (blocks[i].ptr == ptr && blocks[i].in_use) {
            blocks[i].in_use = false;
            return;
        }
    }
}

void GPUMemoryPool::reset() {
    used = 0;
    for (int i = 0; i < num_blocks; i++) {
        blocks[i].in_use = false;
    }
    num_blocks = 0;
}

size_t GPUMemoryPool::getUsed() const {
    return used;
}

size_t GPUMemoryPool::getFree() const {
    return pool_size - used;
}

void GPUMemoryPool::printStats() const {
    printf("GPU Memory Pool: %.2f MB used / %.2f MB total (%d blocks)\n",
           (float)used / (1024.0f * 1024.0f),
           (float)pool_size / (1024.0f * 1024.0f),
           num_blocks);
}