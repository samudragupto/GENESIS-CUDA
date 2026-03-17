#ifndef GPU_MEMORY_POOL_CUH
#define GPU_MEMORY_POOL_CUH

#include <cuda_runtime.h>
#include "constants.cuh"
#include <cstddef>

struct MemBlock {
    void* ptr;
    size_t size;
    bool in_use;
};

class GPUMemoryPool {
public:
    void* pool_base;
    size_t pool_size;
    size_t used;
    MemBlock* blocks;
    int num_blocks;
    int max_blocks;

    void initialize(size_t total_bytes);
    void destroy();
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    void reset();
    size_t getUsed() const;
    size_t getFree() const;
    void printStats() const;
};

inline size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

#endif