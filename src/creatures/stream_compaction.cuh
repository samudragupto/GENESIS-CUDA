#ifndef STREAM_COMPACTION_CUH
#define STREAM_COMPACTION_CUH

#include <cuda_runtime.h>
#include "creature_common.cuh"

struct CompactionBuffers {
    int* d_scan_input;
    int* d_scan_output;
    int* d_temp_indices;
    int  capacity;
};

void allocateCompactionBuffers(CompactionBuffers& buf, int capacity);
void freeCompactionBuffers(CompactionBuffers& buf);

int launchStreamCompaction(
    CreatureData& creatures,
    CompactionBuffers& compaction,
    int current_count,
    cudaStream_t stream = 0
);

#endif