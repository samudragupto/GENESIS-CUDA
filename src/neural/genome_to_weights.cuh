#ifndef GENOME_TO_WEIGHTS_CUH
#define GENOME_TO_WEIGHTS_CUH

#include <cuda_runtime.h>

void launchGenomeToWeightsBatch(
    const float* d_genomes,
    float* d_weights,
    int num_creatures,
    int genome_size,
    int weight_count,
    cudaStream_t stream = 0
);

#endif