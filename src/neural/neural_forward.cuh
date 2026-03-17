#ifndef NEURAL_FORWARD_CUH
#define NEURAL_FORWARD_CUH

#include <cuda_runtime.h>
#include "neural_common.cuh"

void launchNeuralForwardPass(
    const float* d_input,
    const float* d_weights,
    const float* d_genomes,
    float* d_output,
    int num_creatures,
    int genome_size,
    int weight_count,
    cudaStream_t stream = 0
);

#endif