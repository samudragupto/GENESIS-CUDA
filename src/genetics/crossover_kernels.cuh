#ifndef CROSSOVER_KERNELS_CUH
#define CROSSOVER_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../core/constants.cuh"

struct ReproductionPair {
    int parent_a;
    int parent_b;
    int child_index;
};

void launchCrossoverDispatch(
    const float* d_parent_genomes,
    float* d_child_genomes,
    const ReproductionPair* pairs,
    int num_pairs,
    curandState* d_rng,
    int crossover_type,
    cudaStream_t stream = 0
);

#endif