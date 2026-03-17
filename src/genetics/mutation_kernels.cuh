#ifndef MUTATION_KERNELS_CUH
#define MUTATION_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../core/constants.cuh"

struct MutationParams {
    float base_rate;
    float gaussian_sigma;
    float reset_prob;
    float duplication_prob;
    float deletion_prob;
};

void launchMutationKernel(
    float* d_genomes,
    curandState* d_rng,
    int num_children,
    const MutationParams& params,
    cudaStream_t stream = 0
);

#endif