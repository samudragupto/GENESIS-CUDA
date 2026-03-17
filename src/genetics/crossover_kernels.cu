#include "crossover_kernels.cuh"
#include "../core/cuda_utils.cuh"

__global__ void uniformCrossoverKernel(
    const float* __restrict__ parent_genomes,
    float* __restrict__ child_genomes,
    const ReproductionPair* __restrict__ pairs,
    int num_pairs,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    ReproductionPair pair = pairs[idx];
    int pa = pair.parent_a;
    int pb = pair.parent_b;
    int child = pair.child_index;
    curandState local_rng = rng[idx];

    int base_a = pa * GENOME_LENGTH;
    int base_b = pb * GENOME_LENGTH;
    int base_c = child * GENOME_LENGTH;

    for (int g = 0; g < GENOME_LENGTH; g++) {
        if (curand_uniform(&local_rng) < 0.5f) {
            child_genomes[base_c + g] = parent_genomes[base_a + g];
        } else {
            child_genomes[base_c + g] = parent_genomes[base_b + g];
        }
    }
    rng[idx] = local_rng;
}

__global__ void singlePointCrossoverKernel(
    const float* __restrict__ parent_genomes,
    float* __restrict__ child_genomes,
    const ReproductionPair* __restrict__ pairs,
    int num_pairs,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    ReproductionPair pair = pairs[idx];
    curandState local_rng = rng[idx];
    int crossover_point = (int)(curand_uniform(&local_rng) * (GENOME_LENGTH - 1)) + 1;

    int base_a = pair.parent_a * GENOME_LENGTH;
    int base_b = pair.parent_b * GENOME_LENGTH;
    int base_c = pair.child_index * GENOME_LENGTH;

    for (int g = 0; g < GENOME_LENGTH; g++) {
        child_genomes[base_c + g] = (g < crossover_point) ?
            parent_genomes[base_a + g] : parent_genomes[base_b + g];
    }
    rng[idx] = local_rng;
}

__global__ void arithmeticCrossoverKernel(
    const float* __restrict__ parent_genomes,
    float* __restrict__ child_genomes,
    const ReproductionPair* __restrict__ pairs,
    int num_pairs,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    ReproductionPair pair = pairs[idx];
    curandState local_rng = rng[idx];
    float alpha = curand_uniform(&local_rng);

    int base_a = pair.parent_a * GENOME_LENGTH;
    int base_b = pair.parent_b * GENOME_LENGTH;
    int base_c = pair.child_index * GENOME_LENGTH;

    for (int g = 0; g < GENOME_LENGTH; g++) {
        child_genomes[base_c + g] = parent_genomes[base_a + g] * alpha +
                                     parent_genomes[base_b + g] * (1.0f - alpha);
    }
    rng[idx] = local_rng;
}

void launchUniformCrossover(const float* d_parent_genomes, float* d_child_genomes,
    const ReproductionPair* pairs, int num_pairs, curandState* d_rng, cudaStream_t stream) {
    if (num_pairs <= 0) return;
    int block = 256;
    int grid = (num_pairs + block - 1) / block;
    uniformCrossoverKernel<<<grid, block, 0, stream>>>(d_parent_genomes, d_child_genomes, pairs, num_pairs, d_rng);
}

void launchSinglePointCrossover(const float* d_parent_genomes, float* d_child_genomes,
    const ReproductionPair* pairs, int num_pairs, curandState* d_rng, cudaStream_t stream) {
    if (num_pairs <= 0) return;
    int block = 256;
    int grid = (num_pairs + block - 1) / block;
    singlePointCrossoverKernel<<<grid, block, 0, stream>>>(d_parent_genomes, d_child_genomes, pairs, num_pairs, d_rng);
}

void launchArithmeticCrossover(const float* d_parent_genomes, float* d_child_genomes,
    const ReproductionPair* pairs, int num_pairs, curandState* d_rng, cudaStream_t stream) {
    if (num_pairs <= 0) return;
    int block = 256;
    int grid = (num_pairs + block - 1) / block;
    arithmeticCrossoverKernel<<<grid, block, 0, stream>>>(d_parent_genomes, d_child_genomes, pairs, num_pairs, d_rng);
}

void launchCrossoverDispatch(const float* d_parent_genomes, float* d_child_genomes,
    const ReproductionPair* pairs, int num_pairs, curandState* d_rng, int type, cudaStream_t stream) {
    switch (type) {
        case 0: launchUniformCrossover(d_parent_genomes, d_child_genomes, pairs, num_pairs, d_rng, stream); break;
        case 1: launchSinglePointCrossover(d_parent_genomes, d_child_genomes, pairs, num_pairs, d_rng, stream); break;
        case 2: launchArithmeticCrossover(d_parent_genomes, d_child_genomes, pairs, num_pairs, d_rng, stream); break;
        default: launchUniformCrossover(d_parent_genomes, d_child_genomes, pairs, num_pairs, d_rng, stream); break;
    }
}