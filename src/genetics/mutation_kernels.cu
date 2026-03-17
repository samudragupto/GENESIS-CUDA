#include "mutation_kernels.cuh"
#include "../core/cuda_utils.cuh"

__global__ void mutationKernel(
    float* __restrict__ genomes,
    curandState* __restrict__ rng,
    int num_children,
    float base_rate,
    float gaussian_sigma,
    float reset_prob,
    float duplication_prob,
    float deletion_prob
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_children) return;

    curandState local_rng = rng[idx];
    int base = idx * GENOME_LENGTH;

    float self_rate = genomes[base + GENE_MUTATION_RATE];
    float self_strength = genomes[base + GENE_MUTATION_STRENGTH];
    float effective_rate = base_rate * (0.5f + self_rate);
    float effective_sigma = gaussian_sigma * (0.5f + self_strength);

    for (int g = 0; g < GENOME_LENGTH; g++) {
        if (curand_uniform(&local_rng) < effective_rate) {
            float r = curand_uniform(&local_rng);
            if (r < reset_prob) {
                genomes[base + g] = curand_uniform(&local_rng);
            } else if (r < reset_prob + duplication_prob) {
                int src = (int)(curand_uniform(&local_rng) * (GENOME_LENGTH - 1));
                genomes[base + g] = genomes[base + src];
            } else if (r < reset_prob + duplication_prob + deletion_prob) {
                genomes[base + g] = 0.5f;
            } else {
                genomes[base + g] += curand_normal(&local_rng) * effective_sigma;
            }
            genomes[base + g] = fminf(fmaxf(genomes[base + g], 0.0f), 1.0f);
        }
    }

    if (curand_uniform(&local_rng) < 0.01f) {
        float p = curand_normal(&local_rng) * 0.01f;
        genomes[base + GENE_MUTATION_RATE] = fminf(fmaxf(genomes[base + GENE_MUTATION_RATE] + p, 0.0f), 1.0f);
    }
    if (curand_uniform(&local_rng) < 0.01f) {
        float p = curand_normal(&local_rng) * 0.01f;
        genomes[base + GENE_MUTATION_STRENGTH] = fminf(fmaxf(genomes[base + GENE_MUTATION_STRENGTH] + p, 0.0f), 1.0f);
    }

    rng[idx] = local_rng;
}

void launchMutationKernel(float* d_genomes, curandState* d_rng, int num_children,
    const MutationParams& params, cudaStream_t stream) {
    if (num_children <= 0) return;
    int block = 256;
    int grid = (num_children + block - 1) / block;
    mutationKernel<<<grid, block, 0, stream>>>(d_genomes, d_rng, num_children,
        params.base_rate, params.gaussian_sigma, params.reset_prob,
        params.duplication_prob, params.deletion_prob);
}