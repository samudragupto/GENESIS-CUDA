#include "genome_to_weights.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void genomeToWeightsKernel(
    const float* __restrict__ genomes,
    float* __restrict__ weights,
    int num_creatures,
    int genome_size,
    int weight_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;

    const float* genome = genomes + idx * genome_size;
    float* w = weights + idx * weight_count;

    float scale_w1 = sqrtf(2.0f / (float)NEURAL_INPUT_SIZE);
    float scale_w2 = sqrtf(2.0f / (float)NEURAL_HIDDEN1_SIZE);
    float scale_w3 = sqrtf(2.0f / (float)NEURAL_HIDDEN2_SIZE);

    int ng_start = GENE_ACTIVATION_TYPE;

    for (int i = 0; i < TOTAL_USED_WEIGHTS && i < weight_count; i++) {
        int gene_idx = (ng_start + i) % genome_size;
        float gene_val = genome[gene_idx] * 2.0f - 1.0f;

        float scale;
        if (i < B1_OFFSET) scale = scale_w1;
        else if (i < B1_OFFSET + B1_SIZE) scale = scale_w1 * 0.1f;
        else if (i < B2_OFFSET) scale = scale_w2;
        else if (i < B2_OFFSET + B2_SIZE) scale = scale_w2 * 0.1f;
        else if (i < B3_OFFSET) scale = scale_w3;
        else scale = scale_w3 * 0.1f;

        w[i] = gene_val * scale;
    }

    for (int i = TOTAL_USED_WEIGHTS; i < weight_count; i++) {
        w[i] = 0.0f;
    }
}

void launchGenomeToWeightsBatch(
    const float* d_genomes,
    float* d_weights,
    int num_creatures,
    int genome_size,
    int weight_count,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    genomeToWeightsKernel<<<grid, block, 0, stream>>>(
        d_genomes, d_weights, num_creatures, genome_size, weight_count
    );
}