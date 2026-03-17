#include "diversity_index.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void allocateDiversityResult(DiversityResult& dr) {
    CUDA_CHECK(cudaMalloc(&dr.d_shannon, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dr.d_simpson, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dr.d_evenness, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dr.d_richness, sizeof(float)));
}

void freeDiversityResult(DiversityResult& dr) {
    cudaFree(dr.d_shannon);
    cudaFree(dr.d_simpson);
    cudaFree(dr.d_evenness);
    cudaFree(dr.d_richness);
}

__global__ void shannonReductionKernel(
    const int* __restrict__ species_pop,
    float* __restrict__ partial_shannon,
    int* __restrict__ partial_richness,
    int max_species,
    int total_alive
) {
    __shared__ float s_shannon[256];
    __shared__ int s_richness[256];

    int tid = threadIdx.x;
    int sid = blockIdx.x * blockDim.x + threadIdx.x;

    s_shannon[tid] = 0.0f;
    s_richness[tid] = 0;

    if (sid < max_species && total_alive > 0) {
        int count = species_pop[sid];
        if (count > 0) {
            float p = (float)count / (float)total_alive;
            s_shannon[tid] = -p * logf(p);
            s_richness[tid] = 1;
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_shannon[tid] += s_shannon[tid + stride];
            s_richness[tid] += s_richness[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(partial_shannon, s_shannon[0]);
        atomicAdd(partial_richness, s_richness[0]);
    }
}

__global__ void simpsonReductionKernel(
    const int* __restrict__ species_pop,
    float* __restrict__ partial_simpson,
    int max_species,
    int total_alive
) {
    __shared__ float s_simpson[256];

    int tid = threadIdx.x;
    int sid = blockIdx.x * blockDim.x + threadIdx.x;

    s_simpson[tid] = 0.0f;

    if (sid < max_species && total_alive > 1) {
        int count = species_pop[sid];
        if (count > 0) {
            float p = (float)count / (float)total_alive;
            s_simpson[tid] = p * p;
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_simpson[tid] += s_simpson[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(partial_simpson, s_simpson[0]);
    }
}

__global__ void finalizeDiversityKernel(
    float* __restrict__ shannon,
    float* __restrict__ simpson,
    float* __restrict__ evenness,
    const float* __restrict__ raw_shannon,
    const float* __restrict__ raw_simpson,
    const int* __restrict__ richness_val
) {
    if (threadIdx.x != 0) return;

    float h = *raw_shannon;
    float d = *raw_simpson;
    int s = *richness_val;

    *shannon = h;
    *simpson = 1.0f - d;

    if (s > 1) {
        *evenness = h / logf((float)s);
    } else {
        *evenness = 1.0f;
    }
}

__global__ void geneticDiversityKernel(
    const float* __restrict__ genomes,
    const int* __restrict__ alive,
    float* __restrict__ gene_variance,
    float* __restrict__ gene_mean,
    int num_creatures,
    int num_genes
) {
    int gene = blockIdx.x;
    if (gene >= num_genes) return;

    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    __shared__ int s_count[256];

    int tid = threadIdx.x;
    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;
    s_count[tid] = 0;

    for (int i = tid; i < num_creatures; i += blockDim.x) {
        if (alive[i]) {
            float val = genomes[i * GENOME_SIZE + gene];
            s_sum[tid] += val;
            s_sum_sq[tid] += val * val;
            s_count[tid]++;
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && s_count[0] > 0) {
        float mean = s_sum[0] / (float)s_count[0];
        float variance = s_sum_sq[0] / (float)s_count[0] - mean * mean;
        gene_mean[gene] = mean;
        gene_variance[gene] = variance;
    }
}

void launchShannonDiversity(
    const int* d_species_pop_count,
    int max_species,
    int total_alive,
    DiversityResult& result,
    cudaStream_t stream
) {
    float zero_f = 0.0f;
    int zero_i = 0;
    CUDA_CHECK(cudaMemcpyAsync(result.d_shannon, &zero_f, sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.d_richness, &zero_i, sizeof(int), cudaMemcpyHostToDevice, stream));

    int block = 256;
    int grid = (max_species + block - 1) / block;
    shannonReductionKernel<<<grid, block, 0, stream>>>(
        d_species_pop_count, result.d_shannon,
        (int*)result.d_richness, max_species, total_alive
    );
}

void launchSimpsonDiversity(
    const int* d_species_pop_count,
    int max_species,
    int total_alive,
    DiversityResult& result,
    cudaStream_t stream
) {
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(result.d_simpson, &zero, sizeof(float), cudaMemcpyHostToDevice, stream));

    int block = 256;
    int grid = (max_species + block - 1) / block;
    simpsonReductionKernel<<<grid, block, 0, stream>>>(
        d_species_pop_count, result.d_simpson,
        max_species, total_alive
    );

    finalizeDiversityKernel<<<1, 1, 0, stream>>>(
        result.d_shannon, result.d_simpson, result.d_evenness,
        result.d_shannon, result.d_simpson, (int*)result.d_richness
    );
}

void launchGeneticDiversity(
    const float* d_genomes,
    const int* d_alive,
    int num_creatures,
    float* d_genetic_diversity,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    float* d_gene_mean;
    CUDA_CHECK(cudaMalloc(&d_gene_mean, GENOME_SIZE * sizeof(float)));

    geneticDiversityKernel<<<GENOME_SIZE, 256, 0, stream>>>(
        d_genomes, d_alive, d_genetic_diversity, d_gene_mean,
        num_creatures, GENOME_SIZE
    );

    cudaFree(d_gene_mean);
}