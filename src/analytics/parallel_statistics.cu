#include "parallel_statistics.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void allocateReductionBuffers(ReductionBuffers& buf, int max_blocks, int max_species) {
    buf.max_blocks = max_blocks;
    CUDA_CHECK(cudaMalloc(&buf.d_energy_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_health_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_age_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_alive_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_species_set, max_species * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_unique_species_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_block_results, max_blocks * 4 * sizeof(float)));
}

void freeReductionBuffers(ReductionBuffers& buf) {
    cudaFree(buf.d_energy_sum);
    cudaFree(buf.d_health_sum);
    cudaFree(buf.d_age_sum);
    cudaFree(buf.d_alive_count);
    cudaFree(buf.d_species_set);
    cudaFree(buf.d_unique_species_count);
    cudaFree(buf.d_block_results);
}

__global__ void basicStatsReductionKernel(
    const float* __restrict__ energy,
    const float* __restrict__ health,
    const int* __restrict__ age,
    const int* __restrict__ alive,
    float* __restrict__ block_results,
    int num_creatures
) {
    __shared__ float s_energy[256];
    __shared__ float s_health[256];
    __shared__ float s_age[256];
    __shared__ int s_count[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    s_energy[tid] = 0.0f;
    s_health[tid] = 0.0f;
    s_age[tid] = 0.0f;
    s_count[tid] = 0;

    if (idx < num_creatures && alive[idx]) {
        s_energy[tid] = energy[idx];
        s_health[tid] = health[idx];
        s_age[tid] = (float)age[idx];
        s_count[tid] = 1;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_energy[tid] += s_energy[tid + stride];
            s_health[tid] += s_health[tid + stride];
            s_age[tid] += s_age[tid + stride];
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = blockIdx.x * 4;
        block_results[base + 0] = s_energy[0];
        block_results[base + 1] = s_health[0];
        block_results[base + 2] = s_age[0];
        block_results[base + 3] = __int_as_float(s_count[0]);
    }
}

__global__ void finalReduceKernel(
    const float* __restrict__ block_results,
    float* __restrict__ energy_sum,
    float* __restrict__ health_sum,
    float* __restrict__ age_sum,
    int* __restrict__ alive_count,
    int num_blocks
) {
    __shared__ float s_energy[256];
    __shared__ float s_health[256];
    __shared__ float s_age[256];
    __shared__ int s_count[256];

    int tid = threadIdx.x;
    s_energy[tid] = 0.0f;
    s_health[tid] = 0.0f;
    s_age[tid] = 0.0f;
    s_count[tid] = 0;

    if (tid < num_blocks) {
        int base = tid * 4;
        s_energy[tid] = block_results[base + 0];
        s_health[tid] = block_results[base + 1];
        s_age[tid] = block_results[base + 2];
        s_count[tid] = __float_as_int(block_results[base + 3]);
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_energy[tid] += s_energy[tid + stride];
            s_health[tid] += s_health[tid + stride];
            s_age[tid] += s_age[tid + stride];
            s_count[tid] += s_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *energy_sum = s_energy[0];
        *health_sum = s_health[0];
        *age_sum = s_age[0];
        *alive_count = s_count[0];
    }
}

__global__ void writeSnapshotKernel(
    AnalyticsSnapshot* __restrict__ snapshot,
    const float* __restrict__ energy_sum,
    const float* __restrict__ health_sum,
    const float* __restrict__ age_sum,
    const int* __restrict__ alive_count,
    const int* __restrict__ unique_species,
    int tick
) {
    if (threadIdx.x != 0) return;

    int count = *alive_count;
    snapshot->tick = tick;
    snapshot->total_alive = count;
    snapshot->total_species = *unique_species;

    if (count > 0) {
        float fc = (float)count;
        snapshot->avg_energy = *energy_sum / fc;
        snapshot->avg_health = *health_sum / fc;
        snapshot->avg_age = *age_sum / fc;
    } else {
        snapshot->avg_energy = 0.0f;
        snapshot->avg_health = 0.0f;
        snapshot->avg_age = 0.0f;
    }
}

__global__ void countUniqueSpeciesKernel(
    const int* __restrict__ species_id,
    const int* __restrict__ alive,
    int* __restrict__ species_set,
    int* __restrict__ unique_count,
    int num_creatures,
    int max_species
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    int sid = species_id[idx];
    if (sid < 0 || sid >= max_species) return;

    int old = atomicCAS(&species_set[sid], 0, 1);
    if (old == 0) {
        atomicAdd(unique_count, 1);
    }
}

__global__ void perSpeciesStatsKernel(
    const float* __restrict__ energy,
    const float* __restrict__ genomes,
    const int* __restrict__ species_id,
    const int* __restrict__ alive,
    int* __restrict__ species_pop,
    float* __restrict__ species_energy,
    float* __restrict__ species_fitness,
    int num_creatures,
    int max_species
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    int sid = species_id[idx];
    if (sid < 0 || sid >= max_species) return;

    atomicAdd(&species_pop[sid], 1);
    atomicAdd(&species_energy[sid], energy[idx]);

    const float* genome = genomes + idx * GENOME_SIZE;
    float fitness = energy[idx] * 0.3f + genome[GENE_EFFICIENCY] * 0.3f +
                    genome[GENE_SPEED] * 0.2f + genome[GENE_SENSE] * 0.2f;
    atomicAdd(&species_fitness[sid], fitness);
}

void launchComputeBasicStats(
    const CreatureData& creatures,
    ReductionBuffers& reduction,
    AnalyticsSnapshot* d_snapshot,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    grid = min(grid, reduction.max_blocks);

    basicStatsReductionKernel<<<grid, block, 0, stream>>>(
        creatures.d_energy, creatures.d_health,
        creatures.d_age, creatures.d_alive,
        reduction.d_block_results, num_creatures
    );

    finalReduceKernel<<<1, 256, 0, stream>>>(
        reduction.d_block_results,
        reduction.d_energy_sum, reduction.d_health_sum,
        reduction.d_age_sum, reduction.d_alive_count,
        grid
    );
}

void launchComputeSpeciesCount(
    const CreatureData& creatures,
    ReductionBuffers& reduction,
    int num_creatures,
    int max_species,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(reduction.d_species_set, 0, max_species * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(reduction.d_unique_species_count, 0, sizeof(int), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;

    countUniqueSpeciesKernel<<<grid, block, 0, stream>>>(
        creatures.d_species_id, creatures.d_alive,
        reduction.d_species_set, reduction.d_unique_species_count,
        num_creatures, max_species
    );
}

void launchPerSpeciesStats(
    const CreatureData& creatures,
    int* d_species_pop,
    float* d_species_energy,
    float* d_species_fitness,
    int num_creatures,
    int max_species,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(d_species_pop, 0, max_species * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_species_energy, 0, max_species * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_species_fitness, 0, max_species * sizeof(float), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;

    perSpeciesStatsKernel<<<grid, block, 0, stream>>>(
        creatures.d_energy, creatures.d_genomes,
        creatures.d_species_id, creatures.d_alive,
        d_species_pop, d_species_energy, d_species_fitness,
        num_creatures, max_species
    );
}