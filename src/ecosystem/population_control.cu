#include "population_control.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void populationCensusKernel(
    int* __restrict__ species_pop_count,
    float* __restrict__ species_avg_energy,
    const float* __restrict__ energy,
    const int* __restrict__ species_id,
    const int* __restrict__ alive,
    int num_creatures,
    int max_species
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    int sid = species_id[idx];
    if (sid < 0 || sid >= max_species) return;

    atomicAdd(&species_pop_count[sid], 1);
    atomicAdd(&species_avg_energy[sid], energy[idx]);
}

__global__ void normalizeSpeciesStatsKernel(
    float* __restrict__ species_avg_energy,
    const int* __restrict__ species_pop_count,
    int max_species
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= max_species) return;

    int count = species_pop_count[sid];
    if (count > 0) {
        species_avg_energy[sid] /= (float)count;
    } else {
        species_avg_energy[sid] = 0.0f;
    }
}

__global__ void countTotalAliveKernel(
    const int* __restrict__ alive,
    int* __restrict__ total_alive,
    int num_creatures
) {
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_creatures && alive[idx]) {
        atomicAdd(&s_count, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(total_alive, s_count);
    }
}

__global__ void carryingCapacityKernel(
    float* __restrict__ energy,
    float* __restrict__ health,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const int* __restrict__ alive,
    const int* __restrict__ species_id,
    const int* __restrict__ species_pop_count,
    const float* __restrict__ vegetation,
    int world_size,
    int num_creatures,
    int global_max_population,
    int max_species
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    int sid = species_id[idx];
    if (sid < 0 || sid >= max_species) return;

    int species_pop = species_pop_count[sid];
    int species_cap = global_max_population / max(max_species / 10, 1);

    if (species_pop > species_cap * 2) {
        float overcrowding = (float)(species_pop - species_cap) / (float)species_cap;
        float stress = fminf(overcrowding * 0.01f, 0.05f);
        energy[idx] -= stress;
    }

    int gx = min(max((int)pos_x[idx], 0), world_size - 1);
    int gy = min(max((int)pos_y[idx], 0), world_size - 1);
    float local_veg = vegetation[gy * world_size + gx];

    if (local_veg < 0.01f) {
        energy[idx] -= 0.002f;
    }
}

void launchPopulationCensus(
    PopulationStats& stats,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(stats.d_species_pop_count, 0,
        stats.max_species * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(stats.d_species_avg_energy, 0,
        stats.max_species * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(stats.d_total_alive, 0, sizeof(int), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;

    populationCensusKernel<<<grid, block, 0, stream>>>(
        stats.d_species_pop_count, stats.d_species_avg_energy,
        creatures.d_energy, creatures.d_species_id,
        creatures.d_alive, num_creatures, stats.max_species
    );

    int norm_grid = (stats.max_species + 255) / 256;
    normalizeSpeciesStatsKernel<<<norm_grid, 256, 0, stream>>>(
        stats.d_species_avg_energy, stats.d_species_pop_count,
        stats.max_species
    );

    countTotalAliveKernel<<<grid, block, 0, stream>>>(
        creatures.d_alive, stats.d_total_alive, num_creatures
    );
}

void launchCarryingCapacityEnforcement(
    CreatureData& creatures,
    const PopulationStats& stats,
    const EcosystemGridData& eco_grid,
    int num_creatures,
    int global_max_population,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    carryingCapacityKernel<<<grid, block, 0, stream>>>(
        creatures.d_energy, creatures.d_health,
        creatures.d_pos_x, creatures.d_pos_y,
        creatures.d_alive, creatures.d_species_id,
        stats.d_species_pop_count, eco_grid.d_vegetation,
        eco_grid.world_size, num_creatures,
        global_max_population, stats.max_species
    );
}

void launchSpeciesPopulationCount(
    PopulationStats& stats,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    CUDA_CHECK(cudaMemsetAsync(stats.d_species_pop_count, 0,
        stats.max_species * sizeof(int), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    populationCensusKernel<<<grid, block, 0, stream>>>(
        stats.d_species_pop_count, stats.d_species_avg_energy,
        creatures.d_energy, creatures.d_species_id,
        creatures.d_alive, num_creatures, stats.max_species
    );
}

void launchComputeSpeciesAvgEnergy(
    PopulationStats& stats,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int norm_grid = (stats.max_species + 255) / 256;
    normalizeSpeciesStatsKernel<<<norm_grid, 256, 0, stream>>>(
        stats.d_species_avg_energy, stats.d_species_pop_count,
        stats.max_species
    );
}