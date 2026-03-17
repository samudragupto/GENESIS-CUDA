#include "food_web.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void allocateFoodWebData(FoodWebData& fw, int max_species) {
    fw.max_species = max_species;
    CUDA_CHECK(cudaMalloc(&fw.d_trophic_level, max_species * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fw.d_energy_intake_rate, max_species * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fw.d_energy_output_rate, max_species * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&fw.d_predator_count, max_species * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&fw.d_prey_count, max_species * sizeof(int)));

    CUDA_CHECK(cudaMemset(fw.d_trophic_level, 0, max_species * sizeof(float)));
    CUDA_CHECK(cudaMemset(fw.d_energy_intake_rate, 0, max_species * sizeof(float)));
    CUDA_CHECK(cudaMemset(fw.d_energy_output_rate, 0, max_species * sizeof(float)));
    CUDA_CHECK(cudaMemset(fw.d_predator_count, 0, max_species * sizeof(int)));
    CUDA_CHECK(cudaMemset(fw.d_prey_count, 0, max_species * sizeof(int)));
}

void freeFoodWebData(FoodWebData& fw) {
    cudaFree(fw.d_trophic_level);
    cudaFree(fw.d_energy_intake_rate);
    cudaFree(fw.d_energy_output_rate);
    cudaFree(fw.d_predator_count);
    cudaFree(fw.d_prey_count);
}

__global__ void computeTrophicKernel(
    float* __restrict__ trophic_levels,
    const float* __restrict__ genomes,
    const int* __restrict__ species_id,
    const int* __restrict__ alive,
    int num_creatures,
    int max_species
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    const float* genome = genomes + idx * GENOME_SIZE;
    float diet = genome[GENE_DIET];
    int sid = species_id[idx];
    if (sid < 0 || sid >= max_species) return;

    float trophic = 1.0f + diet * 2.0f;
    atomicAdd(&trophic_levels[sid], trophic);
}

__global__ void normalizeTrophicKernel(
    float* __restrict__ trophic_levels,
    const int* __restrict__ pop_counts,
    int max_species
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= max_species) return;

    int count = pop_counts[sid];
    if (count > 0) {
        trophic_levels[sid] /= (float)count;
    } else {
        trophic_levels[sid] = 0.0f;
    }
}

__global__ void energyFlowKernel(
    float* __restrict__ energy_intake,
    float* __restrict__ energy_output,
    const float* __restrict__ energy,
    const float* __restrict__ genomes,
    const int* __restrict__ species_id,
    const int* __restrict__ alive,
    int num_creatures,
    int max_species,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    int sid = species_id[idx];
    if (sid < 0 || sid >= max_species) return;

    const float* genome = genomes + idx * GENOME_SIZE;
    float efficiency = genome[GENE_EFFICIENCY];
    float body_size = 0.5f + genome[GENE_SIZE] * 2.0f;

    float metabolic_output = body_size * (1.0f - efficiency * 0.3f) * 0.01f * dt;
    atomicAdd(&energy_output[sid], metabolic_output);

    float current_energy = energy[idx];
    if (current_energy > 0.5f) {
        float excess = (current_energy - 0.5f) * 0.001f * dt;
        atomicAdd(&energy_intake[sid], excess);
    }
}

__global__ void deadMatterDepositKernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ genomes,
    const int* __restrict__ alive,
    const int* __restrict__ state,
    float* __restrict__ dead_matter,
    float* __restrict__ nutrients,
    int world_size,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;

    if (alive[idx]) return;
    if (state[idx] != STATE_DEAD) return;

    float px = pos_x[idx];
    float py = pos_y[idx];
    int gx = min(max((int)px, 0), world_size - 1);
    int gy = min(max((int)py, 0), world_size - 1);
    int cell = gy * world_size + gx;

    const float* genome = genomes + idx * GENOME_SIZE;
    float body_size = 0.5f + genome[GENE_SIZE] * 2.0f;

    atomicAdd(&dead_matter[cell], body_size * 0.1f);
    atomicAdd(&nutrients[cell], body_size * 0.05f);
}

void launchComputeTrophicLevels(
    FoodWebData& food_web,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(food_web.d_trophic_level, 0,
        food_web.max_species * sizeof(float), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    computeTrophicKernel<<<grid, block, 0, stream>>>(
        food_web.d_trophic_level, creatures.d_genomes,
        creatures.d_species_id, creatures.d_alive,
        num_creatures, food_web.max_species
    );
}

void launchEnergyFlowKernel(
    FoodWebData& food_web,
    const CreatureData& creatures,
    EcosystemGridData& eco_grid,
    int num_creatures,
    int world_size,
    float dt,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(food_web.d_energy_intake_rate, 0,
        food_web.max_species * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(food_web.d_energy_output_rate, 0,
        food_web.max_species * sizeof(float), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    energyFlowKernel<<<grid, block, 0, stream>>>(
        food_web.d_energy_intake_rate, food_web.d_energy_output_rate,
        creatures.d_energy, creatures.d_genomes,
        creatures.d_species_id, creatures.d_alive,
        num_creatures, food_web.max_species, dt
    );
}

void launchDeadMatterDeposit(
    const CreatureData& creatures,
    EcosystemGridData& eco_grid,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    deadMatterDepositKernel<<<grid, block, 0, stream>>>(
        creatures.d_pos_x, creatures.d_pos_y,
        creatures.d_genomes, creatures.d_alive,
        creatures.d_state, eco_grid.d_dead_matter,
        eco_grid.d_mineral_nutrients,
        eco_grid.world_size, num_creatures
    );
}