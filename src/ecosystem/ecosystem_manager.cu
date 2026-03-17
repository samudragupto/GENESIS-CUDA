#include "ecosystem_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void initVegetationKernel(
    float* __restrict__ vegetation,
    const float* __restrict__ heightmap,
    const float* __restrict__ temperature,
    const float* __restrict__ moisture,
    int world_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    float h = heightmap[idx];

    if (h < WATER_LEVEL) {
        vegetation[idx] = 0.0f;
        return;
    }

    float temp = temperature[idx];
    float moist = moisture[idx];

    float temp_factor = 1.0f - fabsf(temp - 20.0f) / 30.0f;
    temp_factor = fmaxf(temp_factor, 0.0f);
    float moist_factor = fminf(moist / 0.5f, 1.0f);
    float alt_factor = 1.0f - fmaxf((h - 0.7f) / 0.3f, 0.0f);
    alt_factor = fmaxf(alt_factor, 0.0f);

    vegetation[idx] = temp_factor * moist_factor * alt_factor * 0.5f;
}

void EcosystemManager::init(int world_size, int max_species, int max_creatures) {
    allocateEcosystemGrid(eco_grid, world_size);
    allocateFoodWebData(food_web, max_species);
    allocateDiseaseData(disease, max_creatures);
    allocatePopulationStats(pop_stats, max_species);

    eco_params.vegetation_growth_rate = 0.01f;
    eco_params.vegetation_spread_rate = 0.005f;
    eco_params.dead_matter_decay_rate = 0.005f;
    eco_params.nutrient_diffusion_rate = 0.1f;
    eco_params.toxicity_decay_rate = 0.001f;
    eco_params.min_growth_temperature = 5.0f;
    eco_params.max_growth_temperature = 40.0f;
    eco_params.optimal_temperature = 22.0f;
    eco_params.min_growth_moisture = 0.1f;
    eco_params.max_vegetation = 2.0f;
    eco_params.dt = 1.0f;

    disease_params.transmission_radius = 5.0f;
    disease_params.transmission_prob = 0.05f;
    disease_params.infection_duration = 100.0f;
    disease_params.mortality_rate = 0.1f;
    disease_params.immunity_duration = 500.0f;
    disease_params.health_damage_rate = 0.005f;
    disease_params.energy_drain_rate = 0.003f;
    disease_params.mutation_chance = 0.01f;
    disease_params.dt = 1.0f;

    CUDA_CHECK(cudaStreamCreate(&stream_vegetation));
    CUDA_CHECK(cudaStreamCreate(&stream_foodweb));
    CUDA_CHECK(cudaStreamCreate(&stream_disease));
    CUDA_CHECK(cudaStreamCreate(&stream_population));

    tick_counter = 0;
    disease_active = 0;
}

void EcosystemManager::destroy() {
    freeEcosystemGrid(eco_grid);
    freeFoodWebData(food_web);
    freeDiseaseData(disease);
    freePopulationStats(pop_stats);

    cudaStreamDestroy(stream_vegetation);
    cudaStreamDestroy(stream_foodweb);
    cudaStreamDestroy(stream_disease);
    cudaStreamDestroy(stream_population);
}

void EcosystemManager::initVegetation(
    const float* d_heightmap,
    const float* d_temperature,
    const float* d_moisture
) {
    dim3 block(16, 16);
    dim3 grid((eco_grid.world_size + 15) / 16, (eco_grid.world_size + 15) / 16);

    initVegetationKernel<<<grid, block>>>(
        eco_grid.d_vegetation, d_heightmap,
        d_temperature, d_moisture, eco_grid.world_size
    );

    launchComputeVegetationCapacity(
        eco_grid, d_heightmap, d_temperature, d_moisture, 0
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

void EcosystemManager::update(
    float dt,
    CreatureData& creatures,
    int num_creatures,
    const float* d_heightmap,
    const float* d_temperature,
    const float* d_moisture,
    const int* d_cell_start,
    const int* d_cell_end,
    const int* d_sorted_indices,
    int spatial_grid_size,
    float cell_size,
    curandState* d_rng
) {
    eco_params.dt = dt;
    disease_params.dt = dt;
    tick_counter++;

    launchVegetationGrowth(
        eco_grid, d_heightmap, d_temperature, d_moisture,
        eco_params, stream_vegetation
    );

    if (tick_counter % 5 == 0) {
        launchVegetationSpread(eco_grid, d_heightmap, eco_params, stream_vegetation);
    }

    launchDeadMatterDecay(eco_grid, d_temperature, eco_params, stream_vegetation);
    launchNutrientDiffusion(eco_grid, eco_params, stream_vegetation);

    if (tick_counter % 50 == 0) {
        launchComputeVegetationCapacity(
            eco_grid, d_heightmap, d_temperature, d_moisture, stream_vegetation
        );
    }

    launchDeadMatterDeposit(creatures, eco_grid, num_creatures, stream_foodweb);

    if (tick_counter % 10 == 0) {
        launchComputeTrophicLevels(food_web, creatures, num_creatures, stream_foodweb);
        launchEnergyFlowKernel(food_web, creatures, eco_grid,
            num_creatures, eco_grid.world_size, dt, stream_foodweb);
    }

    if (disease_active) {
        launchDiseaseSpread(
            disease, creatures,
            d_cell_start, d_cell_end, d_sorted_indices,
            spatial_grid_size, cell_size,
            disease_params, num_creatures, d_rng, stream_disease
        );
        launchDiseaseProgress(
            disease, creatures,
            disease_params, num_creatures, d_rng, stream_disease
        );
    }

    launchPopulationCensus(pop_stats, creatures, num_creatures, stream_population);

    if (tick_counter % 20 == 0) {
        int global_max = MAX_CREATURES;
        launchCarryingCapacityEnforcement(
            creatures, pop_stats, eco_grid,
            num_creatures, global_max, stream_population
        );
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_vegetation));
    CUDA_CHECK(cudaStreamSynchronize(stream_foodweb));
    CUDA_CHECK(cudaStreamSynchronize(stream_disease));
    CUDA_CHECK(cudaStreamSynchronize(stream_population));
}

void EcosystemManager::triggerDisease(int num_creatures, curandState* d_rng) {
    int num_initial = max(num_creatures / 1000, 1);
    launchDiseaseSeed(disease, num_creatures, num_initial, d_rng, 0);
    disease_active = 1;
}

float* EcosystemManager::getVegetationPtr() {
    return eco_grid.d_vegetation;
}

int EcosystemManager::getTotalAlive() {
    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, pop_stats.d_total_alive, sizeof(int), cudaMemcpyDeviceToHost));
    return total;
}