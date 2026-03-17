#ifndef ECOSYSTEM_MANAGER_CUH
#define ECOSYSTEM_MANAGER_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "ecosystem_common.cuh"
#include "food_web.cuh"
#include "resource_manager.cuh"
#include "disease_model.cuh"
#include "population_control.cuh"
#include "../creatures/creature_common.cuh"

class EcosystemManager {
public:
    EcosystemGridData eco_grid;
    EcosystemParams eco_params;
    FoodWebData food_web;
    DiseaseData disease;
    DiseaseParams disease_params;
    PopulationStats pop_stats;

    cudaStream_t stream_vegetation;
    cudaStream_t stream_foodweb;
    cudaStream_t stream_disease;
    cudaStream_t stream_population;

    int tick_counter;
    int disease_active;

    void init(int world_size, int max_species, int max_creatures);
    void destroy();

    void initVegetation(
        const float* d_heightmap,
        const float* d_temperature,
        const float* d_moisture
    );

    void update(
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
    );

    void triggerDisease(int num_creatures, curandState* d_rng);

    float* getVegetationPtr();
    int getTotalAlive();
};

#endif