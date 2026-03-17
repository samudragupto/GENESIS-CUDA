#ifndef ECOSYSTEM_COMMON_CUH
#define ECOSYSTEM_COMMON_CUH

#include <cuda_runtime.h>
#include "../core/constants.cuh"

struct EcosystemGridData {
    float* d_vegetation;
    float* d_vegetation_capacity;
    float* d_dead_matter;
    float* d_mineral_nutrients;
    float* d_toxicity;
    int    world_size;
    int    total_cells;
};

struct EcosystemParams {
    float vegetation_growth_rate;
    float vegetation_spread_rate;
    float dead_matter_decay_rate;
    float nutrient_diffusion_rate;
    float toxicity_decay_rate;
    float min_growth_temperature;
    float max_growth_temperature;
    float optimal_temperature;
    float min_growth_moisture;
    float max_vegetation;
    float dt;
};

struct PopulationStats {
    int* d_species_pop_count;
    float* d_species_avg_energy;
    float* d_species_avg_fitness;
    int* d_total_alive;
    int* d_total_births;
    int* d_total_deaths;
    int max_species;
};

void allocateEcosystemGrid(EcosystemGridData& grid, int world_size);
void freeEcosystemGrid(EcosystemGridData& grid);
void allocatePopulationStats(PopulationStats& stats, int max_species);
void freePopulationStats(PopulationStats& stats);

#endif