#ifndef FOOD_WEB_CUH
#define FOOD_WEB_CUH

#include <cuda_runtime.h>
#include "ecosystem_common.cuh"
#include "../creatures/creature_common.cuh"

struct FoodWebData {
    float* d_trophic_level;
    float* d_energy_intake_rate;
    float* d_energy_output_rate;
    int*   d_predator_count;
    int*   d_prey_count;
    int    max_species;
};

void allocateFoodWebData(FoodWebData& fw, int max_species);
void freeFoodWebData(FoodWebData& fw);

void launchComputeTrophicLevels(
    FoodWebData& food_web,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchEnergyFlowKernel(
    FoodWebData& food_web,
    const CreatureData& creatures,
    EcosystemGridData& eco_grid,
    int num_creatures,
    int world_size,
    float dt,
    cudaStream_t stream = 0
);

void launchDeadMatterDeposit(
    const CreatureData& creatures,
    EcosystemGridData& eco_grid,
    int num_creatures,
    cudaStream_t stream = 0
);

#endif