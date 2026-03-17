#ifndef POPULATION_CONTROL_CUH
#define POPULATION_CONTROL_CUH

#include <cuda_runtime.h>
#include "ecosystem_common.cuh"
#include "../creatures/creature_common.cuh"

void launchPopulationCensus(
    PopulationStats& stats,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchCarryingCapacityEnforcement(
    CreatureData& creatures,
    const PopulationStats& stats,
    const EcosystemGridData& eco_grid,
    int num_creatures,
    int global_max_population,
    cudaStream_t stream = 0
);

void launchSpeciesPopulationCount(
    PopulationStats& stats,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchComputeSpeciesAvgEnergy(
    PopulationStats& stats,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream = 0
);

#endif