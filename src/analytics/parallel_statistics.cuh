#ifndef PARALLEL_STATISTICS_CUH
#define PARALLEL_STATISTICS_CUH

#include <cuda_runtime.h>
#include "../creatures/creature_common.cuh"
#include "analytics_common.cuh"

struct ReductionBuffers {
    float* d_energy_sum;
    float* d_health_sum;
    float* d_age_sum;
    int*   d_alive_count;
    int*   d_species_set;
    int*   d_unique_species_count;
    float* d_block_results;
    int    max_blocks;
};

void allocateReductionBuffers(ReductionBuffers& buf, int max_blocks, int max_species);
void freeReductionBuffers(ReductionBuffers& buf);

void launchComputeBasicStats(
    const CreatureData& creatures,
    ReductionBuffers& reduction,
    AnalyticsSnapshot* d_snapshot,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchComputeSpeciesCount(
    const CreatureData& creatures,
    ReductionBuffers& reduction,
    int num_creatures,
    int max_species,
    cudaStream_t stream = 0
);

void launchPerSpeciesStats(
    const CreatureData& creatures,
    int* d_species_pop,
    float* d_species_energy,
    float* d_species_fitness,
    int num_creatures,
    int max_species,
    cudaStream_t stream = 0
);

#endif