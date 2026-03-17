#ifndef CREATURE_INTERACTION_CUH
#define CREATURE_INTERACTION_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "creature_common.cuh"
#include "../neural/neural_common.cuh"
#include "../spatial/spatial_hash.cuh"
#include "../core/constants.cuh"

struct InteractionParams {
    float eat_radius;
    float attack_radius;
    float mate_radius;
    float eat_energy_gain;
    float attack_damage;
    float attack_energy_cost;
    float mate_energy_cost;
    float cannibalism_penalty;
};

void launchFeedingKernel(
    CreatureData& creatures,
    const CreatureActions& actions,
    float* d_vegetation,
    int world_size,
    const InteractionParams& params,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchCombatKernel(
    CreatureData& creatures,
    const CreatureActions& actions,
    const int* d_cell_start,
    const int* d_cell_end,
    const int* d_sorted_indices,
    int grid_size,
    float cell_size,
    const InteractionParams& params,
    int num_creatures,
    curandState* d_rng,
    cudaStream_t stream = 0
);

void launchMatingKernel(
    CreatureData& creatures,
    const CreatureActions& actions,
    InteractionBuffers& interaction,
    const int* d_cell_start,
    const int* d_cell_end,
    const int* d_sorted_indices,
    int grid_size,
    float cell_size,
    const InteractionParams& params,
    int num_creatures,
    curandState* d_rng,
    cudaStream_t stream = 0
);

#endif