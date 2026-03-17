#ifndef CREATURE_SPAWNER_CUH
#define CREATURE_SPAWNER_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "creature_common.cuh"

void launchInitialSpawn(
    CreatureData& creatures,
    int num_to_spawn,
    const float* d_heightmap,
    int world_size,
    curandState* d_rng_states,
    cudaStream_t stream = 0
);

void launchSpawnOffspring(
    CreatureData& creatures,
    const InteractionBuffers& interaction,
    int current_alive,
    int max_spawn,
    curandState* d_rng_states,
    int* d_species_counter,
    float speciation_threshold,
    cudaStream_t stream = 0
);

#endif