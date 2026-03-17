#ifndef CREATURE_MOVEMENT_CUH
#define CREATURE_MOVEMENT_CUH

#include <cuda_runtime.h>
#include "creature_common.cuh"
#include "../neural/neural_common.cuh"
#include "../core/constants.cuh"

void launchMovementKernel(
    CreatureData& creatures,
    const CreatureActions& actions,
    const float* d_heightmap,
    int world_size,
    float dt,
    int num_creatures,
    cudaStream_t stream = 0
);

#endif