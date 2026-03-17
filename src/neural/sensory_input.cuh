#ifndef SENSORY_INPUT_CUH
#define SENSORY_INPUT_CUH

#include <cuda_runtime.h>
#include "neural_common.cuh"
#include "../creatures/creature_common.cuh"

struct SensoryContext {
    const CreatureData* creatures;
    WorldDataPtrs world;
    SpatialGridPtrs grid;
    int num_creatures;
};

void launchSensoryInputKernel(
    float* d_sensory_output,
    const SensoryContext& ctx,
    cudaStream_t stream = 0
);

#endif