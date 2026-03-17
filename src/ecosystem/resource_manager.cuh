#ifndef RESOURCE_MANAGER_CUH
#define RESOURCE_MANAGER_CUH

#include <cuda_runtime.h>
#include "ecosystem_common.cuh"

void launchVegetationGrowth(
    EcosystemGridData& eco_grid,
    const float* d_heightmap,
    const float* d_temperature,
    const float* d_moisture,
    const EcosystemParams& params,
    cudaStream_t stream = 0
);

void launchVegetationSpread(
    EcosystemGridData& eco_grid,
    const float* d_heightmap,
    const EcosystemParams& params,
    cudaStream_t stream = 0
);

void launchDeadMatterDecay(
    EcosystemGridData& eco_grid,
    const float* d_temperature,
    const EcosystemParams& params,
    cudaStream_t stream = 0
);

void launchNutrientDiffusion(
    EcosystemGridData& eco_grid,
    const EcosystemParams& params,
    cudaStream_t stream = 0
);

void launchComputeVegetationCapacity(
    EcosystemGridData& eco_grid,
    const float* d_heightmap,
    const float* d_temperature,
    const float* d_moisture,
    cudaStream_t stream = 0
);

#endif