#ifndef HEAT_DIFFUSION_CUH
#define HEAT_DIFFUSION_CUH

#include <cuda_runtime.h>
#include "climate_common.cuh"

void launchHeatDiffusionKernel(
    float* d_temperature,
    const float* d_solar_flux,
    const float* d_heightmap,
    int world_size,
    float thermal_conductivity,
    float dt,
    cudaStream_t stream = 0
);

#endif