#ifndef SOLAR_MODEL_CUH
#define SOLAR_MODEL_CUH

#include <cuda_runtime.h>
#include "climate_common.cuh"

void launchSolarRadiationKernel(
    float* d_solar_flux,
    const float* d_heightmap,
    int world_size,
    float solar_constant,
    float sun_angle,
    float axial_tilt,
    float dt,
    cudaStream_t stream = 0
);

#endif