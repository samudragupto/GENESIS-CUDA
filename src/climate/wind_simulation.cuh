#ifndef WIND_SIMULATION_CUH
#define WIND_SIMULATION_CUH

#include <cuda_runtime.h>
#include "climate_common.cuh"

void launchPressureComputeKernel(
    float* d_pressure,
    const float* d_temperature,
    int world_size,
    float base_pressure,
    cudaStream_t stream = 0
);

void launchWindUpdateKernel(
    float* d_wind_x,
    float* d_wind_y,
    const float* d_pressure,
    int world_size,
    float wind_damping,
    float coriolis_factor,
    float dt,
    cudaStream_t stream = 0
);

#endif