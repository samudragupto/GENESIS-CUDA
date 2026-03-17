#ifndef HUMIDITY_TRANSPORT_CUH
#define HUMIDITY_TRANSPORT_CUH

#include <cuda_runtime.h>
#include "climate_common.cuh"

void launchEvaporationKernel(
    float* d_humidity,
    const float* d_temperature,
    const float* d_moisture,
    const float* d_heightmap,
    int world_size,
    float evaporation_rate,
    float water_level,
    float dt,
    cudaStream_t stream = 0
);

void launchHumidityAdvectionKernel(
    float* d_humidity,
    const float* d_wind_x,
    const float* d_wind_y,
    int world_size,
    float dt,
    cudaStream_t stream = 0
);

#endif