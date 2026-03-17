#ifndef PRECIPITATION_CUH
#define PRECIPITATION_CUH

#include <cuda_runtime.h>
#include "climate_common.cuh"

void launchCloudFormationKernel(
    float* d_cloud_cover,
    const float* d_humidity,
    const float* d_temperature,
    int world_size,
    const ClimateParams& params,
    cudaStream_t stream = 0
);

void launchRainfallKernel(
    float* d_moisture,
    float* d_cloud_cover,
    float* d_humidity,
    const float* d_temperature,
    int world_size,
    const ClimateParams& params,
    cudaStream_t stream = 0
);

#endif