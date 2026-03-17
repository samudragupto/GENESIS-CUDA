#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "precipitation.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void cloudFormationKernel(
    float* __restrict__ cloud_cover,
    const float* __restrict__ humidity,
    const float* __restrict__ temperature,
    int world_size,
    float condensation_threshold,
    float cloud_formation_rate,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    float h = humidity[idx];
    float t = temperature[idx];

    float saturation = 0.1f + t * 0.01f;
    float excess = h - saturation * condensation_threshold;

    if (excess > 0.0f) {
        cloud_cover[idx] += excess * cloud_formation_rate * dt;
        cloud_cover[idx] = fminf(cloud_cover[idx], 1.0f);
    } else {
        cloud_cover[idx] *= (1.0f - 0.01f * dt);
        cloud_cover[idx] = fmaxf(cloud_cover[idx], 0.0f);
    }
}

__global__ void rainfallKernel(
    float* __restrict__ moisture,
    float* __restrict__ cloud_cover,
    float* __restrict__ humidity,
    const float* __restrict__ temperature,
    int world_size,
    float rainfall_rate,
    float snow_temperature,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    float cloud = cloud_cover[idx];

    if (cloud > 0.3f) {
        float rain_amount = (cloud - 0.3f) * rainfall_rate * dt;
        float temp = temperature[idx];

        if (temp > snow_temperature) {
            moisture[idx] += rain_amount;
        } else {
            moisture[idx] += rain_amount * 0.5f;
        }

        cloud_cover[idx] -= rain_amount * 0.5f;
        cloud_cover[idx] = fmaxf(cloud_cover[idx], 0.0f);

        humidity[idx] -= rain_amount * 0.3f;
        humidity[idx] = fmaxf(humidity[idx], 0.0f);
    }

    moisture[idx] = fmaxf(fminf(moisture[idx], 1.0f), 0.0f);
}

void launchCloudFormationKernel(
    float* d_cloud_cover,
    const float* d_humidity,
    const float* d_temperature,
    int world_size,
    const ClimateParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    cloudFormationKernel<<<grid, block, 0, stream>>>(
        d_cloud_cover, d_humidity, d_temperature,
        world_size, params.condensation_threshold,
        params.cloud_formation_rate, params.dt
    );
    KERNEL_CHECK();
}

void launchRainfallKernel(
    float* d_moisture,
    float* d_cloud_cover,
    float* d_humidity,
    const float* d_temperature,
    int world_size,
    const ClimateParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    rainfallKernel<<<grid, block, 0, stream>>>(
        d_moisture, d_cloud_cover, d_humidity,
        d_temperature, world_size,
        params.rainfall_rate, params.snow_temperature, params.dt
    );
    KERNEL_CHECK();
}