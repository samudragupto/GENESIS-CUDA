#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "humidity_transport.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__device__ __forceinline__ float calc_saturation_humidity(float temperature) {
    return 0.01f * expf(0.05f * temperature);
}

__global__ void evaporationKernel(
    float* __restrict__ humidity,
    const float* __restrict__ temperature,
    const float* __restrict__ moisture,
    const float* __restrict__ heightmap,
    const float* __restrict__ wind_x,
    const float* __restrict__ wind_y,
    int world_size,
    float evaporation_rate,
    float water_level,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    
    float t = temperature[idx];
    if (t <= 0.0f) return; 
    
    float h = heightmap[idx];
    float m = moisture[idx];
    float current_hum = humidity[idx];
    
    float wx = wind_x[idx];
    float wy = wind_y[idx];
    float wind_speed = sqrtf(wx * wx + wy * wy);
    
    float e_sat = calc_saturation_humidity(t);
    float deficit = fmaxf(e_sat - current_hum, 0.0f);
    
    float windFactor = 1.0f + 0.1f * wind_speed;
    float surfaceFactor = (h < water_level) ? 1.0f : fminf(m, 1.0f);
    
    float evap = evaporation_rate * deficit * windFactor * surfaceFactor * dt;
    
    humidity[idx] = fminf(current_hum + evap, 1.0f);
}

__global__ void humidityAdvectionKernel(
    float* __restrict__ humidity,
    const float* __restrict__ wind_x,
    const float* __restrict__ wind_y,
    int world_size,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    
    float wx = wind_x[idx];
    float wy = wind_y[idx];
    
    float src_xf = (float)x - wx * dt;
    float src_yf = (float)y - wy * dt;
    
    int src_x = min(max((int)src_xf, 0), world_size - 1);
    int src_y = min(max((int)src_yf, 0), world_size - 1);
    int src_x1 = min(src_x + 1, world_size - 1);
    int src_y1 = min(src_y + 1, world_size - 1);
    
    float fx = src_xf - (float)src_x;
    float fy = src_yf - (float)src_y;
    
    float h00 = humidity[src_y * world_size + src_x];
    float h10 = humidity[src_y * world_size + src_x1];
    float h01 = humidity[src_y1 * world_size + src_x];
    float h11 = humidity[src_y1 * world_size + src_x1];
    
    float h0 = h00 * (1.0f - fx) + h10 * fx;
    float h1 = h01 * (1.0f - fx) + h11 * fx;
    float interpolated_hum = h0 * (1.0f - fy) + h1 * fy;
    
    humidity[idx] = interpolated_hum;
}

void launchEvaporationKernel(
    float* d_humidity,
    const float* d_temperature,
    const float* d_moisture,
    const float* d_heightmap,
    int world_size,
    float evaporation_rate,
    float water_level,
    float dt,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    
    float* d_wind_x = nullptr; 
    float* d_wind_y = nullptr; 

    CUDA_CHECK(cudaMalloc(&d_wind_x, world_size * world_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wind_y, world_size * world_size * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_wind_x, 0, world_size * world_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_wind_y, 0, world_size * world_size * sizeof(float), stream));

    evaporationKernel<<<grid, block, 0, stream>>>(
        d_humidity, d_temperature, d_moisture, d_heightmap,
        d_wind_x, d_wind_y, world_size, evaporation_rate, water_level, dt
    );
    KERNEL_CHECK();
    
    cudaFree(d_wind_x);
    cudaFree(d_wind_y);
}

void launchHumidityAdvectionKernel(
    float* d_humidity,
    const float* d_wind_x,
    const float* d_wind_y,
    int world_size,
    float dt,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    humidityAdvectionKernel<<<grid, block, 0, stream>>>(
        d_humidity, d_wind_x, d_wind_y, world_size, dt
    );
    KERNEL_CHECK();
}