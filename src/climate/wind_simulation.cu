#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "wind_simulation.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void pressureComputeKernel(
    float* __restrict__ pressure,
    const float* __restrict__ temperature,
    int world_size,
    float base_pressure,
    float pressure_coeff
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    float temp = temperature[idx];
    
    pressure[idx] = base_pressure - (temp * pressure_coeff);
}

__global__ void windUpdateKernel(
    float* __restrict__ wind_x,
    float* __restrict__ wind_y,
    const float* __restrict__ pressure,
    int world_size,
    float wind_damping,
    float coriolis_param,
    float friction_coeff,
    float max_wind,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= world_size - 1 || y < 1 || y >= world_size - 1) return;

    int idx = y * world_size + x;
    
    float dp_dx = (pressure[y * world_size + (x + 1)] - pressure[y * world_size + (x - 1)]) * 0.5f;
    float dp_dy = (pressure[(y + 1) * world_size + x] - pressure[(y - 1) * world_size + x]) * 0.5f;

    float wx = wind_x[idx];
    float wy = wind_y[idx];

    float lat = ((float)y / (float)world_size) * CUDART_PI_F - (CUDART_PI_F * 0.5f);
    float f_coriolis = coriolis_param * sinf(lat);

    float rho_air = 1.225f; 
    float accelX = -dp_dx / rho_air;
    float accelY = -dp_dy / rho_air;

    float new_wx = wx + (accelX + f_coriolis * wy - friction_coeff * wx) * dt;
    float new_wy = wy + (accelY - f_coriolis * wx - friction_coeff * wy) * dt;

    new_wx *= wind_damping;
    new_wy *= wind_damping;

    float speed = sqrtf(new_wx * new_wx + new_wy * new_wy);
    if (speed > max_wind) {
        new_wx = (new_wx / speed) * max_wind;
        new_wy = (new_wy / speed) * max_wind;
    }

    wind_x[idx] = new_wx;
    wind_y[idx] = new_wy;
}

void launchPressureComputeKernel(
    float* d_pressure,
    const float* d_temperature,
    int world_size,
    float base_pressure,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    
    float pressure_coeff = 50.0f;

    pressureComputeKernel<<<grid, block, 0, stream>>>(
        d_pressure, d_temperature, world_size, base_pressure, pressure_coeff
    );
    KERNEL_CHECK();
}

void launchWindUpdateKernel(
    float* d_wind_x,
    float* d_wind_y,
    const float* d_pressure,
    int world_size,
    float wind_damping,
    float coriolis_factor,
    float dt,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    
    float friction_coeff = 0.05f;
    float max_wind = 40.0f;

    windUpdateKernel<<<grid, block, 0, stream>>>(
        d_wind_x, d_wind_y, d_pressure,
        world_size, wind_damping, coriolis_factor, 
        friction_coeff, max_wind, dt
    );
    KERNEL_CHECK();
}