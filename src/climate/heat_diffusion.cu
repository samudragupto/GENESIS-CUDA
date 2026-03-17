#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "heat_diffusion.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void heatDiffusionKernel(
    float* __restrict__ temperature,
    const float* __restrict__ solar_flux,
    const float* __restrict__ heightmap,
    int world_size,
    float thermal_conductivity,
    float dt,
    float base_temp,
    float emissivity,
    float sh_water,
    float sh_land,
    float min_temp,
    float max_temp
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= world_size - 1 || y < 1 || y >= world_size - 1) return;

    int idx = y * world_size + x;
    
    float c = temperature[idx];
    float n = temperature[(y - 1) * world_size + x];
    float s = temperature[(y + 1) * world_size + x];
    float e = temperature[y * world_size + (x + 1)];
    float w = temperature[y * world_size + (x - 1)];

    float laplacian = (n + s + e + w - 4.0f * c);
    
    float h = heightmap[idx];
    float heat_capacity = (h < WATER_LEVEL) ? sh_water : sh_land;
    
    float elevation_cooling = (h > WATER_LEVEL) ? (h - WATER_LEVEL) * 15.0f : 0.0f;
    
    float radiative_loss = emissivity * 5.67e-8f * powf(c + 273.15f, 4.0f);
    radiative_loss = fminf(radiative_loss, 500.0f); 
    
    float incoming_heat = solar_flux[idx];
    
    float delta_T = (thermal_conductivity * laplacian * 100.0f + incoming_heat - radiative_loss) / heat_capacity;
    
    float new_temp = c + delta_T * dt - elevation_cooling * 0.001f * dt;
    
    new_temp = fmaxf(min_temp, fminf(new_temp, max_temp));
    
    temperature[idx] = new_temp;
}

void launchHeatDiffusionKernel(
    float* d_temperature,
    const float* d_solar_flux,
    const float* d_heightmap,
    int world_size,
    float thermal_conductivity,
    float dt,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    
    float base_temp = 20.0f;
    float emissivity = 0.95f;
    float sh_water = 4184.0f;
    float sh_land = 1000.0f;
    float min_temp = -50.0f;
    float max_temp = 60.0f;

    heatDiffusionKernel<<<grid, block, 0, stream>>>(
        d_temperature, d_solar_flux, d_heightmap,
        world_size, thermal_conductivity, dt,
        base_temp, emissivity, sh_water, sh_land, min_temp, max_temp
    );
    KERNEL_CHECK();
}