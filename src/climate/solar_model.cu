#include "solar_model.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void solarRadiationKernel(
    float* __restrict__ solar_flux,
    const float* __restrict__ heightmap,
    int world_size,
    float solar_constant,
    float sun_angle,
    float axial_tilt,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;

    float lat = ((float)y / (float)world_size) * CUDART_PI_F - (CUDART_PI_F * 0.5f);
    float declination = axial_tilt * (CUDART_PI_F / 180.0f) * cosf(sun_angle);

    float zenith = sinf(lat) * sinf(declination) + cosf(lat) * cosf(declination) * cosf(sun_angle);
    zenith = fmaxf(zenith, 0.0f);

    float h = heightmap[idx];
    float albedo;
    if (h < WATER_LEVEL) {
        albedo = 0.06f;
    } else if (h > 0.8f) {
        albedo = 0.8f;
    } else if (h > 0.6f) {
        albedo = 0.3f * 1.2f;
    } else {
        albedo = 0.3f;
    }

    float airMass = (zenith > 0.01f) ? (1.0f / zenith) : 100.0f;
    float atmosphericTransmission = expf(-0.1f * airMass);

    float radiation = solar_constant * zenith * atmosphericTransmission * (1.0f - albedo);

    solar_flux[idx] = radiation * dt;
}

void launchSolarRadiationKernel(
    float* d_solar_flux,
    const float* d_heightmap,
    int world_size,
    float solar_constant,
    float sun_angle,
    float axial_tilt,
    float dt,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    solarRadiationKernel<<<grid, block, 0, stream>>>(
        d_solar_flux, d_heightmap, world_size,
        solar_constant, sun_angle, axial_tilt, dt
    );
    KERNEL_CHECK();
}