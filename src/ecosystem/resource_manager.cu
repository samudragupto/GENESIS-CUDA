#include "resource_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void vegetationCapacityKernel(
    float* __restrict__ capacity,
    const float* __restrict__ heightmap,
    const float* __restrict__ temperature,
    const float* __restrict__ moisture,
    int world_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    float h = heightmap[idx];

    if (h < WATER_LEVEL) {
        capacity[idx] = 0.0f;
        return;
    }

    float temp = temperature[idx];
    float moist = moisture[idx];

    float temp_factor = 1.0f - fabsf(temp - 20.0f) / 30.0f;
    temp_factor = fmaxf(temp_factor, 0.0f);

    float moist_factor = fminf(moist / 0.5f, 1.0f);

    float altitude_factor = 1.0f - fmaxf((h - 0.7f) / 0.3f, 0.0f);
    altitude_factor = fmaxf(altitude_factor, 0.0f);

    capacity[idx] = temp_factor * moist_factor * altitude_factor;
}

__global__ void vegetationGrowthKernel(
    float* __restrict__ vegetation,
    const float* __restrict__ capacity,
    const float* __restrict__ dead_matter,
    const float* __restrict__ nutrients,
    const float* __restrict__ heightmap,
    const float* __restrict__ temperature,
    const float* __restrict__ moisture,
    float growth_rate,
    float min_temp,
    float max_temp,
    float min_moisture,
    float max_veg,
    float dt,
    int world_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    float h = heightmap[idx];
    if (h < WATER_LEVEL) {
        vegetation[idx] = 0.0f;
        return;
    }

    float temp = temperature[idx];
    if (temp < min_temp || temp > max_temp) {
        vegetation[idx] *= (1.0f - 0.001f * dt);
        return;
    }

    float moist = moisture[idx];
    if (moist < min_moisture) {
        vegetation[idx] *= (1.0f - 0.002f * dt);
        return;
    }

    float cap = capacity[idx];
    float current = vegetation[idx];
    float nutrient_boost = 1.0f + nutrients[idx] * 2.0f;
    float compost_boost = 1.0f + dead_matter[idx] * 1.5f;

    float logistic_growth = growth_rate * current * (1.0f - current / (cap * max_veg + 0.001f));
    logistic_growth *= nutrient_boost * compost_boost;

    float spontaneous = 0.0001f * cap * moist * dt;

    vegetation[idx] = fmaxf(current + (logistic_growth + spontaneous) * dt, 0.0f);
    vegetation[idx] = fminf(vegetation[idx], max_veg);
}

__global__ void vegetationSpreadKernel(
    float* __restrict__ vegetation,
    const float* __restrict__ heightmap,
    float spread_rate,
    float dt,
    int world_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= world_size - 1 || y < 1 || y >= world_size - 1) return;

    int idx = y * world_size + x;
    if (heightmap[idx] < WATER_LEVEL) return;

    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int ni = (y + dy) * world_size + (x + dx);
            if (heightmap[ni] >= WATER_LEVEL) {
                sum += vegetation[ni];
                count++;
            }
        }
    }

    if (count > 0) {
        float avg_neighbor = sum / (float)count;
        float current = vegetation[idx];
        float spread = (avg_neighbor - current) * spread_rate * dt;
        vegetation[idx] = fmaxf(current + spread, 0.0f);
    }
}

__global__ void deadMatterDecayKernel(
    float* __restrict__ dead_matter,
    float* __restrict__ nutrients,
    const float* __restrict__ temperature,
    float decay_rate,
    float dt,
    int world_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;

    float temp = temperature[idx];
    float temp_factor = fmaxf(fminf((temp - 5.0f) / 25.0f, 2.0f), 0.1f);

    float dm = dead_matter[idx];
    float decay = dm * decay_rate * temp_factor * dt;

    dead_matter[idx] = fmaxf(dm - decay, 0.0f);
    nutrients[idx] += decay * 0.6f;
}

__global__ void nutrientDiffusionKernel(
    float* __restrict__ nutrients,
    float diffusion_rate,
    float dt,
    int world_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= world_size - 1 || y < 1 || y >= world_size - 1) return;

    int idx = y * world_size + x;

    float c = nutrients[idx];
    float n = nutrients[(y - 1) * world_size + x];
    float s = nutrients[(y + 1) * world_size + x];
    float e = nutrients[y * world_size + (x + 1)];
    float w = nutrients[y * world_size + (x - 1)];

    float laplacian = (n + s + e + w - 4.0f * c);
    nutrients[idx] = fmaxf(c + diffusion_rate * laplacian * dt, 0.0f);

    float decay = 0.0001f * dt;
    nutrients[idx] *= (1.0f - decay);
}

void launchComputeVegetationCapacity(
    EcosystemGridData& eco_grid,
    const float* d_heightmap,
    const float* d_temperature,
    const float* d_moisture,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((eco_grid.world_size + 15) / 16, (eco_grid.world_size + 15) / 16);
    vegetationCapacityKernel<<<grid, block, 0, stream>>>(
        eco_grid.d_vegetation_capacity, d_heightmap,
        d_temperature, d_moisture, eco_grid.world_size
    );
}

void launchVegetationGrowth(
    EcosystemGridData& eco_grid,
    const float* d_heightmap,
    const float* d_temperature,
    const float* d_moisture,
    const EcosystemParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((eco_grid.world_size + 15) / 16, (eco_grid.world_size + 15) / 16);
    vegetationGrowthKernel<<<grid, block, 0, stream>>>(
        eco_grid.d_vegetation, eco_grid.d_vegetation_capacity,
        eco_grid.d_dead_matter, eco_grid.d_mineral_nutrients,
        d_heightmap, d_temperature, d_moisture,
        params.vegetation_growth_rate,
        params.min_growth_temperature,
        params.max_growth_temperature,
        params.min_growth_moisture,
        params.max_vegetation,
        params.dt, eco_grid.world_size
    );
}

void launchVegetationSpread(
    EcosystemGridData& eco_grid,
    const float* d_heightmap,
    const EcosystemParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((eco_grid.world_size + 15) / 16, (eco_grid.world_size + 15) / 16);
    vegetationSpreadKernel<<<grid, block, 0, stream>>>(
        eco_grid.d_vegetation, d_heightmap,
        params.vegetation_spread_rate, params.dt,
        eco_grid.world_size
    );
}

void launchDeadMatterDecay(
    EcosystemGridData& eco_grid,
    const float* d_temperature,
    const EcosystemParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((eco_grid.world_size + 15) / 16, (eco_grid.world_size + 15) / 16);
    deadMatterDecayKernel<<<grid, block, 0, stream>>>(
        eco_grid.d_dead_matter, eco_grid.d_mineral_nutrients,
        d_temperature, params.dead_matter_decay_rate,
        params.dt, eco_grid.world_size
    );
}

void launchNutrientDiffusion(
    EcosystemGridData& eco_grid,
    const EcosystemParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((eco_grid.world_size + 15) / 16, (eco_grid.world_size + 15) / 16);
    nutrientDiffusionKernel<<<grid, block, 0, stream>>>(
        eco_grid.d_mineral_nutrients,
        params.nutrient_diffusion_rate,
        params.dt, eco_grid.world_size
    );
}