#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "biome_classifier.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void biomeClassificationKernel(
    int* __restrict__ biome_map,
    const float* __restrict__ heightmap,
    const float* __restrict__ moisture_map,
    int world_size,
    float sea_level
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    
    float h = heightmap[idx];
    float m = moisture_map ? moisture_map[idx] : 0.5f;

    int biome = 0; // Default

    if (h < sea_level) {
        if (h < sea_level - 0.2f) {
            biome = BIOME_DEEP_OCEAN;
        } else {
            biome = BIOME_OCEAN;
        }
    } else if (h < sea_level + 0.05f) {
        if (m > 0.6f) biome = BIOME_SWAMP;
        else biome = BIOME_BEACH;
    } else if (h > 0.8f) {
        biome = BIOME_SNOW_PEAK;
    } else if (h > 0.6f) {
        if (h > 0.75f && m < 0.3f) biome = BIOME_SNOW_PEAK;
        else biome = BIOME_MOUNTAIN;
    } else {
        float temp = 1.0f - (h - sea_level) / (0.6f - sea_level);
        
        if (temp < 0.3f) {
            biome = (m > 0.4f) ? BIOME_TAIGA : BIOME_TUNDRA;
        } else if (temp < 0.6f) {
            if (m < 0.25f)       biome = BIOME_GRASSLAND;
            else if (m < 0.6f)   biome = BIOME_FOREST;
            else                 biome = BIOME_TAIGA;
        } else {
            if (m < 0.2f)        biome = BIOME_DESERT;
            else if (m < 0.45f)  biome = BIOME_SAVANNA;
            else if (m < 0.7f)   biome = BIOME_FOREST;
            else                 biome = BIOME_RAINFOREST;
        }
    }
    
    biome_map[idx] = biome;
}

void launchBiomeClassification(
    int* d_biome_map,
    const float* d_heightmap,
    const float* d_moisture_map,
    int world_size,
    float sea_level,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((world_size + 15) / 16, (world_size + 15) / 16);
    
    biomeClassificationKernel<<<grid, block, 0, stream>>>(
        d_biome_map, d_heightmap, d_moisture_map, world_size, sea_level
    );
    KERNEL_CHECK();
}