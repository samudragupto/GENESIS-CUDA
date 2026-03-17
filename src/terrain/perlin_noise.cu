#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "perlin_noise.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include <curand_kernel.h>
#include <cstdlib>

__constant__ int d_terrain_permutation[512];

__device__ __forceinline__ float mix_lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ __forceinline__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ __forceinline__ float gradDot2D(int hash, float x, float y) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : h == 12 || h == 14 ? x : 0.0f;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__device__ float perlin2D(float x, float y) {
    int X = ((int)floorf(x)) & 255;
    int Y = ((int)floorf(y)) & 255;

    x -= floorf(x);
    y -= floorf(y);

    float u = fade(x);
    float v = fade(y);

    int A = d_terrain_permutation[X] + Y;
    int B = d_terrain_permutation[(X + 1) & 255] + Y; 

    int aa = d_terrain_permutation[A & 255];          
    int ab = d_terrain_permutation[(A + 1) & 255];
    int ba = d_terrain_permutation[B & 255];
    int bb = d_terrain_permutation[(B + 1) & 255];

    float x1 = mix_lerp(gradDot2D(aa, x, y), gradDot2D(ba, x - 1.0f, y), u);
    float x2 = mix_lerp(gradDot2D(ab, x, y - 1.0f), gradDot2D(bb, x - 1.0f, y - 1.0f), u);

    return (mix_lerp(x1, x2, v) + 1.0f) * 0.5f;
}

__global__ void perlinHeightmapKernel(
    float* __restrict__ heightmap,
    int world_size,
    int octaves,
    float lacunarity,
    float persistence
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    float amplitude = 1.0f;
    float frequency = TERRAIN_NOISE_FREQ;
    float noiseHeight = 0.0f;
    float max_possible = 0.0f;

    for (int i = 0; i < octaves; i++) {
        float sampleX = (float)x * frequency;
        float sampleY = (float)y * frequency;

        float perlinValue = perlin2D(sampleX, sampleY);
        noiseHeight += perlinValue * amplitude;
        max_possible += amplitude;

        amplitude *= persistence;
        frequency *= lacunarity;
    }

    heightmap[y * world_size + x] = noiseHeight / max_possible;
}

void launchPerlinHeightmap(
    float* d_heightmap,
    int world_size,
    int octaves,
    float lacunarity,
    float persistence,
    unsigned int seed,
    cudaStream_t stream
) {
    // Generate the permutation table on the CPU
    int h_perm[512];
    for (int i = 0; i < 256; i++) {
        h_perm[i] = i;
    }
    
    // Shuffle based on the provided terrain seed
    srand(seed);
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = h_perm[i];
        h_perm[i] = h_perm[j];
        h_perm[j] = tmp;
    }
    
    // Duplicate the array to avoid out-of-bounds math during noise generation
    for (int i = 0; i < 256; i++) {
        h_perm[256 + i] = h_perm[i];
    }

    // Copy to the LOCAL constant memory variable in this file
    CUDA_CHECK(cudaMemcpyToSymbol(d_terrain_permutation, h_perm, 512 * sizeof(int)));

    // Launch the kernel
    dim3 block(16, 16);
    dim3 grid(computeGrid2D(world_size, world_size, 16, 16));
    
    perlinHeightmapKernel<<<grid, block, 0, stream>>>(
        d_heightmap, world_size, octaves, lacunarity, persistence
    );
    KERNEL_CHECK();
}