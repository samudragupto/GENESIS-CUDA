#ifndef PERLIN_NOISE_CUH
#define PERLIN_NOISE_CUH

#include <cuda_runtime.h>

void launchPerlinHeightmap(
    float* d_heightmap,
    int world_size,
    int octaves,
    float lacunarity,
    float persistence,
    unsigned int seed,
    cudaStream_t stream = 0
);

#endif