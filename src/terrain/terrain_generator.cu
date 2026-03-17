#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "terrain_generator.cuh"
#include "perlin_noise.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void normalizeTerrainKernel(
    float* __restrict__ heightmap,
    int world_size,
    float max_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    
    float h = heightmap[idx];
    
    h = (h + 1.0f) * 0.5f;
    
    float terracing = floorf(h * 10.0f) / 10.0f;
    h = h * 0.7f + terracing * 0.3f;
    
    heightmap[idx] = h * max_height;
}

namespace genesis {

void TerrainGenerator::generate(unsigned int seed, cudaStream_t stream) {
    GpuTimer timer;

    launchPerlinHeightmap(
        d_heightmap_,
        width_,
        octaves_,
        lacunarity_,
        persistence_,
        seed,
        stream
    );

    dim3 block(16, 16);
    dim3 grid((width_ + 15) / 16, (height_ + 15) / 16);
    
    normalizeTerrainKernel<<<grid, block, 0, stream>>>(
        d_heightmap_,
        width_,
        TERRAIN_MAX_HEIGHT
    );
    KERNEL_CHECK();
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("[TerrainGenerator] Terrain generated in %.2f ms\n", timer.elapsedMs());
}

} // namespace genesis