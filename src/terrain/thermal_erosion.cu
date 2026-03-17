#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "thermal_erosion.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void thermalErosionKernel(
    float* __restrict__ heightmap,
    int world_size,
    float talus_angle
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= world_size - 1 || y < 1 || y >= world_size - 1) return;

    int idx = y * world_size + x;
    float h = heightmap[idx];

    float h_n = heightmap[(y - 1) * world_size + x];
    float h_s = heightmap[(y + 1) * world_size + x];
    float h_e = heightmap[y * world_size + (x + 1)];
    float h_w = heightmap[y * world_size + (x - 1)];

    float max_diff = 0.0f;
    int max_idx = -1;

    if (h - h_n > max_diff) { max_diff = h - h_n; max_idx = (y - 1) * world_size + x; }
    if (h - h_s > max_diff) { max_diff = h - h_s; max_idx = (y + 1) * world_size + x; }
    if (h - h_e > max_diff) { max_diff = h - h_e; max_idx = y * world_size + (x + 1); }
    if (h - h_w > max_diff) { max_diff = h - h_w; max_idx = y * world_size + (x - 1); }

    if (max_diff > talus_angle && max_idx != -1) {
        float transfer = max_diff * 0.1f;
        heightmap[idx] -= transfer;
        atomicAdd(&heightmap[max_idx], transfer);
    }
}

void launchThermalErosion(
    float* d_heightmap,
    int world_size,
    float talus_angle,
    int iterations,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(computeGrid2D(world_size, world_size, 16, 16));

    for (int i = 0; i < iterations; i++) {
        thermalErosionKernel<<<grid, block, 0, stream>>>(
            d_heightmap, world_size, talus_angle
        );
    }
    KERNEL_CHECK();
}