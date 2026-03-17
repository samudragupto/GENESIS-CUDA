#include "sph_kernels.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void k_fluidTerrainCoupling(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    const float* __restrict__ heightmap,
    int particleCount,
    SPHSimParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    
    float3 pos = positions[idx];
    float3 vel = velocities[idx];
    
    int gx = (int)pos.x;
    int gy = (int)pos.z;
    int ws = (int)(params.worldMaxX - params.worldMinX);
    
    if (gx >= 0 && gx < ws && gy >= 0 && gy < ws) {
        float h = heightmap[gy * ws + gx] * 100.0f; // Scale height appropriately
        if (pos.y < h) {
            pos.y = h;
            vel.y *= -0.5f;
            vel.x *= 0.8f;
            vel.z *= 0.8f;
        }
    }
    
    positions[idx] = pos;
    velocities[idx] = vel;
}

void launchFluidTerrainCoupling(
    float3* positions, float3* velocities, const float* d_heightmap,
    int particleCount, cudaStream_t stream
) {
    if (particleCount <= 0) return;
    SPHSimParams params; // Dummy
    int block = 256;
    int grid = (particleCount + block - 1) / block;
    k_fluidTerrainCoupling<<<grid, block, 0, stream>>>(positions, velocities, d_heightmap, particleCount, params);
    KERNEL_CHECK();
}