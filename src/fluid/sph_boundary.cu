#include "sph_kernels.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void k_applyBoundary(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    int particleCount,
    SPHSimParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    
    float3 pos = positions[idx];
    float3 vel = velocities[idx];
    
    float restitution = 0.5f;
    float margin = 0.01f;
    
    if (pos.x < params.worldMinX + margin) { pos.x = params.worldMinX + margin; vel.x *= -restitution; }
    if (pos.x > params.worldMaxX - margin) { pos.x = params.worldMaxX - margin; vel.x *= -restitution; }
    
    if (pos.y < params.worldMinY + margin) { pos.y = params.worldMinY + margin; vel.y *= -restitution; }
    if (pos.y > params.worldMaxY - margin) { pos.y = params.worldMaxY - margin; vel.y *= -restitution; }
    
    if (pos.z < params.worldMinZ + margin) { pos.z = params.worldMinZ + margin; vel.z *= -restitution; }
    if (pos.z > params.worldMaxZ - margin) { pos.z = params.worldMaxZ - margin; vel.z *= -restitution; }
    
    positions[idx] = pos;
    velocities[idx] = vel;
}

void launchApplyBoundary(
    float3* positions, float3* velocities, const float* d_heightmap,
    int particleCount, cudaStream_t stream
) {
    if (particleCount <= 0) return;
    SPHSimParams params; // Dummy
    int block = 256;
    int grid = (particleCount + block - 1) / block;
    k_applyBoundary<<<grid, block, 0, stream>>>(positions, velocities, particleCount, params);
    KERNEL_CHECK();
}