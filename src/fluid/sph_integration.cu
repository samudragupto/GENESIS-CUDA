#include "sph_kernels.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void k_integrate(
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    const float3* __restrict__ forces,
    const float* __restrict__ densities,
    int particleCount,
    SPHSimParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    
    float3 pos = positions[idx];
    float3 vel = velocities[idx];
    float3 f = forces[idx];
    float rho = densities[idx];
    
    float3 acc = make_float3(f.x / rho, f.y / rho, f.z / rho);
    
    vel.x += acc.x * params.dt;
    vel.y += acc.y * params.dt;
    vel.z += acc.z * params.dt;
    
    pos.x += vel.x * params.dt;
    pos.y += vel.y * params.dt;
    pos.z += vel.z * params.dt;
    
    positions[idx] = pos;
    velocities[idx] = vel;
}

void launchIntegrate(
    float3* positions, float3* velocities, const float3* forces,
    const float* densities, const uint32_t* sortedIndices,
    int particleCount, cudaStream_t stream
) {
    if (particleCount <= 0) return;
    SPHSimParams params; // Dummy
    int block = 256;
    int grid = (particleCount + block - 1) / block;
    k_integrate<<<grid, block, 0, stream>>>(positions, velocities, forces, densities, particleCount, params);
    KERNEL_CHECK();
}