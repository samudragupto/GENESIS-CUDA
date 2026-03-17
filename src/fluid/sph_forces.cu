#include "sph_kernels.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void k_computeForces(
    const float3* __restrict__ sortedPositions,
    const float3* __restrict__ sortedVelocities,
    const float* __restrict__ densities,
    const float* __restrict__ pressures,
    float3* __restrict__ forces,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,
    int particleCount,
    SPHSimParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    
    float3 pos_i = sortedPositions[idx];
    float3 vel_i = sortedVelocities[idx];
    float rho_i = densities[idx];
    float p_i = pressures[idx];
    
    int3 cell_i = getCell(pos_i, params);
    float3 force = make_float3(0, 0, 0);
    
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int3 nc = make_int3(cell_i.x+dx, cell_i.y+dy, cell_i.z+dz);
                if (nc.x<0||nc.x>=params.gridDimX||
                    nc.y<0||nc.y>=params.gridDimY||
                    nc.z<0||nc.z>=params.gridDimZ) continue;
                    
                uint32_t ch = hashCell(nc, params);
                uint32_t start = cellStart[ch];
                if (start == 0xFFFFFFFF) continue;
                uint32_t end = cellEnd[ch];
                
                for (uint32_t j = start; j < end; j++) {
                    if (idx == j) continue;
                    
                    float3 pos_j = sortedPositions[j];
                    float3 diff = make_float3(pos_i.x-pos_j.x, pos_i.y-pos_j.y, pos_i.z-pos_j.z);
                    float r2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
                    
                    if (r2 < params.smoothingRadiusSq && r2 > 0.0001f) {
                        float r = sqrtf(r2);
                        float rho_j = densities[j];
                        float p_j = pressures[j];
                        
                        // Pressure force
                        float pTerm = params.particleMass * (p_i + p_j) / (2.0f * rho_j);
                        float gradW = kernelSpikyGrad(r, params.smoothingRadius, params.spikyGradCoeff);
                        force.x -= pTerm * gradW * diff.x;
                        force.y -= pTerm * gradW * diff.y;
                        force.z -= pTerm * gradW * diff.z;
                        
                        // Viscosity force
                        float3 vel_j = sortedVelocities[j];
                        float vTerm = params.viscosity * params.particleMass / rho_j;
                        float lapW = kernelViscLap(r, params.smoothingRadius, params.viscLapCoeff);
                        force.x += vTerm * (vel_j.x - vel_i.x) * lapW;
                        force.y += vTerm * (vel_j.y - vel_i.y) * lapW;
                        force.z += vTerm * (vel_j.z - vel_i.z) * lapW;
                    }
                }
            }
        }
    }
    
    force.y += params.gravity * rho_i; // Gravity
    forces[idx] = force;
}

void launchComputeForces(
    const float3* sortedPositions, const float3* sortedVelocities,
    const float* densities, const float* pressures, float3* forces,
    const uint32_t* cellStart, const uint32_t* cellEnd,
    const uint32_t* sortedIndices, int particleCount, cudaStream_t stream
) {
    if (particleCount <= 0) return;
    SPHSimParams params; // Dummy
    int block = 256;
    int grid = (particleCount + block - 1) / block;
    k_computeForces<<<grid, block, 0, stream>>>(
        sortedPositions, sortedVelocities, densities, pressures, forces,
        cellStart, cellEnd, particleCount, params
    );
    KERNEL_CHECK();
}