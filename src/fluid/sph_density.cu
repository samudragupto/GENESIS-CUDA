#include "sph_kernels.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void k_computeHashes(
    const float3* __restrict__ positions,
    uint32_t* __restrict__ cellHashes,
    uint32_t* __restrict__ particleIndices,
    int particleCount,
    SPHSimParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    float3 pos = positions[idx];
    int3 cell = getCell(pos, params);
    cellHashes[idx] = hashCell(cell, params);
    particleIndices[idx] = (uint32_t)idx;
}

__global__ void k_findCellBounds(
    uint32_t* __restrict__ cellStart,
    uint32_t* __restrict__ cellEnd,
    const uint32_t* __restrict__ sortedHashes,
    int particleCount
) {
    extern __shared__ uint32_t sharedHash[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t hash = 0xFFFFFFFF;
    if (idx < particleCount) {
        hash = sortedHashes[idx];
        sharedHash[threadIdx.x + 1] = hash;
        if (idx > 0 && threadIdx.x == 0)
            sharedHash[0] = sortedHashes[idx - 1];
    }
    __syncthreads();
    if (idx < particleCount) {
        if (idx == 0 || hash != sharedHash[threadIdx.x]) {
            cellStart[hash] = idx;
            if (idx > 0) cellEnd[sharedHash[threadIdx.x]] = idx;
        }
        if (idx == particleCount - 1) cellEnd[hash] = idx + 1;
    }
}

__global__ void k_computeDensityPressure(
    const float3* __restrict__ sortedPositions,
    float* __restrict__ densities,
    float* __restrict__ pressures,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,
    int particleCount,
    SPHSimParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;
    
    float3 pos_i = sortedPositions[idx];
    int3 cell_i = getCell(pos_i, params);
    float density = 0.0f;
    
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
                    float3 pos_j = sortedPositions[j];
                    float3 diff = make_float3(pos_i.x-pos_j.x, pos_i.y-pos_j.y, pos_i.z-pos_j.z);
                    float r2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
                    density += params.particleMass * kernelPoly6(r2, params.smoothingRadiusSq, params.poly6Coeff);
                }
            }
        }
    }
    density = fmaxf(density, params.restDensity * 0.01f);
    densities[idx] = density;
    pressures[idx] = params.gasConstant * (density - params.restDensity);
}

void launchComputeHashes(
    const float3* positions, uint32_t* cellHashes, uint32_t* particleIndices,
    int particleCount, cudaStream_t stream
) {
    if (particleCount <= 0) return;
    SPHSimParams params; // Dummy since we are removing d_sphParams
    int block = 256;
    int grid = (particleCount + block - 1) / block;
    k_computeHashes<<<grid, block, 0, stream>>>(positions, cellHashes, particleIndices, particleCount, params);
    KERNEL_CHECK();
}

void launchFindCellBounds(
    uint32_t* cellStart, uint32_t* cellEnd, const uint32_t* sortedHashes,
    int particleCount, int totalCells, cudaStream_t stream
) {
    if (particleCount <= 0) return;
    CUDA_CHECK(cudaMemsetAsync(cellStart, 0xFF, totalCells * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(cellEnd, 0x00, totalCells * sizeof(uint32_t), stream));
    int block = 256;
    int grid = (particleCount + block - 1) / block;
    size_t smem = (block + 1) * sizeof(uint32_t);
    k_findCellBounds<<<grid, block, smem, stream>>>(cellStart, cellEnd, sortedHashes, particleCount);
    KERNEL_CHECK();
}

void launchComputeDensityPressure(
    const float3* sortedPositions, float* densities, float* pressures,
    const uint32_t* cellStart, const uint32_t* cellEnd, const uint32_t* sortedIndices,
    int particleCount, cudaStream_t stream
) {
    if (particleCount <= 0) return;
    SPHSimParams params; // Dummy
    int block = 256;
    int grid = (particleCount + block - 1) / block;
    k_computeDensityPressure<<<grid, block, 0, stream>>>(
        sortedPositions, densities, pressures, cellStart, cellEnd, particleCount, params
    );
    KERNEL_CHECK();
}