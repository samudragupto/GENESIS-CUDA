#ifndef SPH_KERNELS_CUH
#define SPH_KERNELS_CUH

#include <cuda_runtime.h>
#include <cstdint>

#ifndef SPH_PI
#define SPH_PI 3.14159265358979323846f
#endif

struct SPHSimParams {
    float smoothingRadius;
    float smoothingRadiusSq;
    float particleMass;
    float poly6Coeff;
    float spikyGradCoeff;
    float viscLapCoeff;
    float gasConstant;
    float restDensity;
    float viscosity;
    float gravity;
    float dt;
    float cellSize;
    int gridDimX;
    int gridDimY;
    int gridDimZ;
    int totalCells;
    float worldMinX, worldMinY, worldMinZ;
    float worldMaxX, worldMaxY, worldMaxZ;
};

__device__ __forceinline__ int3 getCell(float3 pos, const SPHSimParams& params) {
    return make_int3(
        (int)((pos.x - params.worldMinX) / params.cellSize),
        (int)((pos.y - params.worldMinY) / params.cellSize),
        (int)((pos.z - params.worldMinZ) / params.cellSize)
    );
}

__device__ __forceinline__ uint32_t hashCell(int3 cell, const SPHSimParams& params) {
    return (uint32_t)(
        (cell.z * params.gridDimY + cell.y) * params.gridDimX + cell.x
    ) % (uint32_t)params.totalCells;
}

__device__ __forceinline__ float kernelPoly6(float r2, float h2, float coeff) {
    if (r2 >= h2) return 0.0f;
    float diff = h2 - r2;
    return coeff * diff * diff * diff;
}

__device__ __forceinline__ float kernelSpikyGrad(float r, float h, float coeff) {
    if (r >= h || r < 0.001f) return 0.0f;
    float diff = h - r;
    return coeff * diff * diff / r;
}

__device__ __forceinline__ float kernelViscLap(float r, float h, float coeff) {
    if (r >= h) return 0.0f;
    return coeff * (h - r);
}

#endif