#ifndef GENESIS_FLUID_TERRAIN_COUPLING_CUH
#define GENESIS_FLUID_TERRAIN_COUPLING_CUH

#include <cuda_runtime.h>

namespace genesis {
namespace sph {

struct TerrainCouplingParams {
    float erosionRate;           // sediment pickup rate
    float depositionRate;        // sediment deposit rate  
    float sedimentCapacityFactor;// carrying capacity multiplier
    float minErosionVelocity;   // minimum velocity for erosion
    float evaporationRate;       // water→atmosphere transfer
    float absorptionRate;        // water→terrain absorption
};

// Apply fluid erosion/deposition to terrain heightmap
void launchFluidErosion(
    const float3* particlePositions,
    const float3* particleVelocities,
    const float*  particleDensities,
    float*        heightmap,          // modified in-place
    float*        moistureMap,        // modified in-place
    float*        sedimentCarried,    // per-particle sediment load
    int           particleCount,
    int           hmWidth,
    int           hmHeight,
    float         hmScaleXZ,
    float         hmScaleY,
    const TerrainCouplingParams& params,
    cudaStream_t  stream = 0
);

// Evaporate fluid particles near surface, spawn new ones from rain
void launchEvaporationAbsorption(
    float3*       positions,
    float3*       velocities,
    int*          particleAlive,     // 0 = dead, 1 = alive
    const float*  temperatureMap,    // from climate system
    int           particleCount,
    int           hmWidth,
    int           hmHeight,
    float         hmScaleXZ,
    cudaStream_t  stream = 0
);

} // namespace sph
} // namespace genesis

#endif