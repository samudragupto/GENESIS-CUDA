#ifndef GENESIS_SPH_FORCES_CUH
#define GENESIS_SPH_FORCES_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace genesis {
namespace sph {

// Launch pressure + viscosity + surface tension + gravity force computation
void launchComputeForces(
    const float3*   sortedPositions,
    const float3*   sortedVelocities,
    const float*    densities,
    const float*    pressures,
    float3*         forces,
    float3*         colorFieldNormals,   // for surface tension
    float*          colorFieldLaplacian, // for surface tension
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    int             particleCount,
    cudaStream_t    stream = 0
);

// Separate surface tension kernel (needs color field from first pass)
void launchComputeSurfaceTension(
    float3*         forces,
    const float3*   colorFieldNormals,
    const float*    colorFieldLaplacian,
    const float*    densities,
    int             particleCount,
    cudaStream_t    stream = 0
);

} // namespace sph
} // namespace genesis

#endif