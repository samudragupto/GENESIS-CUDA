#ifndef GENESIS_SPH_DENSITY_CUH
#define GENESIS_SPH_DENSITY_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace genesis {
namespace sph {

/*******************************************************************************
 * SPH Density & Pressure Computation
 *
 * Kernel workflow:
 *   1. computeHashes     → hash each particle to grid cell
 *   2. (external sort)   → sort particles by hash (thrust)
 *   3. findCellBounds    → detect start/end of each cell in sorted array
 *   4. computeDensity    → sum Poly6 kernel over neighbors
 *   5. computePressure   → equation of state P = k(ρ - ρ₀)
 ******************************************************************************/

// Launch hash computation: 1 thread per particle
void launchComputeHashes(
    const float3* positions,
    uint32_t*     cellHashes,
    uint32_t*     particleIndices,
    int           particleCount,
    cudaStream_t  stream = 0
);

// Launch cell boundary detection: 1 thread per particle
void launchFindCellBounds(
    uint32_t*     cellStart,
    uint32_t*     cellEnd,
    const uint32_t* sortedHashes,
    int           particleCount,
    int           totalCells,
    cudaStream_t  stream = 0
);

// Launch density computation: 1 thread per particle, shared memory tiling
void launchComputeDensityPressure(
    const float3*   sortedPositions,
    float*          densities,
    float*          pressures,
    const uint32_t* cellStart,
    const uint32_t* cellEnd,
    const uint32_t* sortedIndices,
    int             particleCount,
    cudaStream_t    stream = 0
);

} // namespace sph
} // namespace genesis

#endif