#ifndef GENESIS_SPH_BOUNDARY_CUH
#define GENESIS_SPH_BOUNDARY_CUH

#include <cuda_runtime.h>

namespace genesis {
namespace sph {

// Enforce world boundaries + terrain collision
void launchEnforceBoundaries(
    float3*       positions,
    float3*       velocities,
    const float*  heightmap,      // terrain heightmap [W x H]
    int           hmWidth,
    int           hmHeight,
    float         hmScaleXZ,      // world units per heightmap texel
    float         hmScaleY,       // height multiplier
    int           particleCount,
    cudaStream_t  stream = 0
);

} // namespace sph
} // namespace genesis

#endif