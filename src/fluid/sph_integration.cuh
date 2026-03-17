#ifndef GENESIS_SPH_INTEGRATION_CUH
#define GENESIS_SPH_INTEGRATION_CUH

#include <cuda_runtime.h>

namespace genesis {
namespace sph {

// Leapfrog integration: v(t+dt/2) = v(t-dt/2) + a(t)*dt,  x(t+dt) = x(t) + v(t+dt/2)*dt
void launchIntegrate(
    float3*       positions,
    float3*       velocities,
    const float3* forces,
    const float*  densities,
    int           particleCount,
    float         deltaTime,
    bool          firstStep,      // use Euler for first half-step
    cudaStream_t  stream = 0
);

// Velocity clamping to prevent explosion
void launchClampVelocities(
    float3*      velocities,
    int          particleCount,
    float        maxSpeed,
    cudaStream_t stream = 0
);

} // namespace sph
} // namespace genesis

#endif