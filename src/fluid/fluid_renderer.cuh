#ifndef GENESIS_FLUID_RENDERER_CUH
#define GENESIS_FLUID_RENDERER_CUH

#include <cuda_runtime.h>

namespace genesis {
namespace sph {

struct FluidRenderBuffers {
    float*  depthBuffer;        // screen-space depth per pixel
    float*  thicknessBuffer;    // accumulated fluid thickness
    float3* normalBuffer;       // reconstructed surface normals
    float4* colorBuffer;        // final fluid color RGBA
    int     screenWidth;
    int     screenHeight;
};

// Splat particles to depth buffer (screen-space method)
void launchSplatParticles(
    const float3*       particlePositions,
    const float*        particleDensities,
    int                 particleCount,
    FluidRenderBuffers& renderBuffers,
    const float*        viewMatrix,       // 4x4 column-major
    const float*        projMatrix,       // 4x4 column-major
    float               pointRadius,
    cudaStream_t        stream = 0
);

// Bilateral filter to smooth depth buffer
void launchBilateralFilter(
    float*       depthOut,
    const float* depthIn,
    int          width,
    int          height,
    int          filterRadius,
    float        spatialSigma,
    float        depthSigma,
    cudaStream_t stream = 0
);

// Reconstruct normals from smoothed depth buffer
void launchComputeNormalsFromDepth(
    float3*      normals,
    const float* depth,
    int          width,
    int          height,
    const float* invProjMatrix,
    cudaStream_t stream = 0
);

} // namespace sph
} // namespace genesis

#endif