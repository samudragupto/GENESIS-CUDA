#include "sph_kernels.cuh"
#include "../core/cuda_utils.cuh"

void launchRenderFluidScreenSpace(
    const float3* d_positions,
    int num_particles,
    float* d_depth_buffer,
    float4* d_color_buffer,
    int screen_width,
    int screen_height,
    cudaStream_t stream
) {
    // Currently deferred to OpenGL/Rasterizer in actual render loop
    (void)d_positions; (void)num_particles; (void)d_depth_buffer;
    (void)d_color_buffer; (void)screen_width; (void)screen_height;
    (void)stream;
}