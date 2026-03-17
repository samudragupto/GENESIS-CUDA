#ifndef CREATURE_RENDERER_CUH
#define CREATURE_RENDERER_CUH

#include <cuda_runtime.h>
#include "creature_common.cuh"

struct CreatureRenderData {
    RenderInstance* d_instances;
    int* d_visible_count;
    int max_visible;
};

void allocateCreatureRenderData(CreatureRenderData& rd, int max_visible);
void freeCreatureRenderData(CreatureRenderData& rd);

void launchCreatureFrustumCull(
    const CreatureData& creatures,
    CreatureRenderData& render_data,
    const float* d_heightmap,
    int world_size,
    float cam_x, float cam_y, float cam_z,
    float view_radius,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchBuildInstanceData(
    const CreatureData& creatures,
    CreatureRenderData& render_data,
    const float* d_heightmap,
    int world_size,
    int num_creatures,
    cudaStream_t stream = 0
);

#endif