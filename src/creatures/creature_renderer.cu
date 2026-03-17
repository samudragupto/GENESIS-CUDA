#include "creature_renderer.cuh"
#include "../core/constants.cuh"
#include "../core/cuda_utils.cuh"

void allocateCreatureRenderData(CreatureRenderData& rd, int max_visible) {
    rd.max_visible = max_visible;
    CUDA_CHECK(cudaMalloc(&rd.d_instances, max_visible * sizeof(RenderInstance)));
    CUDA_CHECK(cudaMalloc(&rd.d_visible_count, sizeof(int)));
}

void freeCreatureRenderData(CreatureRenderData& rd) {
    cudaFree(rd.d_instances);
    cudaFree(rd.d_visible_count);
}

__device__ float3 speciesColor(int species_id, const float* genome) {
    float r = genome[GENE_COLOR_R];
    float g = genome[GENE_COLOR_G];
    float b = genome[GENE_COLOR_B];
    float diet = genome[GENE_DIET];
    r = r * 0.7f + diet * 0.3f;
    b = b * 0.7f + (1.0f - diet) * 0.3f;
    return make_float3(r, g, b);
}

__device__ float sampleTerrainHeight(
    const float* __restrict__ heightmap,
    float px, float py,
    int world_size
) {
    if (!heightmap) return 0.0f;
    int gx = min(max((int)px, 0), world_size - 1);
    int gy = min(max((int)py, 0), world_size - 1);
    return heightmap[gy * world_size + gx];
}

__global__ void buildInstanceKernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ vel_x,
    const float* __restrict__ vel_y,
    const float* __restrict__ genomes,
    const int*   __restrict__ alive,
    const int*   __restrict__ species_id,
    const float* __restrict__ heightmap,
    int world_size,
    float cam_x, float cam_y, float cam_z,
    float view_radius_sq,
    RenderInstance* __restrict__ instances,
    int* __restrict__ visible_count,
    int max_visible,
    int num_creatures,
    int has_heightmap
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    float px = pos_x[idx];
    float py = pos_y[idx];

    float dx = px - cam_x;
    float dy = py - cam_y;
    float dist2 = dx * dx + dy * dy;

    if (dist2 > view_radius_sq) return;

    int vis_idx = atomicAdd(visible_count, 1);
    if (vis_idx >= max_visible) return;

    float h = 0.0f;
    if (has_heightmap) {
        h = sampleTerrainHeight(heightmap, px, py, world_size);
    }

    const float* genome = genomes + idx * GENOME_SIZE;
    float body_size = 0.5f + genome[GENE_SIZE] * 2.0f;
    float3 color = speciesColor(species_id[idx], genome);

    float vx = vel_x[idx];
    float vy = vel_y[idx];
    float rotation = atan2f(vy, vx);

    instances[vis_idx].x = px;
    instances[vis_idx].y = h * 100.0f + body_size * 0.5f;
    instances[vis_idx].z = py;
    instances[vis_idx].r = color.x;
    instances[vis_idx].g = color.y;
    instances[vis_idx].b = color.z;
    instances[vis_idx].scale = body_size;
    instances[vis_idx].rotation = rotation;
}

void launchBuildInstanceData(
    const CreatureData& creatures,
    CreatureRenderData& render_data,
    const float* d_heightmap,
    int world_size,
    int num_creatures,
    cudaStream_t stream
) {
    float cam_x = (float)world_size * 0.5f;
    float cam_y = (float)world_size * 0.5f;
    float cam_z = 50.0f;
    float view_radius = 500.0f;

    launchCreatureFrustumCull(
        creatures, render_data,
        d_heightmap, world_size,
        cam_x, cam_y, cam_z,
        view_radius, num_creatures, stream
    );
}

void launchCreatureFrustumCull(
    const CreatureData& creatures,
    CreatureRenderData& render_data,
    const float* d_heightmap,
    int world_size,
    float cam_x, float cam_y, float cam_z,
    float view_radius,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(render_data.d_visible_count, 0, sizeof(int), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    float view_radius_sq = view_radius * view_radius;
    int has_heightmap = (d_heightmap != nullptr) ? 1 : 0;

    buildInstanceKernel<<<grid, block, 0, stream>>>(
        creatures.d_pos_x, creatures.d_pos_y,
        creatures.d_vel_x, creatures.d_vel_y,
        creatures.d_genomes, creatures.d_alive,
        creatures.d_species_id, d_heightmap,
        world_size, cam_x, cam_y, cam_z,
        view_radius_sq, render_data.d_instances,
        render_data.d_visible_count,
        render_data.max_visible, num_creatures,
        has_heightmap
    );
}