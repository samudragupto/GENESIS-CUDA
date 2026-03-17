#include "terrain_renderer.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void allocateTerrainMesh(TerrainMeshData& mesh, int world_size, int lod) {
    mesh.world_size = world_size;
    mesh.lod_levels = lod;

    int step = 1;
    int verts_per_side = world_size / step;
    mesh.num_vertices = verts_per_side * verts_per_side;
    mesh.num_indices = (verts_per_side - 1) * (verts_per_side - 1) * 6;

    CUDA_CHECK(cudaMalloc(&mesh.d_vertices, mesh.num_vertices * sizeof(TerrainVertex)));
    CUDA_CHECK(cudaMalloc(&mesh.d_indices, mesh.num_indices * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(mesh.d_vertices, 0, mesh.num_vertices * sizeof(TerrainVertex)));
    CUDA_CHECK(cudaMemset(mesh.d_indices, 0, mesh.num_indices * sizeof(unsigned int)));
}

void freeTerrainMesh(TerrainMeshData& mesh) {
    cudaFree(mesh.d_vertices);
    cudaFree(mesh.d_indices);
    mesh.d_vertices = nullptr;
    mesh.d_indices = nullptr;
    mesh.num_vertices = 0;
    mesh.num_indices = 0;
}

__global__ void buildTerrainVerticesKernel(
    TerrainVertex* __restrict__ vertices,
    const float* __restrict__ heightmap,
    const float* __restrict__ normal_x,
    const float* __restrict__ normal_y,
    const float* __restrict__ normal_z,
    const int* __restrict__ biome_map,
    int world_size,
    int step,
    int verts_per_side
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    if (vx >= verts_per_side || vy >= verts_per_side) return;

    int vert_idx = vy * verts_per_side + vx;
    int wx = vx * step;
    int wy = vy * step;

    wx = min(wx, world_size - 1);
    wy = min(wy, world_size - 1);

    int world_idx = wy * world_size + wx;

    float h = heightmap[world_idx];

    vertices[vert_idx].x = (float)wx;
    vertices[vert_idx].y = h * 100.0f;
    vertices[vert_idx].z = (float)wy;

    if (normal_x && normal_y && normal_z) {
        vertices[vert_idx].nx = normal_x[world_idx];
        vertices[vert_idx].ny = normal_y[world_idx];
        vertices[vert_idx].nz = normal_z[world_idx];
    } else {
        float h_r = (wx + 1 < world_size) ? heightmap[wy * world_size + wx + 1] : h;
        float h_u = (wy + 1 < world_size) ? heightmap[(wy + 1) * world_size + wx] : h;
        float dx = (h_r - h) * 100.0f;
        float dz = (h_u - h) * 100.0f;
        float inv_len = rsqrtf(dx * dx + 1.0f + dz * dz);
        vertices[vert_idx].nx = -dx * inv_len;
        vertices[vert_idx].ny = inv_len;
        vertices[vert_idx].nz = -dz * inv_len;
    }

    vertices[vert_idx].u = (float)wx / (float)world_size;
    vertices[vert_idx].v = (float)wy / (float)world_size;

    if (biome_map) {
        vertices[vert_idx].biome_id = (float)biome_map[world_idx];
    } else {
        if (h < WATER_LEVEL) {
            vertices[vert_idx].biome_id = -1.0f;
        } else if (h < WATER_LEVEL + 0.05f) {
            vertices[vert_idx].biome_id = 0.0f;
        } else if (h < 0.6f) {
            vertices[vert_idx].biome_id = 1.0f;
        } else if (h < 0.8f) {
            vertices[vert_idx].biome_id = 4.0f;
        } else {
            vertices[vert_idx].biome_id = 5.0f;
        }
    }
}

__global__ void buildTerrainIndicesKernel(
    unsigned int* __restrict__ indices,
    int verts_per_side
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int quads_per_side = verts_per_side - 1;
    if (tx >= quads_per_side || ty >= quads_per_side) return;

    int quad_idx = ty * quads_per_side + tx;
    int base = quad_idx * 6;

    unsigned int tl = ty * verts_per_side + tx;
    unsigned int tr = tl + 1;
    unsigned int bl = (ty + 1) * verts_per_side + tx;
    unsigned int br = bl + 1;

    indices[base + 0] = tl;
    indices[base + 1] = bl;
    indices[base + 2] = tr;
    indices[base + 3] = tr;
    indices[base + 4] = bl;
    indices[base + 5] = br;
}

__global__ void lodSelectionKernel(
    int* __restrict__ lod_per_tile,
    float cam_x, float cam_y, float cam_z,
    float lod_dist_0, float lod_dist_1, float lod_dist_2,
    int world_size,
    int tile_size
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int tiles_per_side = world_size / tile_size;
    if (tx >= tiles_per_side || ty >= tiles_per_side) return;

    float tile_cx = (float)(tx * tile_size + tile_size / 2);
    float tile_cz = (float)(ty * tile_size + tile_size / 2);

    float dx = tile_cx - cam_x;
    float dz = tile_cz - cam_z;
    float dist = sqrtf(dx * dx + dz * dz);

    int lod;
    if (dist < lod_dist_0) {
        lod = 0;
    } else if (dist < lod_dist_1) {
        lod = 1;
    } else if (dist < lod_dist_2) {
        lod = 2;
    } else {
        lod = 3;
    }

    lod_per_tile[ty * tiles_per_side + tx] = lod;
}

__global__ void vegetationColorKernel(
    TerrainVertex* __restrict__ vertices,
    const float* __restrict__ vegetation,
    const float* __restrict__ heightmap,
    int world_size,
    float water_level,
    int verts_per_side,
    int step
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    if (vx >= verts_per_side || vy >= verts_per_side) return;

    int vert_idx = vy * verts_per_side + vx;
    int wx = min(vx * step, world_size - 1);
    int wy = min(vy * step, world_size - 1);
    int world_idx = wy * world_size + wx;

    float h = heightmap[world_idx];
    if (h < water_level) return;

    float veg = vegetation[world_idx];
    float green_boost = fminf(veg * 0.5f, 0.3f);

    if (vertices[vert_idx].biome_id >= 0.0f && vertices[vert_idx].biome_id <= 2.0f) {
        vertices[vert_idx].biome_id = 1.0f + green_boost;
    }
}

void launchBuildTerrainVertices(
    TerrainMeshData& mesh,
    const float* d_heightmap,
    const float* d_normalmap_x,
    const float* d_normalmap_y,
    const float* d_normalmap_z,
    const int* d_biome_map,
    int world_size,
    int step,
    cudaStream_t stream
) {
    int verts_per_side = world_size / step;
    mesh.num_vertices = verts_per_side * verts_per_side;

    dim3 block(16, 16);
    dim3 grid((verts_per_side + 15) / 16, (verts_per_side + 15) / 16);

    buildTerrainVerticesKernel<<<grid, block, 0, stream>>>(
        mesh.d_vertices, d_heightmap,
        d_normalmap_x, d_normalmap_y, d_normalmap_z,
        d_biome_map, world_size, step, verts_per_side
    );
}

void launchBuildTerrainIndices(
    TerrainMeshData& mesh,
    int world_size,
    int step,
    cudaStream_t stream
) {
    int verts_per_side = world_size / step;
    int quads_per_side = verts_per_side - 1;
    mesh.num_indices = quads_per_side * quads_per_side * 6;

    dim3 block(16, 16);
    dim3 grid((quads_per_side + 15) / 16, (quads_per_side + 15) / 16);

    buildTerrainIndicesKernel<<<grid, block, 0, stream>>>(
        mesh.d_indices, verts_per_side
    );
}

void launchTerrainLODSelection(
    TerrainMeshData& mesh,
    float cam_x, float cam_y, float cam_z,
    const TerrainRenderConfig& config,
    int world_size,
    int* d_lod_per_tile,
    int tile_size,
    cudaStream_t stream
) {
    int tiles_per_side = world_size / tile_size;
    dim3 block(16, 16);
    dim3 grid((tiles_per_side + 15) / 16, (tiles_per_side + 15) / 16);

    lodSelectionKernel<<<grid, block, 0, stream>>>(
        d_lod_per_tile, cam_x, cam_y, cam_z,
        config.lod_distance_0, config.lod_distance_1, config.lod_distance_2,
        world_size, tile_size
    );
}

void launchTerrainColorFromVegetation(
    TerrainMeshData& mesh,
    const float* d_vegetation,
    const float* d_heightmap,
    int world_size,
    float water_level,
    cudaStream_t stream
) {
    int step = max(1, world_size / (int)sqrtf((float)mesh.num_vertices));
    int verts_per_side = world_size / step;

    dim3 block(16, 16);
    dim3 grid((verts_per_side + 15) / 16, (verts_per_side + 15) / 16);

    vegetationColorKernel<<<grid, block, 0, stream>>>(
        mesh.d_vertices, d_vegetation, d_heightmap,
        world_size, water_level, verts_per_side, step
    );
}