#ifndef TERRAIN_RENDERER_CUH
#define TERRAIN_RENDERER_CUH

#include <cuda_runtime.h>

struct TerrainVertex {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
    float biome_id;
};

struct TerrainMeshData {
    TerrainVertex* d_vertices;
    unsigned int* d_indices;
    int num_vertices;
    int num_indices;
    int world_size;
    int lod_levels;
};

struct TerrainRenderConfig {
    float texel_scale;
    float height_scale;
    int wireframe;
    int lod_enabled;
    float lod_distance_0;
    float lod_distance_1;
    float lod_distance_2;
    float water_level;
};

void allocateTerrainMesh(TerrainMeshData& mesh, int world_size, int lod);
void freeTerrainMesh(TerrainMeshData& mesh);

void launchBuildTerrainVertices(
    TerrainMeshData& mesh,
    const float* d_heightmap,
    const float* d_normalmap_x,
    const float* d_normalmap_y,
    const float* d_normalmap_z,
    const int* d_biome_map,
    int world_size,
    int step,
    cudaStream_t stream = 0
);

void launchBuildTerrainIndices(
    TerrainMeshData& mesh,
    int world_size,
    int step,
    cudaStream_t stream = 0
);

void launchTerrainLODSelection(
    TerrainMeshData& mesh,
    float cam_x, float cam_y, float cam_z,
    const TerrainRenderConfig& config,
    int world_size,
    int* d_lod_per_tile,
    int tile_size,
    cudaStream_t stream = 0
);

void launchTerrainColorFromVegetation(
    TerrainMeshData& mesh,
    const float* d_vegetation,
    const float* d_heightmap,
    int world_size,
    float water_level,
    cudaStream_t stream = 0
);

#endif