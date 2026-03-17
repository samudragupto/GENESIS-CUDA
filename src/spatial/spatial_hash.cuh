#ifndef SPATIAL_HASH_CUH
#define SPATIAL_HASH_CUH

#include <cuda_runtime.h>
#include "../core/constants.cuh"

struct SpatialHashGrid {
    int* d_cell_start;
    int* d_cell_end;
    int* d_particle_hash;
    int* d_particle_index;
    int* d_sorted_indices;
    int  grid_size;
    int  max_particles;
    float cell_size;
    int  total_cells;
};

void initSpatialHashGrid(SpatialHashGrid& grid, int max_particles, int grid_size);
void freeSpatialHashGrid(SpatialHashGrid& grid);

void updateSpatialHashGrid(
    SpatialHashGrid& grid,
    const float* d_pos_x,
    const float* d_pos_y,
    const int* d_alive,
    int num_particles,
    float world_size,
    cudaStream_t stream = 0
);

#endif