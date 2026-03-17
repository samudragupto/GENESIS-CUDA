#include "spatial_hash.cuh"
#include "sort_particles.cuh"
#include "../core/cuda_utils.cuh"
#include <algorithm>

__global__ void hashParticlesKernel(
    int* __restrict__ particle_hash,
    int* __restrict__ particle_index,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const int* __restrict__ alive,
    int num_particles,
    float cell_size,
    int grid_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    if (!alive || !alive[idx]) {
        particle_hash[idx] = grid_size * grid_size; 
        particle_index[idx] = idx;
        return;
    }

    int cx = min(max((int)(pos_x[idx] / cell_size), 0), grid_size - 1);
    int cy = min(max((int)(pos_y[idx] / cell_size), 0), grid_size - 1);
    
    particle_hash[idx] = cy * grid_size + cx;
    particle_index[idx] = idx;
}

__global__ void findCellBoundsKernel(
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    const int* __restrict__ particle_hash,
    const int* __restrict__ particle_index,
    int* __restrict__ sorted_indices,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    int hash = particle_hash[idx];
    sorted_indices[idx] = particle_index[idx];

    if (idx == 0) {
        cell_start[hash] = 0;
    } else {
        int prev_hash = particle_hash[idx - 1];
        if (hash != prev_hash) {
            cell_start[hash] = idx;
            if (prev_hash >= 0 && prev_hash < num_particles) {
                cell_end[prev_hash] = idx;
            }
        }
    }

    if (idx == num_particles - 1) {
        cell_end[hash] = num_particles;
    }
}

void initSpatialHashGrid(SpatialHashGrid& grid, int max_particles, int grid_size) {
    grid.max_particles = max_particles;
    grid.grid_size = grid_size;
    grid.cell_size = 4.0f;
    grid.total_cells = grid_size * grid_size + 1;

    CUDA_CHECK(cudaMalloc(&grid.d_cell_start, grid.total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.d_cell_end, grid.total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.d_particle_hash, max_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.d_particle_index, max_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid.d_sorted_indices, max_particles * sizeof(int)));
}

void freeSpatialHashGrid(SpatialHashGrid& grid) {
    cudaFree(grid.d_cell_start);
    cudaFree(grid.d_cell_end);
    cudaFree(grid.d_particle_hash);
    cudaFree(grid.d_particle_index);
    cudaFree(grid.d_sorted_indices);
}

void updateSpatialHashGrid(
    SpatialHashGrid& grid, const float* d_pos_x, const float* d_pos_y, 
    const int* d_alive, int num_particles, float world_size, cudaStream_t stream
) {
    if (num_particles <= 0) return;
    grid.cell_size = world_size / (float)grid.grid_size;
    
    CUDA_CHECK(cudaMemsetAsync(grid.d_cell_start, 0xFF, grid.total_cells * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(grid.d_cell_end, 0xFF, grid.total_cells * sizeof(int), stream));

    int block = 256;
    int grid_dim = (num_particles + block - 1) / block;

    hashParticlesKernel<<<grid_dim, block, 0, stream>>>(
        grid.d_particle_hash, grid.d_particle_index, d_pos_x, d_pos_y, 
        d_alive, num_particles, grid.cell_size, grid.grid_size
    );

    launchSortParticles(grid.d_particle_hash, grid.d_particle_index, num_particles, stream);

    findCellBoundsKernel<<<grid_dim, block, 0, stream>>>(
        grid.d_cell_start, grid.d_cell_end, grid.d_particle_hash, 
        grid.d_particle_index, grid.d_sorted_indices, num_particles
    );
}