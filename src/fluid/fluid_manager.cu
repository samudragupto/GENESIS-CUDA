#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "fluid_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include "../fluid/sph_kernels.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <cstdio>

void FluidManager::init(int max_particles, float world_size) {
    enabled = 0;
    particles.max_particles = max_particles;
    particles.num_particles = 0;

    CUDA_CHECK(cudaMalloc(&particles.d_pos_x, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_pos_y, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_pos_z, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_vel_x, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_vel_y, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_vel_z, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_density, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_pressure, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_force_x, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_force_y, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_force_z, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&particles.d_cell_hash, max_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&particles.d_sorted_index, max_particles * sizeof(int)));

    int grid_cells = 256 * 256;
    CUDA_CHECK(cudaMalloc(&particles.d_cell_start, grid_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&particles.d_cell_end, grid_cells * sizeof(int)));

    CUDA_CHECK(cudaMemset(particles.d_pos_x, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_pos_y, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_pos_z, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_vel_x, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_vel_y, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_vel_z, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_density, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_pressure, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_force_x, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_force_y, 0, max_particles * sizeof(float)));
    CUDA_CHECK(cudaMemset(particles.d_force_z, 0, max_particles * sizeof(float)));

    sph_params.smoothing_length = 0.04f;
    sph_params.rest_density = 1000.0f;
    sph_params.gas_constant = 2000.0f;
    sph_params.viscosity = 0.01f;
    sph_params.surface_tension = 0.0728f;
    sph_params.gravity = -9.81f;
    sph_params.dt = 0.001f;
    sph_params.particle_mass = 0.02f;
    sph_params.world_size_x = world_size;
    sph_params.world_size_y = world_size;
    sph_params.world_size_z = 100.0f;
    sph_params.grid_size = 256;

    CUDA_CHECK(cudaStreamCreate(&stream_sph));
}

void FluidManager::destroy() {
    cudaFree(particles.d_pos_x);
    cudaFree(particles.d_pos_y);
    cudaFree(particles.d_pos_z);
    cudaFree(particles.d_vel_x);
    cudaFree(particles.d_vel_y);
    cudaFree(particles.d_vel_z);
    cudaFree(particles.d_density);
    cudaFree(particles.d_pressure);
    cudaFree(particles.d_force_x);
    cudaFree(particles.d_force_y);
    cudaFree(particles.d_force_z);
    cudaFree(particles.d_cell_hash);
    cudaFree(particles.d_sorted_index);
    cudaFree(particles.d_cell_start);
    cudaFree(particles.d_cell_end);
    cudaStreamDestroy(stream_sph);
}

void FluidManager::update(float dt, const float* d_heightmap, int world_size) {
    if (!enabled || particles.num_particles <= 0) return;
    (void)dt;
    (void)d_heightmap;
    (void)world_size;
}

void FluidManager::spawnParticles(int count, float x_min, float x_max,
    float y_min, float y_max, float z_min, float z_max) {
    (void)count;
    (void)x_min; (void)x_max;
    (void)y_min; (void)y_max;
    (void)z_min; (void)z_max;
}

int FluidManager::getParticleCount() const {
    return particles.num_particles;
}