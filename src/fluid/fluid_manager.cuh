#ifndef FLUID_MANAGER_CUH
#define FLUID_MANAGER_CUH

#include <cuda_runtime.h>
#include "../core/constants.cuh"

struct FluidParticleData {
    float* d_pos_x;
    float* d_pos_y;
    float* d_pos_z;
    float* d_vel_x;
    float* d_vel_y;
    float* d_vel_z;
    float* d_density;
    float* d_pressure;
    float* d_force_x;
    float* d_force_y;
    float* d_force_z;
    int*   d_cell_hash;
    int*   d_sorted_index;
    int*   d_cell_start;
    int*   d_cell_end;
    int    max_particles;
    int    num_particles;
};

struct SPHParams {
    float smoothing_length;
    float rest_density;
    float gas_constant;
    float viscosity;
    float surface_tension;
    float gravity;
    float dt;
    float particle_mass;
    float world_size_x;
    float world_size_y;
    float world_size_z;
    int   grid_size;
};

class FluidManager {
public:
    FluidParticleData particles;
    SPHParams sph_params;
    cudaStream_t stream_sph;
    int enabled;

    void init(int max_particles, float world_size);
    void destroy();
    void update(float dt, const float* d_heightmap, int world_size);
    void spawnParticles(int count, float x_min, float x_max,
                        float y_min, float y_max, float z_min, float z_max);
    int getParticleCount() const;
};

#endif